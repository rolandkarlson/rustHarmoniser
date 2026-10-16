use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use rustnote_core::model::{Config, Note};
use rustnote_core::render::Leading;
use serde::Deserialize;
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};

use crate::ableton;
use crate::output::{
    default_output_settings, plan_clips, run_render, validate_output_settings, ClipPlan,
    ClipSummary, OutputSettings,
};

#[derive(Clone, Default)]
struct AppState {
    session: Arc<Mutex<RenderSession>>,
}

#[derive(Default)]
struct RenderSession {
    last: Option<LastRender>,
}

struct LastRender {
    config: Config,
    notes: Vec<Note>,
    output: OutputSettings,
    clips: Result<Vec<ClipPlan>, String>,
}

#[derive(Deserialize)]
struct GenerateEnvelope {
    config: Config,
    #[serde(default)]
    output: Option<OutputSettings>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum GenerateRequest {
    Envelope(GenerateEnvelope),
    Legacy(Config),
}

impl GenerateRequest {
    fn into_parts(self) -> (Config, Option<OutputSettings>) {
        match self {
            Self::Envelope(request) => (request.config, request.output),
            Self::Legacy(config) => (config, None),
        }
    }
}

#[derive(Deserialize)]
struct OutputRequest {
    output: OutputSettings,
}

pub async fn start_server() {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_headers(Any)
        .allow_methods(Any);
    let state = AppState::default();
    let app = Router::new()
        .route("/api/config", get(get_config))
        .route("/api/generate", post(generate_music))
        .route("/api/clip-notes/:track/:clip", get(get_clip_notes))
        .route("/api/last-chord/:track/:clip", get(get_last_chord))
        .route(
            "/api/output/preview",
            get(get_output_preview).post(preview_output),
        )
        .route("/api/output/export", post(export_output))
        .route("/api/output/session", get(get_output_session))
        .route("/api/ableton/session", get(get_output_session))
        .with_state(state)
        .layer(cors);

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("API Server running at http://{}", addr);
    let _ = webbrowser::open("http://localhost:5173");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn get_config() -> impl IntoResponse {
    let mut config = Config::default();
    config.init_contours();
    Json(config)
}

async fn get_clip_notes(Path((track, clip)): Path<(i32, i32)>) -> impl IntoResponse {
    match ableton::get_clip_notes(track, clip).await {
        Ok(notes) => (StatusCode::OK, Json(notes)).into_response(),
        Err(e) => bridge_error(e),
    }
}

async fn get_last_chord(Path((track, clip)): Path<(i32, i32)>) -> impl IntoResponse {
    match ableton::get_last_chord(track, clip).await {
        Ok(notes) => (StatusCode::OK, Json(json!({ "notes": notes }))).into_response(),
        Err(e) => bridge_error(e),
    }
}

async fn get_output_session() -> impl IntoResponse {
    match ableton::get_output_session().await {
        Ok(session) => (StatusCode::OK, Json(session)).into_response(),
        Err(e) => bridge_error(e),
    }
}

async fn generate_music(
    State(state): State<AppState>,
    Json(request): Json<GenerateRequest>,
) -> impl IntoResponse {
    // A single session lock serializes render/cache/export decisions. That
    // prevents an explicit Send from racing a newer Generate into the cache.
    let mut session = state.session.lock().await;
    let (config, requested_output) = request.into_parts();
    let output = requested_output.unwrap_or_else(|| default_output_settings(&config));

    // When automatic export is enabled, reject route mistakes before the CPU
    // render and before any Live request. Disabled output may stay temporarily
    // invalid while the user changes mode or voice count.
    if output.enabled {
        if let Err(e) = validate_output_settings(&config, &output, true) {
            return bad_request(e.to_string());
        }
    }

    let leading = match fetch_leading(&config).await {
        Ok(value) => value,
        Err(e) => return bridge_error(e),
    };
    let render_config = config.clone();
    let render_leading = leading.clone();
    let render_output = output.clone();
    let handle = tokio::task::spawn_blocking(move || {
        run_render(&render_config, render_leading.as_ref(), Some(render_output))
    });
    let completed = match handle.await {
        Ok(Ok(completed)) => completed,
        Ok(Err(e)) => {
            let status = if e.kind() == std::io::ErrorKind::InvalidInput {
                StatusCode::BAD_REQUEST
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            return (
                status,
                Json(json!({"status":"error", "message":e.to_string()})),
            )
                .into_response();
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"status":"error", "message":format!("render task failed: {e}")})),
            )
                .into_response()
        }
    };

    session.last = Some(LastRender {
        config: config.clone(),
        notes: completed.result.notes.clone(),
        output: completed.output.clone(),
        clips: completed.clips.clone(),
    });

    // Cache and archive already exist at this point. A Live failure is reported
    // as an export result and never erases the successful musical render.
    let export = maybe_export(&config, &completed.output, &completed.clips).await;
    let preview = preview_json(&config, &completed.output, &completed.clips);
    (
        StatusCode::OK,
        Json(json!({
            "status": "success",
            "message": completed.message,
            "selected_scale_modes": completed.result.selected_scale_modes,
            "breakdown": completed.result.breakdown,
            "output": { "settings": completed.output, "preview": preview, "export": export },
        })),
    )
        .into_response()
}

async fn fetch_leading(config: &Config) -> std::io::Result<Option<Leading>> {
    if !config.use_leading_voice {
        return Ok(None);
    }
    let clip = ableton::get_clip(config.leading_voice_track, config.leading_voice_clip).await?;
    let pattern = clip
        .notes
        .iter()
        .map(|n| Note {
            pitch: n.pitch,
            start: n.start_time,
            duration: n.duration,
            velocity: n.velocity.round() as i32,
            muted: i32::from(n.mute),
            channel: 0,
            probability: (n.probability * 100.0).round() as i32,
        })
        .collect();
    Ok(Some(Leading {
        notes: pattern,
        clip_length: clip.length,
    }))
}

async fn maybe_export(
    config: &Config,
    output: &OutputSettings,
    clips: &Result<Vec<ClipPlan>, String>,
) -> Value {
    if !output.enabled {
        return json!({"status":"disabled"});
    }
    let clips = match clips {
        Ok(clips) => clips,
        Err(message) => return json!({"status":"failed", "message":message, "uncertain":false}),
    };
    export_clips(config, clips).await
}

async fn export_clips(config: &Config, clips: &[ClipPlan]) -> Value {
    let source = config
        .use_leading_voice
        .then_some((config.leading_voice_track, config.leading_voice_clip));
    match ableton::write_harmonizer_clips(clips, source).await {
        Ok(result) => {
            let complete = result
                .get("complete")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            json!({"status": if complete {"success"} else {"partial"}, "result":result, "uncertain":false})
        }
        Err(e) => {
            let message = e.to_string();
            json!({"status":"failed", "uncertain":message.contains("outcome is uncertain"), "message":message})
        }
    }
}

async fn get_output_preview(State(state): State<AppState>) -> impl IntoResponse {
    let session = state.session.lock().await;
    match session.last.as_ref() {
        Some(last) => (
            StatusCode::OK,
            Json(preview_json(&last.config, &last.output, &last.clips)),
        )
            .into_response(),
        None => no_render(),
    }
}

async fn preview_output(
    State(state): State<AppState>,
    Json(request): Json<OutputRequest>,
) -> impl IntoResponse {
    let mut session = state.session.lock().await;
    let Some(last) = session.last.as_mut() else {
        return no_render();
    };
    let clips = plan_clips(&last.notes, &last.config, &request.output).map_err(|e| e.to_string());
    last.output = request.output;
    last.clips = clips;
    match &last.clips {
        Ok(_) => (
            StatusCode::OK,
            Json(preview_json(&last.config, &last.output, &last.clips)),
        )
            .into_response(),
        Err(message) => bad_request(message.clone()),
    }
}

async fn export_output(
    State(state): State<AppState>,
    Json(request): Json<OutputRequest>,
) -> impl IntoResponse {
    let mut session = state.session.lock().await;
    let Some(last) = session.last.as_mut() else {
        return no_render();
    };
    if let Err(e) = validate_output_settings(&last.config, &request.output, true) {
        return bad_request(e.to_string());
    }
    let clips = match plan_clips(&last.notes, &last.config, &request.output) {
        Ok(clips) => clips,
        Err(e) => return bad_request(e.to_string()),
    };
    last.output = request.output;
    last.clips = Ok(clips.clone());
    let export = export_clips(&last.config, &clips).await;
    let status = match export.get("status").and_then(Value::as_str) {
        Some("success" | "partial") => StatusCode::OK,
        _ => StatusCode::BAD_GATEWAY,
    };
    let mut body = export;
    if let Some(object) = body.as_object_mut() {
        object.insert(
            "preview".into(),
            preview_json(&last.config, &last.output, &last.clips),
        );
    }
    (status, Json(body)).into_response()
}

fn preview_json(
    config: &Config,
    output: &OutputSettings,
    clips: &Result<Vec<ClipPlan>, String>,
) -> Value {
    let voice_count = if config.fugue_mode {
        config.fugue_voice_count
    } else {
        5
    };
    let length = i64::from(config.pl)
        .checked_mul(4)
        .and_then(|v| v.checked_mul(i64::from(config.render_length)));
    let source = json!({
        "voice_count": voice_count,
        "length": length,
        "lead_clip": config.use_leading_voice.then(|| json!({
            "track_index": config.leading_voice_track,
            "clip_index": config.leading_voice_clip,
        })),
    });
    match clips {
        Ok(clips) => {
            json!({"settings":output, "source":source, "clips":clips.iter().map(ClipSummary::from).collect::<Vec<_>>(), "error":null})
        }
        Err(message) => json!({"settings":output, "source":source, "clips":[], "error":message}),
    }
}

fn bad_request(message: impl Into<String>) -> axum::response::Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"status":"error", "message":message.into()})),
    )
        .into_response()
}

fn bridge_error(error: std::io::Error) -> axum::response::Response {
    (
        StatusCode::BAD_GATEWAY,
        Json(json!({"status":"error", "message":error.to_string()})),
    )
        .into_response()
}

fn no_render() -> axum::response::Response {
    (StatusCode::NOT_FOUND, Json(json!({"status":"error", "message":"Generate music first; there is no cached output to preview or export."}))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_request_accepts_legacy_config_and_new_envelope() {
        let config = Config::default();
        let bare: GenerateRequest =
            serde_json::from_value(serde_json::to_value(&config).unwrap()).unwrap();
        assert!(matches!(bare, GenerateRequest::Legacy(_)));
        let envelope: GenerateRequest =
            serde_json::from_value(json!({"config":config,"output":{"enabled":false,"routes":[]}}))
                .unwrap();
        assert!(matches!(envelope, GenerateRequest::Envelope(_)));
    }

    #[tokio::test]
    async fn disabled_output_never_connects_to_live() {
        let value = maybe_export(
            &Config::default(),
            &OutputSettings::default(),
            &Err("bad route".into()),
        )
        .await;
        assert_eq!(value["status"], "disabled");
    }

    #[test]
    fn explicit_export_rejects_misrouting_before_live() {
        let mut config = Config::default();
        config.use_leading_voice = true;
        config.leading_voice_track = 4;
        config.leading_voice_clip = 2;
        let output = OutputSettings {
            enabled: false,
            routes: vec![crate::output::OutputRoute::default()],
        };
        assert!(validate_output_settings(&config, &output, true)
            .unwrap_err()
            .to_string()
            .contains("Lead Clip"));
    }
}
