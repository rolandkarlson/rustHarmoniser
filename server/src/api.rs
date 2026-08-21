use axum::{
    extract::Path,
    routing::{get, post},
    Router, Json, response::{IntoResponse}, http::StatusCode,
};
use tower_http::cors::{Any, CorsLayer};
use std::net::SocketAddr;
use crate::ableton;
use crate::output::run_render;
use rustnote_core::model::{Config, Note};
use rustnote_core::render::Leading;

pub async fn start_server() {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_headers(Any)
        .allow_methods(Any);

    let app = Router::new()
        .route("/api/config", get(get_config))
        .route("/api/generate", post(generate_music))
        .route("/api/clip-notes/:track/:clip", get(get_clip_notes))
        .route("/api/last-chord/:track/:clip", get(get_last_chord))
        .layer(cors);

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("API Server running at http://{}", addr);

    // Open standard vite dev port
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
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({ "status": "error", "message": e.to_string() })),
        )
            .into_response(),
    }
}

async fn get_last_chord(Path((track, clip)): Path<(i32, i32)>) -> impl IntoResponse {
    match ableton::get_last_chord(track, clip).await {
        Ok(notes) => (StatusCode::OK, Json(serde_json::json!({ "notes": notes }))).into_response(),
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({ "status": "error", "message": e.to_string() })),
        )
            .into_response(),
    }
}

async fn generate_music(Json(config): Json<Config>) -> impl IntoResponse {
    let leading = if config.use_leading_voice {
        match ableton::get_clip(config.leading_voice_track, config.leading_voice_clip).await {
            Ok(clip) => {
                let pattern: Vec<Note> = clip.notes.iter().map(|n| Note {
                    pitch: n.pitch,
                    start: n.start_time,
                    duration: n.duration,
                    velocity: n.velocity.round() as i32,
                    muted: 0,
                    channel: 0,
                    probability: (n.probability * 100.0).round() as i32,
                }).collect();
                Some(Leading { notes: pattern, clip_length: clip.length })
            }
            Err(e) => {
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(serde_json::json!({ "status": "error", "message": format!("ableton fetch failed: {e}") })),
                ).into_response();
            }
        }
    } else {
        None
    };

    // The render is CPU-bound and can take a while at high beam widths; keep it
    // off the async worker threads.
    let handle = tokio::task::spawn_blocking(move || {
        run_render(&config, leading.as_ref())
    });
    match handle.await {
        Ok(Ok((msg, result))) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "success",
                "message": msg,
                // The per-chord score breakdown, so the GUI (or the browser's
                // network tab) can inspect why each voicing won.
                "breakdown": result.breakdown,
            })),
        ).into_response(),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "status": "error", "message": e.to_string() })),
        ).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "status": "error", "message": format!("render task failed: {e}") })),
        ).into_response(),
    }
}
