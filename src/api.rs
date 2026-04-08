use axum::{
    routing::{get, post},
    Router, Json, response::{IntoResponse}, http::StatusCode,
};
use tower_http::cors::{Any, CorsLayer};
use std::net::SocketAddr;
use crate::model::Config;
use crate::run_generation;

pub async fn start_server() {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_headers(Any)
        .allow_methods(Any);

    let app = Router::new()
        .route("/api/config", get(get_config))
        .route("/api/generate", post(generate_music))
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

async fn generate_music(Json(config): Json<Config>) -> impl IntoResponse {
    // Note: Since this is blocking logic, we should ideally use spawn_blocking but it's fine for local tools.
    match run_generation(&config, None) {
        Ok(msg) => (StatusCode::OK, Json(serde_json::json!({ "status": "success", "message": msg }))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "status": "error", "message": e.to_string() }))).into_response(),
    }
}
