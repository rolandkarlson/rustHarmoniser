use serde::{Deserialize, Serialize};
use serde_json::json;
use std::io;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::timeout;

const HOST: &str = "localhost";
const PORT: u16 = 9877;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbletonNote {
    pub pitch: i32,
    pub start_time: f64,
    pub duration: f64,
    pub velocity: f64,
    pub mute: bool,
    pub probability: f64,
    pub velocity_deviation: f64,
    pub release_velocity: f64,
    pub note_id: i64,
}

#[derive(Debug, Clone)]
pub struct AbletonClip {
    pub notes: Vec<AbletonNote>,
    pub length: f64,
}

#[derive(Debug, Deserialize)]
struct ClipNotesResult {
    #[serde(default)]
    length: f64,
    notes: Vec<AbletonNote>,
}

#[derive(Debug, Deserialize)]
struct ClipNotesResponse {
    status: String,
    #[serde(default)]
    result: Option<ClipNotesResult>,
    #[serde(default)]
    message: Option<String>,
}

pub async fn get_clip(track_index: i32, clip_index: i32) -> io::Result<AbletonClip> {
    let cmd = json!({
        "type": "get_clip_notes",
        "params": { "track_index": track_index, "clip_index": clip_index }
    });
    let payload = serde_json::to_vec(&cmd)?;

    let mut stream = timeout(Duration::from_secs(10), TcpStream::connect((HOST, PORT)))
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "connect timeout"))??;

    stream.write_all(&payload).await?;

    let mut buf = Vec::with_capacity(8192);
    let mut chunk = [0u8; 8192];
    loop {
        let n = timeout(Duration::from_secs(10), stream.read(&mut chunk))
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "read timeout"))??;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Ok(resp) = serde_json::from_slice::<ClipNotesResponse>(&buf) {
            return finalize(resp);
        }
    }
    let resp: ClipNotesResponse = serde_json::from_slice(&buf).map_err(|e| {
        io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("connection closed before complete JSON: {e}"),
        )
    })?;
    finalize(resp)
}

fn finalize(resp: ClipNotesResponse) -> io::Result<AbletonClip> {
    if resp.status != "success" {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            resp.message.unwrap_or_else(|| "ableton error".to_string()),
        ));
    }
    let r = resp.result.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing result"))?;
    Ok(AbletonClip { notes: r.notes, length: r.length })
}

pub async fn get_clip_notes(track_index: i32, clip_index: i32) -> io::Result<Vec<AbletonNote>> {
    Ok(get_clip(track_index, clip_index).await?.notes)
}
