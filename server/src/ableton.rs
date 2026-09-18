use serde::{Deserialize, Serialize};
use serde_json::json;
use serde_json::Value;
use std::io;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::timeout;

const HOST: &str = "localhost";
const PORT: u16 = 9877;
const MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

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

async fn command(
    command_type: &str,
    params: Value,
    timeout_secs: u64,
    mutation: bool,
) -> io::Result<Value> {
    let cmd = json!({
        "type": command_type,
        "params": params,
    });
    let payload = serde_json::to_vec(&cmd)?;

    let mut stream = timeout(Duration::from_secs(10), TcpStream::connect((HOST, PORT)))
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "connect timeout"))??;

    if let Err(e) = stream.write_all(&payload).await {
        return Err(if mutation { uncertain(e) } else { e });
    }

    let mut buf = Vec::with_capacity(8192);
    let mut chunk = [0u8; 8192];
    loop {
        let n = match timeout(Duration::from_secs(timeout_secs), stream.read(&mut chunk)).await {
            Ok(result) => result.map_err(|e| if mutation { uncertain(e) } else { e })?,
            Err(_) => {
                return Err(if mutation {
                    uncertain(io::Error::new(io::ErrorKind::TimedOut, "read timeout"))
                } else {
                    io::Error::new(io::ErrorKind::TimedOut, "read timeout")
                })
            }
        };
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        if buf.len() > MAX_RESPONSE_BYTES {
            return Err(if mutation {
                uncertain(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Ableton response exceeded 64 MiB",
                ))
            } else {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Ableton response exceeded 64 MiB",
                )
            });
        }
        if let Ok(resp) = serde_json::from_slice::<Value>(&buf) {
            if resp.get("status").and_then(Value::as_str) != Some("success") {
                let error = io::Error::new(
                    io::ErrorKind::Other,
                    resp.get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("Ableton error"),
                );
                return Err(
                    if mutation && error.to_string().to_ascii_lowercase().contains("timeout") {
                        uncertain(error)
                    } else {
                        error
                    },
                );
            }
            return Ok(resp.get("result").cloned().unwrap_or_else(|| json!({})));
        }
    }
    let resp: Value = serde_json::from_slice(&buf).map_err(|e| {
        let error = io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("connection closed before complete JSON: {e}"),
        );
        if mutation {
            uncertain(error)
        } else {
            error
        }
    })?;
    if resp.get("status").and_then(Value::as_str) != Some("success") {
        let error = io::Error::new(
            io::ErrorKind::Other,
            resp.get("message")
                .and_then(Value::as_str)
                .unwrap_or("Ableton error"),
        );
        return Err(
            if mutation && error.to_string().to_ascii_lowercase().contains("timeout") {
                uncertain(error)
            } else {
                error
            },
        );
    }
    Ok(resp.get("result").cloned().unwrap_or_else(|| json!({})))
}

fn uncertain(error: io::Error) -> io::Error {
    io::Error::new(
        error.kind(),
        format!("Ableton write outcome is uncertain; do not retry automatically: {error}"),
    )
}

pub async fn get_clip(track_index: i32, clip_index: i32) -> io::Result<AbletonClip> {
    let value = command(
        "get_clip_notes",
        json!({ "track_index": track_index, "clip_index": clip_index }),
        10,
        false,
    )
    .await?;
    let r: ClipNotesResult = serde_json::from_value(value)?;
    Ok(AbletonClip {
        notes: r.notes,
        length: r.length,
    })
}

pub async fn get_script_info() -> io::Result<Value> {
    command("get_script_info", json!({}), 10, false).await
}

pub async fn get_output_session() -> io::Result<Value> {
    let script = get_script_info().await?;
    let session = command("get_session_info", json!({}), 10, false).await?;
    let track_count = session
        .get("track_count")
        .and_then(Value::as_i64)
        .unwrap_or(0)
        .clamp(0, 1024);
    let mut tracks = Vec::with_capacity(track_count as usize);
    for track_index in 0..track_count {
        tracks.push(
            command(
                "get_track_info",
                json!({"track_index": track_index}),
                10,
                false,
            )
            .await?,
        );
    }
    Ok(json!({"status": "success", "script": script, "session": session, "tracks": tracks}))
}

pub async fn write_harmonizer_clips(
    clips: &[crate::output::ClipPlan],
    source: Option<(i32, i32)>,
) -> io::Result<Value> {
    let info = get_script_info().await?;
    let supported = info
        .get("capabilities")
        .and_then(Value::as_array)
        .map_or(false, |caps| {
            caps.iter()
                .any(|c| c.as_str() == Some("write_harmonizer_clips"))
        });
    if !supported {
        return Err(io::Error::new(io::ErrorKind::Unsupported, "AbletonMCP does not advertise write_harmonizer_clips; install the rustHarmoniser bridge extension and restart Live"));
    }
    command("write_harmonizer_clips", json!({
        "clips": clips,
        "source": source.map(|(track_index, clip_index)| json!({"track_index": track_index, "clip_index": clip_index})),
    }), 40, true).await
}

pub async fn get_clip_notes(track_index: i32, clip_index: i32) -> io::Result<Vec<AbletonNote>> {
    Ok(get_clip(track_index, clip_index).await?.notes)
}

/// Fetch a clip and return the pitches of its LAST chord — the group of notes
/// whose start times fall within `EPS` beats of the latest onset — sorted high
/// → low. Used to seed the 5 voices' starting notes ("gen phrases, pull the
/// resolving chord out of Ableton, use it as the next seed").
pub async fn get_last_chord(track_index: i32, clip_index: i32) -> io::Result<Vec<i32>> {
    const EPS: f64 = 0.05; // beats; groups near-simultaneous onsets into one chord
    let clip = get_clip(track_index, clip_index).await?;

    // Ignore muted notes; they aren't sounding.
    let mut onsets: Vec<&AbletonNote> = clip.notes.iter().filter(|n| !n.mute).collect();
    if onsets.is_empty() {
        return Ok(Vec::new());
    }

    let max_start = onsets
        .iter()
        .map(|n| n.start_time)
        .fold(f64::NEG_INFINITY, f64::max);

    onsets.retain(|n| n.start_time >= max_start - EPS);
    // High → low, matching the voice order (voice 0 = top).
    onsets.sort_by(|a, b| b.pitch.cmp(&a.pitch));

    let mut pitches: Vec<i32> = onsets.iter().map(|n| n.pitch).collect();
    pitches.dedup();
    Ok(pitches)
}
