//! Every sink a finished render is written to. The engine itself (rustnote-core)
//! is pure; this module owns the side effects: output.json, the Max
//! harmonize.js, and the per-render archive.

use rustnote_core::model::{Config, Note};
use rustnote_core::render::{render, Leading, RenderResult};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::time::Instant;

/// Run a render and write it to every sink. Returns the human-readable status
/// message shown in the GUI, plus the result for the API response.
pub fn run_render(config: &Config, leading: Option<&Leading>) -> std::io::Result<(String, RenderResult)> {
    let start_time = Instant::now();
    let result = render(config, leading, None);
    let notes = &result.notes;

    let json = serde_json::to_string_pretty(notes)?;
    let mut file = File::create("output.json")?;
    file.write_all(json.as_bytes())?;

    append_to_js_file(notes)?;

    // Archive the completed render: one self-contained JSON per render in
    // render/, holding the full input (config + any leading clip) and the
    // output notes. Best-effort — a failed archive write must not fail the
    // render that already succeeded.
    let archived = match archive_render("render", config, leading, &result) {
        Ok(path) => format!(" — archived to {path}"),
        Err(e) => {
            eprintln!("render archive failed: {e}");
            String::new()
        }
    };

    let msg = format!("Generated {} notes in {:?}{}", notes.len(), start_time.elapsed(), archived);
    Ok((msg, result))
}

/// Write one JSON document per completed render into `dir`: the input that
/// fully determines the render (the config exactly as the GUI posted it,
/// including rng_seed, plus the leading clip if one was fetched from Ableton)
/// and the output notes. The unix-millis filename keeps renders from ever
/// overwriting each other. Returns the path written.
pub fn archive_render(
    dir: &str,
    config: &Config,
    leading: Option<&Leading>,
    result: &RenderResult,
) -> std::io::Result<String> {
    std::fs::create_dir_all(dir)?;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let path = format!("{dir}/render_{ts}.json");
    let doc = serde_json::json!({
        "timestamp_unix_ms": ts,
        "input": {
            "config": config,
            "leading": leading.map(|l| serde_json::json!({
                "notes": l.notes,
                "clip_length": l.clip_length,
            })),
        },
        "output": {
            "note_count": result.notes.len(),
            "notes": result.notes,
            // Named score contributions per chosen chord — the "why" behind
            // every voicing, for tuning by numbers instead of by ear alone.
            "breakdown": result.breakdown,
        },
    });
    let mut file = File::create(&path)?;
    file.write_all(serde_json::to_string_pretty(&doc)?.as_bytes())?;
    Ok(path)
}

fn append_to_js_file(notes: &[Note]) -> std::io::Result<()> {
    #[cfg(windows)]
    let path = "C:\\Users\\rolan\\Documents\\Ableton\\User Library\\Presets\\MIDI Effects\\Max MIDI Effect\\harmonizer\\harmonize.js";
    #[cfg(not(windows))]
    let path = "/Users/roland/Music/Ableton/User Library/Presets/Instruments/Max Instrument/harmonizer/harmonize.js";

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)?;

    let mut content = String::new();
    file.read_to_string(&mut content)?;

    if let Some(idx) = content.find("//REPLACE") {
        let new_content = format!("{}{}\n\n{}\n.writeMidi();",
            &content[..idx + "//REPLACE".len()],
            "\n",
            serde_json::to_string(notes)?
        );

        let mut file = File::create(path)?;
        file.write_all(new_content.as_bytes())?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archive_render_round_trips_input_and_output() {
        let dir = std::env::temp_dir().join(format!("rustnote_archive_test_{}", std::process::id()));
        let dir = dir.to_str().unwrap();
        let _ = std::fs::remove_dir_all(dir);

        let mut config = Config::default();
        config.init_contours();
        config.rng_seed = 1234.0;
        let result = RenderResult {
            notes: vec![Note::new(60, 0.0, 4.0, 100, 0, 0), Note::new(48, 0.0, 4.0, 90, 0, 4)],
            breakdown: vec![],
            schillinger_notes: vec![],
        };
        let leading = Leading {
            notes: vec![Note::new(72, 0.0, 2.0, 80, 0, 0)],
            clip_length: 8.0,
        };

        let path = archive_render(dir, &config, Some(&leading), &result).unwrap();
        let doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();

        // The input block reproduces the render: config (with seed) + leading clip.
        assert_eq!(doc["input"]["config"]["rng_seed"], 1234.0);
        assert_eq!(doc["input"]["leading"]["clip_length"], 8.0);
        assert_eq!(doc["input"]["leading"]["notes"].as_array().unwrap().len(), 1);
        // The output block holds the generated notes verbatim.
        assert_eq!(doc["output"]["note_count"], 2);
        let out: Vec<Note> = serde_json::from_value(doc["output"]["notes"].clone()).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].pitch, 60);

        // Without a leading clip the field is null, not absent.
        let path2 = archive_render(dir, &config, None, &result).unwrap();
        let doc2: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path2).unwrap()).unwrap();
        assert!(doc2["input"]["leading"].is_null());

        let _ = std::fs::remove_dir_all(dir);
    }
}
