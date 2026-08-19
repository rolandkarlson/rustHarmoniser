mod utils;
mod model;
mod music_theory;
mod rhythm;
mod harmonizer;
mod schillinger;
mod api;
mod ableton;

use model::{Config, Note};

use rhythm::{gen_rythm2, transform_rhythm};
use harmonizer::{gen_voice, gen_voice_from_notes, harmonise2, HarmonizerState};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::time::Instant;

fn main() -> std::io::Result<()> {
    println!("Starting Web Server mode...");
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            api::start_server().await;
        });
    Ok(())
}

use std::sync::mpsc::Sender;
use crate::utils::ArrayExt;

pub fn run_generation(config: &Config, progress_sender: Option<Sender<(usize, usize)>>) -> std::io::Result<String> {
    run_generation_with_leading(config, progress_sender, None)
}

pub fn run_generation_with_leading(
    config: &Config,
    progress_sender: Option<Sender<(usize, usize)>>,
    leading: Option<(Vec<Note>, f64)>,
) -> std::io::Result<String> {
    utils::SeededRng::set_seed(config.rng_seed);
    let start_time = Instant::now();
    // 2. Rhythm Generation Rules (Simplified port)
    // In JS, `baseRythm` and `rules` match `transformRhythm`.
    // Let's create `rythmForVoice` similar to JS logic.
    let base_rythm = vec![
        vec![4.0],
        vec![4.0],
        vec![4.0],
        vec![4.0],
    ];
    
    // Just generating some rhythm data similar to JS loop
    let mut rythm_for_voice: Vec<Vec<f64>> = Vec::new();
    for x in 0..5 {
        let mut rrr = Vec::new();
        // pattern selection from JS `[...].get(x)`
        let pattern_idx = x % base_rythm.len(); 
        // actually JS: `genRythm2(4, [ [1], [1]... ].get(x))` - wait, it passes simple [1] arrays?
        // Ah, JS `rythmForVoice` loop: `var ss = genRythm2(4, [ [1] ... ].get(x))`
        // It uses `[1]` for all 5 voices?
        // `[[1], [1], [1], [1], [1]].get(x)` -> always `[1]`.
        // So `genRythm2(4, [1])`.
        // Wait, `genRythm2` takes `pn` (pattern notes).
        
        for _ in 0..80 { 
            let ss = gen_rythm2(4.0, &vec![4.0]);
            // JS: for (var i=0; i<PL; i++) rrr.push(ss); PL=8
            for _ in 0..8 {
                rrr.extend(ss.clone());
            }
        }
        rythm_for_voice.push(rrr);
    }
    
    // 3. Generate Voices
    // JS: `harmonise2(genVoice(70...).concat(genVoice(65...))...)`
    // We concatenate them all into one `income` list.
    
    // Helper to extract a "bar" function logic: we generated flat lists `rythm_for_voice`
    // JS `bar` function: `return rythmForVoice[0].get(Math.floor(pos / 4))`
    // We can pre-calculate the note durations based on the flat list? 
    // `gen_voice` in my Rust implementation takes `rhythm_data`.
    
    let mut income = Vec::new();

    // Starting/seed pitch per voice (high → low). Configurable via the GUI
    // "Start Notes" modal; falls back per index to the historical defaults.
    let start = |voice: usize| -> i32 {
        config
            .start_notes
            .get(voice)
            .copied()
            .unwrap_or(model::DEFAULT_START_NOTES[voice])
    };

    if config.use_leading_voice {
        if let Some((pattern, source_len)) = leading.as_ref() {
            if !pattern.is_empty() {
                income.extend(gen_voice_from_notes(pattern, *source_len, &config));
            } else {
                income.extend(gen_voice(start(0), &config.voice_rhythm, &[0], 0, 1, &config));
            }
        } else {
            income.extend(gen_voice(start(0), &config.voice_rhythm, &[0], 0, 1, &config));
        }
    } else {
        income.extend(gen_voice(start(0), &config.voice_rhythm, &[0], 0, 1, &config));
    }
    income.extend(gen_voice(start(1), &config.voice_rhythm, &[0], 1, 1, &config));
    income.extend(gen_voice(start(2), &config.voice_rhythm, &[0], 2, 1, &config));


    income.extend(gen_voice(start(3), &config.voice_rhythm, &[0], 3, 1, &config));
    income.extend(gen_voice(start(4), &config.voice_rhythm, &[0], 4, 1, &config));

    // Sort income by start time then pitch
    income.sort_by(|a, b| {
        if (a.start - b.start).abs() > 0.001 {
            a.start.partial_cmp(&b.start).unwrap()
        } else {
            b.pitch.cmp(&a.pitch) // Secondary: pitch descending? JS: `b.pitch - a.pitch` -> Descending
        }
    });

    // 4. Harmonize
    let schillinger_notes = schillinger::gen_schillinger_progression(&config);
    let state = HarmonizerState {
        schillinger_notes,
        voice_contour: config.voice_contour.clone(),
        contour_resolution: config.voice_contour_resolution,
        harmony_contour: config.harmony_distance_contour.clone(),
        harmony_contour_resolution: config.voice_contour_resolution,
        harmony_matrix_contour: config.harmony_matrix_contour.clone(),
        harmony_matrix: config.harmony_matrix.clone(),
        melody_force_contour: config.melody_force_contour.clone(),
    };

    let mut notes = harmonise2(income, &config, &state, progress_sender.as_ref());

    // 5. Apply main pitch offset
    for note in &mut notes {
        note.pitch += config.main_pitch;
    }

    // 6. Output
    let json = serde_json::to_string_pretty(&notes)?;
    let mut file = File::create("output.json")?;
    file.write_all(json.as_bytes())?;

    // Append to JS file
    append_to_js_file(&notes)?;

    // 7. Archive the completed render: one self-contained JSON per render in
    // render/, holding the full input (config + any leading clip) and the
    // output notes. Best-effort — a failed archive write must not fail the
    // render that already succeeded.
    let archived = match archive_render("render", &config, leading.as_ref(), &notes) {
        Ok(path) => format!(" — archived to {path}"),
        Err(e) => {
            eprintln!("render archive failed: {e}");
            String::new()
        }
    };

    Ok(format!("Generated {} notes in {:?}{}", notes.len(), start_time.elapsed(), archived))
}

/// Write one JSON document per completed render into `dir`: the input that
/// fully determines the render (the config exactly as the GUI posted it,
/// including rng_seed, plus the leading clip if one was fetched from Ableton)
/// and the output notes. The unix-millis filename keeps renders from ever
/// overwriting each other. Returns the path written.
fn archive_render(
    dir: &str,
    config: &Config,
    leading: Option<&(Vec<Note>, f64)>,
    notes: &[Note],
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
            "leading": leading.map(|(pattern, length)| serde_json::json!({
                "notes": pattern,
                "clip_length": length,
            })),
        },
        "output": {
            "note_count": notes.len(),
            "notes": notes,
        },
    });
    let mut file = File::create(&path)?;
    file.write_all(serde_json::to_string_pretty(&doc)?.as_bytes())?;
    Ok(path)
}

#[cfg(test)]
mod repro_tests {
    use super::*;
    use harmonizer::HarmonizerState;

    /// Build the same voices + state `run_generation` does, minus file I/O.
    fn pipeline(config: &Config) -> (Vec<Note>, Vec<Vec<Vec<i32>>>) {
        utils::SeededRng::set_seed(config.rng_seed);
        let start = |voice: usize| -> i32 {
            config.start_notes.get(voice).copied().unwrap_or(model::DEFAULT_START_NOTES[voice])
        };
        let mut income = Vec::new();
        for v in 0..5 {
            income.extend(gen_voice(start(v), &config.voice_rhythm, &[0], v as i32, 1, config));
        }
        income.sort_by(|a, b| {
            if (a.start - b.start).abs() > 0.001 {
                a.start.partial_cmp(&b.start).unwrap()
            } else {
                b.pitch.cmp(&a.pitch)
            }
        });
        let schillinger_notes = schillinger::gen_schillinger_progression(config);
        let state = HarmonizerState {
            schillinger_notes: schillinger_notes.clone(),
            voice_contour: config.voice_contour.clone(),
            contour_resolution: config.voice_contour_resolution,
            harmony_contour: config.harmony_distance_contour.clone(),
            harmony_contour_resolution: config.voice_contour_resolution,
            harmony_matrix_contour: config.harmony_matrix_contour.clone(),
            harmony_matrix: config.harmony_matrix.clone(),
            melody_force_contour: config.melody_force_contour.clone(),
        };
        (harmonise2(income, config, &state, None), schillinger_notes)
    }

    fn default_config() -> Config {
        let mut c = Config::default();
        c.init_contours();
        c
    }

    /// Share of bass notes (channel 4) sitting on the bar's chord root.
    fn bass_on_root(notes: &[Note], sch: &[Vec<Vec<i32>>]) -> f64 {
        let bars = &sch[0];
        let mut hits = 0;
        let mut total = 0;
        for n in notes.iter().filter(|n| n.channel == 4) {
            let bar = (n.start / 4.0).floor() as usize % bars.len();
            let root = bars[bar][0].rem_euclid(12);
            total += 1;
            if n.pitch.rem_euclid(12) == root {
                hits += 1;
            }
        }
        hits as f64 / total.max(1) as f64
    }

    #[test]
    fn archive_render_round_trips_input_and_output() {
        let dir = std::env::temp_dir().join(format!("rustnote_archive_test_{}", std::process::id()));
        let dir = dir.to_str().unwrap();
        let _ = std::fs::remove_dir_all(dir);

        let mut config = default_config();
        config.rng_seed = 1234.0;
        let notes = vec![Note::new(60, 0.0, 4.0, 100, 0, 0), Note::new(48, 0.0, 4.0, 90, 0, 4)];
        let leading = (vec![Note::new(72, 0.0, 2.0, 80, 0, 0)], 8.0);

        let path = archive_render(dir, &config, Some(&leading), &notes).unwrap();
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
        let path2 = archive_render(dir, &config, None, &notes).unwrap();
        let doc2: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path2).unwrap()).unwrap();
        assert!(doc2["input"]["leading"].is_null());

        let _ = std::fs::remove_dir_all(dir);
    }

    /// Musical read-out of a default render, for tuning by numbers rather than
    /// by ear alone:
    ///   cargo test --release musical_stats -- --ignored --nocapture
    #[test]
    #[ignore]
    fn musical_stats() {
        for w in [0.0, 0.25, 0.5, 1.0, 2.0] {
            let mut c = default_config();
            c.root_position_weight = w;
            let (notes, sch) = pipeline(&c);
            println!("root_position_weight={w:<4} bass on root {:.2}", bass_on_root(&notes, &sch));
        }
        for w in [0.0, 0.5, 1.0, 1.5] {
            println!(
                "tendency_weight={w:<4}     leading tones resolving up {:.2}",
                leading_tone_resolution_rate(w),
            );
        }
    }

    #[test]
    fn full_pipeline_runs_and_is_deterministic() {
        let cfg = default_config();
        let (a, _) = pipeline(&cfg);
        assert!(!a.is_empty(), "the pipeline produced no notes");
        for n in &a {
            assert!((0..128).contains(&n.pitch), "pitch {} out of MIDI range", n.pitch);
            assert_eq!(n.muted, 0);
        }
        let (b, _) = pipeline(&cfg);
        let pa: Vec<i32> = a.iter().map(|n| n.pitch).collect();
        let pb: Vec<i32> = b.iter().map(|n| n.pitch).collect();
        assert_eq!(pa, pb, "the same seed must give the same render");
    }

    #[test]
    fn root_position_weight_moves_the_bass_onto_chord_roots() {
        // End-to-end version of the unit test: over a whole default render, the
        // bass sits on the chord root far more often with the preference on.
        let mut on = default_config();
        on.root_position_weight = 2.0;
        let (notes_on, sch) = pipeline(&on);
        let with = bass_on_root(&notes_on, &sch);

        let mut off = default_config();
        off.root_position_weight = 0.0;
        let (notes_off, sch_off) = pipeline(&off);
        let without = bass_on_root(&notes_off, &sch_off);

        assert!(
            with > without,
            "bass-on-root {with:.2} with the preference vs {without:.2} without",
        );
    }

    /// Share of leading tones (pc 11 in the default key of C) that resolve up
    /// by semitone to the tonic, over a whole default render.
    fn leading_tone_resolution_rate(tendency_weight: f64) -> f64 {
        let mut c = default_config();
        c.tendency_weight = tendency_weight;
        let (notes, _) = pipeline(&c);
        let mut by_channel: std::collections::HashMap<i32, Vec<&Note>> = std::collections::HashMap::new();
        for n in &notes {
            by_channel.entry(n.channel).or_default().push(n);
        }
        let (mut resolved, mut total) = (0, 0);
        for line in by_channel.values_mut() {
            line.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
            for pair in line.windows(2) {
                if pair[0].pitch.rem_euclid(12) == 11 {
                    total += 1;
                    if pair[1].pitch - pair[0].pitch == 1 {
                        resolved += 1;
                    }
                }
            }
        }
        resolved as f64 / total.max(1) as f64
    }

    #[test]
    fn tendency_weight_makes_leading_tones_resolve() {
        // Without the term the leading tone is just another scale degree and
        // resolves by accident (~6%); with it, a third of them step up to the
        // tonic — the rest are bars where the next chord has no tonic to land
        // on, or where crossing/budget constraints outrank it.
        let off = leading_tone_resolution_rate(0.0);
        let on = leading_tone_resolution_rate(1.5);
        assert!(on > off + 0.15, "resolution rate {off:.2} → {on:.2} is not a real effect");
    }

    #[test]
    fn generated_progression_renders_end_to_end() {
        let mut cfg = default_config();
        cfg.use_generated_progression = true;
        let (notes, sch) = pipeline(&cfg);
        assert!(!notes.is_empty());
        assert_eq!(sch[0].len(), (cfg.pl * cfg.render_length) as usize);
    }

    /// TEMP repro: run the full generation pipeline (minus file side effects)
    /// with a config supplied via REPRO_CONFIG and dump the first chords vs the
    /// Schillinger scale per bar. Run with:
    ///   REPRO_CONFIG=path cargo test repro_first_chord_scale -- --ignored --nocapture
    #[test]
    #[ignore]
    fn repro_first_chord_scale() {
        let path = std::env::var("REPRO_CONFIG").expect("set REPRO_CONFIG");
        let config: Config =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        utils::SeededRng::set_seed(config.rng_seed);

        let start = |voice: usize| -> i32 {
            config.start_notes.get(voice).copied().unwrap_or(model::DEFAULT_START_NOTES[voice])
        };
        let mut income = Vec::new();
        for v in 0..5 {
            income.extend(gen_voice(start(v), &config.voice_rhythm, &[0], v as i32, 1, &config));
        }
        income.sort_by(|a, b| {
            if (a.start - b.start).abs() > 0.001 {
                a.start.partial_cmp(&b.start).unwrap()
            } else {
                b.pitch.cmp(&a.pitch)
            }
        });

        let schillinger_notes = schillinger::gen_schillinger_progression(&config);
        for (v, bars) in schillinger_notes.iter().enumerate() {
            println!("voice {v}: bar0={:?} bar1={:?}", bars.get(0), bars.get(1));
        }

        let state = HarmonizerState {
            schillinger_notes: schillinger_notes.clone(),
            voice_contour: config.voice_contour.clone(),
            contour_resolution: config.voice_contour_resolution,
            harmony_contour: config.harmony_distance_contour.clone(),
            harmony_contour_resolution: config.voice_contour_resolution,
            harmony_matrix_contour: config.harmony_matrix_contour.clone(),
            harmony_matrix: config.harmony_matrix.clone(),
            melody_force_contour: config.melody_force_contour.clone(),
        };

        let notes = harmonise2(income, &config, &state, None);

        let mut starts: Vec<f64> = notes.iter().map(|n| n.start).collect();
        starts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        starts.dedup_by(|a, b| (*a - *b).abs() < 0.001);
        for &s in starts.iter().take(6) {
            let bar = (s / 4.0).floor() as usize;
            let mut group: Vec<&Note> = notes.iter()
                .filter(|n| (n.start - s).abs() < 0.001)
                .collect();
            group.sort_by_key(|n| n.channel);
            println!("--- start {s} (bar {bar}) ---");
            for n in group {
                let voice_bars = &schillinger_notes[(n.channel as usize) % schillinger_notes.len()];
                let scale = &voice_bars[bar % voice_bars.len()];
                let in_scale = scale.iter().any(|&p| (p % 12 + 12) % 12 == (n.pitch % 12 + 12) % 12);
                println!(
                    "ch {} pitch {} pc {} scale {:?} in_scale {}",
                    n.channel, n.pitch, (n.pitch % 12 + 12) % 12, scale, in_scale
                );
            }
        }
    }
}

fn append_to_js_file(notes: &[Note]) -> std::io::Result<()> {
   // let path = "C:\\Users\\rolan\\Documents\\Ableton\\User Library\\Presets\\MIDI Effects\\Max MIDI Effect\\harmonizer\\harmonize.js";
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
        // Truncate file and write from the beginning up to the marker + new content
        let new_content = format!("{}{}\n\n{}\n.writeMidi();",
            &content[..idx + "//REPLACE".len()],
            "\n", // Just a newline after REPLACE
            serde_json::to_string(notes)?
        );
        
        // Re-open in truncate mode to overwrite
        let mut file = File::create(path)?;
        file.write_all(new_content.as_bytes())?;
    }
    
    Ok(())
}
