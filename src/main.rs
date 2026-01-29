mod utils;
mod model;
mod music_theory;
mod rhythm;
mod harmonizer;
mod schillinger;

use model::{Config, Note};

use rhythm::{gen_rythm2, transform_rhythm};
use harmonizer::{gen_voice, harmonise2, HarmonizerState};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::time::Instant;

fn main() -> std::io::Result<()> {
    let start_time = Instant::now();
    // 1. Setup

    let config = Config::default();
    
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

    income.extend(gen_voice(70, &config.voice_rhythm, &[0], 0, 1, &config));
    income.extend(gen_voice(65, &config.voice_rhythm, &[0], 1, 1, &config));
    income.extend(gen_voice(60, &config.voice_rhythm, &[0], 2, 1, &config));
    

    income.extend(gen_voice(50, &config.voice_rhythm, &[0], 3, 1, &config));
    income.extend(gen_voice(40, &config.voice_rhythm, &[0], 4, 1, &config));

    // Sort income by start time then pitch
    income.sort_by(|a, b| {
        if (a.start - b.start).abs() > 0.001 {
            a.start.partial_cmp(&b.start).unwrap()
        } else {
            b.pitch.cmp(&a.pitch) // Secondary: pitch descending? JS: `b.pitch - a.pitch` -> Descending
        }
    });

    // 4. Harmonize
    let schillinger_notes = schillinger::gen_schillinger_progression();
    let state = HarmonizerState {
        schillinger_notes,
    };

    let notes = harmonise2(income, &config, &state);

    // 5. Output
    let json = serde_json::to_string_pretty(&notes)?;
    let mut file = File::create("output.json")?;
    file.write_all(json.as_bytes())?;

    // Append to JS file
    append_to_js_file(&notes)?;
    
    println!("Generated {} notes to output.json", notes.len());
    println!("Execution time: {:?}", start_time.elapsed());

    Ok(())
}

fn append_to_js_file(notes: &[Note]) -> std::io::Result<()> {
    //let path = "C:\\Users\\rolan\\Documents\\Ableton\\User Library\\Presets\\MIDI Effects\\Max MIDI Effect\\harmonizer\\harmonize.js";
    let path = "/Users/roland/Music/Ableton/User Library/Presets/Instruments/Max Instrument/harmonize.js";

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
