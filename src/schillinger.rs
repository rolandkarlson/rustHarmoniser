use crate::utils::{mod_shim, SeededRng, ArrayExt};
use crate::music_theory::{generate_mode_from_steps};
use crate::model::Config;

// PL was 8 in JS, now passed via Config
const EXP: i32 = 2;

pub fn generate_progression(length: usize, mode: i32) -> Vec<i32> {
    if length == 0 {
        return vec![];
    }
    let ionian_transitions: [&[i32]; 7] = [&[3, 4, 5, 1, 2], &[4, 6], &[5, 3], &[0, 1, 4], &[0, 5], &[3, 1, 4], &[0]];

    // 2. DORIAN - Focus on i (0) and IV (3)
    let dorian_transitions: [&[i32]; 7] = [&[3, 6, 1], &[0, 3], &[3, 4], &[0, 6], &[0, 3], &[6, 3], &[0, 3]];

    // 3. PHRYGIAN (0=i, 1=II, 2=III, 3=iv, 4=v°, 5=VI, 6=vii)
    // The flat-2nd creates a major II chord. The i -> II -> i movement is the core Phrygian sound.
    let phrygian_transitions: [&[i32]; 7] = [
        &[1, 3, 5],       // i   -> Heavily favors sliding up to II, or going to iv or VI
        &[0],             // II  -> THE characteristic chord. Slams straight back to i
        &[1, 3],          // III -> Moves to II or iv
        &[0, 1],          // iv  -> Resolves to i, or steps to II
        &[1, 5],          // v°  -> Diminished. Pushes to II or VI
        &[1, 0],          // VI  -> Moves to II or i
        &[0, 2],          // vii -> Steps to i or III
    ];

    // 4. LYDIAN (0=I, 1=II, 2=iii, 3=iv°, 4=V, 5=vi, 6=vii)
    // The sharp-4th creates a major II chord. Bouncing between I and II is the classic Lydian float.
    let lydian_transitions: [&[i32]; 7] = [
        &[1, 4, 2],       // I   -> Favors moving to II, V, or iii
        &[0, 4],          // II  -> THE characteristic chord. Resolves back to I, or moves to V
        &[0, 1],          // iii -> Moves to I or II
        &[4, 2],          // iv° -> Diminished. Pushes to V or iii
        &[0, 1],          // V   -> Resolves to I or steps to II
        &[1, 4],          // vi  -> Pushes to II or V
        &[0, 2],          // vii -> Resolves to I or iii
    ];

    // 5. MIXOLYDIAN - Focus on I (0), IV (3), and VII (6)
    let mixolydian_transitions: [&[i32]; 7] = [&[3, 6, 4], &[0, 3], &[3, 5], &[0, 6], &[0, 3], &[3, 1], &[0, 3]];

    // 6. AEOLIAN (Minor) - Uses borrowed V for strong resolution
    let aeolian_transitions: [&[i32]; 7] = [&[2, 3, 4, 5, 6], &[4, 6], &[5, 3], &[0, 1, 4, 6], &[0, 5], &[1, 3, 4], &[2, 0]];

    // 7. LOCRIAN (0=i°, 1=II, 2=iii, 3=iv, 4=V, 5=VI, 6=vii)
    // The root chord is diminished, making this mode inherently unstable and dark. It rarely resolves cleanly.
    let locrian_transitions: [&[i32]; 7] = [
        &[1, 3, 5],       // i°  -> Diminished root! Pushes away to II, iv, or VI
        &[0, 3],          // II  -> Steps back to i°, or moves to iv
        &[1, 5],          // iii -> Pushes to II or VI
        &[0, 1],          // iv  -> Moves back to i° or II
        &[1, 5],          // V   -> Pushes to II or VI
        &[0, 1],          // VI  -> Moves to i° or II
        &[0, 5],          // vii -> Pushes to i° or VI
    ];

    let transitions = match mode {
        0 => ionian_transitions,
        1=> dorian_transitions,
        2=> phrygian_transitions,
        3 => lydian_transitions,
        4 => mixolydian_transitions,
        5 => aeolian_transitions,
        6 => locrian_transitions,
        _ => panic!("Invalid mode"),
    };
    let max_attempts = 1000;
    
    for _ in 0..max_attempts {
        let mut progression = Vec::with_capacity(length);
        let mut current_chord = 0;
        progression.push(current_chord);

        for _ in 1..length {
            let possible_next_chords = transitions[current_chord as usize];
            let random_index = SeededRng::random_int(possible_next_chords.len() as i32) as usize;
            current_chord = possible_next_chords[random_index];
            progression.push(current_chord);
        }

        if *progression.last().unwrap() == 4 {
            return progression;
        }
    }

    // Fallback if we can't natively end on 0 (e.g. length is 2)
    let mut progression = Vec::with_capacity(length);
    let mut current_chord = 0;
    progression.push(current_chord);

    for i in 1..length {
        if i == length - 1 {
            progression.push(0);
            continue;
        }
        let possible_next_chords = transitions[current_chord as usize];
        let random_index = SeededRng::random_int(possible_next_chords.len() as i32) as usize;
        current_chord = possible_next_chords[random_index];
        progression.push(current_chord);
    }

    progression
}

fn find_sequence_with_condition(possible_steps: &[i32], sequence_length: i32) -> Option<Vec<i32>> {
    let max_attempts = 1000000;
    let mut attempts = 0;

    while attempts < max_attempts {
        let mut sequence = vec![0];
        let mut current_sum = 0;
        let mut last_value = -999;

        for _ in 1..sequence_length {
            let random_index = (SeededRng::seeded_random(1.0, 0.0) * possible_steps.len() as f64).floor() as usize;
            let mut step = possible_steps[random_index];

            if last_value + step == 0 {
                step *= -1;
            }
            last_value = step;
            current_sum += step;
            
            if last_value + step == 0 {
                step *= -1;
            }
            // Logic in JS for `step *= -1` appears TWICE.
            // JS:
            // if (lastValue + step === 0) { step *= -1; }
            // lastValue = step;
            // currentSum += step;
            // if (lastValue + step === 0) { step *= -1; } -> This checks step against itself? lastValue is step.
            // wait, lastValue is updated to step. so `step + step == 0`? No.
            // JS: `lastValue = step; ... if (lastValue + step === 0)` -> checks `step + step === 0` -> `2*step === 0`.
            // only if step is 0 (which it isn't, -2 etc * EXP).
            // So the second check is redundant or logic error in original JS, but I must replicate strictly if I want exact same behavior?
            // Actually, `currentSum` is updated.
            // JS:
            /*
            if (lastValue + step === 0) {
                step *= -1;
            }
            lastValue = step;
            // 2. Calculate the new sum
            currentSum += step;

            if (lastValue + step === 0) {
                step *= -1;
            }
            // 3. Add the new sum to the sequence
            sequence.push(currentSum);
            */
            // Since `lastValue = step`, `lastValue + step` is `2*step`.
            // The second check is effectively never true unless step=0.
            // I will copy it verbatim to be safe.
            
            sequence.push(mod_shim(current_sum, 7));
        }

        let last_element = sequence.last().unwrap();
        let modulo_result = mod_shim(*last_element, 7);
        // JS: `var moduloResult = ((lastElement % 7) + 7) % 7;`
        
        let target = mod_shim(4, 7);

        if modulo_result == target {
            return Some(sequence);
        }
        attempts += 1;
    }
    None
}

pub const NUM_VOICES: usize = 16;

pub fn gen_schillinger_progression(config: &Config) -> Vec<Vec<Vec<i32>>> {

    let seq = &config.schillinger_sequence;
    let bars = seq.len();

    let chord_list = vec![
        vec![0,1,2],
        vec![0,1,2,4,5],
        vec![0,1,2,4,5],
        vec![0,1,2,3,4],
        vec![0,1,2,3,4,5],
        vec![0,1,2,3,4,5,6]
    ];

    let mut per_voice: Vec<Vec<Vec<i32>>> = Vec::with_capacity(NUM_VOICES);

    for voice in 0..NUM_VOICES {
        let mut chord_notes = Vec::with_capacity(bars);

        let voice_ex_contour: Option<&Vec<f64>> = config
            .schillinger_ex_contour
            .as_ref()
            .and_then(|outer| outer.get(voice));

        let voice_chord_contour: Option<&Vec<f64>> = config
            .chord_structure_contour
            .as_ref()
            .and_then(|outer| outer.get(voice));

        for i in 0..bars {
            let start_time = i as f64 * 4.0;
            let contour_idx = (start_time / config.voice_contour_resolution).floor() as usize;

            let current_mode = if let Some(mc) = &config.mode_contour {
                if !mc.is_empty() {
                    let contour_val = mc.get_wrapped(contour_idx).round() as i32;
                    mod_shim(contour_val, 7)
                } else {
                    config.mode
                }
            } else {
                config.mode
            };

            let chord_idx = match voice_chord_contour {
                Some(cc) if !cc.is_empty() => cc.get_wrapped(contour_idx).round() as usize,
                _ => 0,
            };

            let n_struct = if chord_idx < chord_list.len() {
                &chord_list[chord_idx]
            } else {
                &config.chord_structure
            };

            let scale = generate_mode_from_steps(config.root, &current_mode);
            let ex = match voice_ex_contour {
                Some(ec) if !ec.is_empty() => ec.get_wrapped(contour_idx).round() as i32,
                _ => 2,
            };
            let notes: Vec<i32> = n_struct.iter().map(|&itm| {
                 let idx = (itm * ex) + seq[i as usize % seq.len()];
                 scale[mod_shim(idx, scale.len() as i32) as usize]
            }).collect();

            chord_notes.push(notes);
        }

        per_voice.push(chord_notes);
    }

    per_voice
}
