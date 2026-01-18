use crate::utils::{mod_shim, ArrayExt};

pub fn gen_scale(ap: &[i32], center_octave: i32) -> Vec<i32> {
    let start = if center_octave - 1 < 0 { 0 } else { 0 };
    let end = if center_octave + 1 > 8 { 0 } else { 8 };
    let mut ar = Vec::new();

    for i in start..end {
        for &note in ap {
            ar.push(note + 12 * i);
        }
    }
    ar
}

pub fn get_harmonic_score_adjusted(note_a: i32, note_b: i32) -> f64 {
    let low_note = note_a.min(note_b);
    let high_note = note_a.max(note_b);
    let dist = high_note - low_note;

    let mut effective_dist = dist;
    if dist > 23 {
        effective_dist = 12 + (dist % 12);
    }

    if dist % 12 == 1 || dist % 12 == 11 || dist % 12 == 6{
        // "Clash" penalty in JS was -100000000, here we return 0.0 or handle it in mapping
        // JS returned -100000000.
        return -100000000.0;
    }

    let score: f64 = match effective_dist {
        0 => 1.0,   // Unison
        1 => 0.0,   // Min 2nd
        2 => 0.2,   // Maj 2nd
        3 => 0.6,   // Min 3rd
        4 => 0.8,   // Maj 3rd
        5 => 0.7,   // P4
        6 => 0.0,   // Tritone
        7 => 1.0,   // P5
        8 => 0.7,   // Min 6th
        9 => 0.8,   // Maj 6th
        10 => 0.3,  // Min 7th
        11 => 0.4,  // Maj 7th
        12 => 1.0,  // Octave
        13 => 0.0,  // Min 9th
        14 => 0.85, // Maj 9th
        15 => 0.7,  // Min 10th
        16 => 0.9,  // Maj 10th
        17 => 0.7,  // P11
        18 => 0.2,  // #11
        19 => 1.0,  // P12
        20 => 0.7,  // Min 13th
        21 => 0.85, // Maj 13th
        22 => 0.5,  // Min 14th
        23 => 0.6,  // Maj 14th
        _ => 0.0,
    };

    // JS: return Math.max(0.0, Math.min(1.0, score));
    // But wait, if it returns -100000000 above, the clamp would make it 0.0.
    // The JS code has `if (dist % 12 === 1 || ...)` return -big;
    // THEN `var score = harmonyMap...`
    // THEN `return Math.max(...)`.
    // So the -big IS Clamped to 0.0?
    // Wait, let's re-read JS.
    // if (...) return -100000000;
    // ...
    // return Math.max(...)
    // The return -100000000 is an EARLY return. So it returns negative.
    // My previous assumption was correct, it returns negative.
    // The clamp is only for the map lookup part effectively?
    // No, if it returns early, the clamp isn't reached.
    // So distinct behavior.

    score.clamp(0.0, 1.0)
}


pub fn generate_mode_from_steps(root: i32, mode: i32) -> Vec<i32> {
    let step_pattern = vec![2, 2, 1, 2, 2, 2, 1];

    // rotate
    let steps_rot = if mode > 0 {
         let split_idx = mode as usize % step_pattern.len();
         let (left, right) = step_pattern.split_at(split_idx);
         [right, left].concat()
    } else {
        step_pattern.clone()
    };

    // remove last? JS: modePattern.pop();
    // Actually we need to walk it.
    let mut mode_pattern = steps_rot;
    mode_pattern.pop();

    let mut mode_notes = vec![root];
    let mut current_note = root;

    for step in mode_pattern {
        current_note = (current_note + step) % 12;
        mode_notes.push(current_note);
    }
    mode_notes.sort();
    mode_notes
}

