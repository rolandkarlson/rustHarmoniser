use std::cmp::min;
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

// Cost of intervals 0..23. Access is O(1).
// Special "Clash" penalty of -100,000,000.0 is encoded directly at indices 1, 11, 13, 23.
pub const INTERVAL_SCORES: [f32; 24] = [
    1.0,           // 0: Unison
    -100_000_000.0, // 1: Min 2nd (Clash)
    0.2,           // 2: Maj 2nd
    0.6,           // 3: Min 3rd
    0.8,           // 4: Maj 3rd
    0.7,           // 5: P4
    0.0,           // 6: Tritone
    1.0,           // 7: P5
    0.7,           // 8: Min 6th
    0.8,           // 9: Maj 6th
    0.3,           // 10: Min 7th
    -100_000_000.0, // 11: Maj 7th (Clash)
    1.0,           // 12: Octave
    -100_000_000.0, // 13: Min 9th (Clash - 1 % 12 == 1)
    0.85,          // 14: Maj 9th
    0.7,           // 15: Min 10th
    0.9,           // 16: Maj 10th
    0.7,           // 17: P11
    0.2,           // 18: #11
    1.0,           // 19: P12
    0.7,           // 20: Min 13th
    0.85,          // 21: Maj 13th
    0.5,           // 22: Min 14th
    -100_000_000.0, // 23: Maj 14th (Clash - 23 % 12 == 11)
];

pub fn get_harmonic_score_adjusted(note_a: i32, note_b: i32) -> f64 {
    let dist = (note_a - note_b).abs();

    let effective_dist = if dist > 23 {
        12 + (dist % 12)
    } else {
        dist
    };
    
    // Safety clamp to 23 not strictly needed if logic is sound, but good for rust panic avoidance
    let idx = if effective_dist > 23 { 
        // fallback logic for very large distances if any?
        // Logic says 12 + (dist % 12), so max is 12 + 11 = 23.
        effective_dist 
    } else { 
        effective_dist 
    };

    INTERVAL_SCORES[idx as usize] as f64
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
