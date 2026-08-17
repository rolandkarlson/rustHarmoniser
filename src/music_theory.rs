pub fn gen_scale(ap: &[i32], center_octave: i32) -> Vec<i32> {
    if ap.is_empty() {
        return Vec::new();
    }

    let center = center_octave.clamp(0, 10);
    let start = (center - 1).max(0);
    let end = (center + 1).min(10);
    let mut ar = Vec::new();

    for i in start..=end {
        for &note in ap {
            let pitch = note + 12 * i;
            if (0..=127).contains(&pitch) {
                ar.push(pitch);
            }
        }
    }

    ar.sort_unstable();
    ar.dedup();
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
    //
    if dist % 12 == 1 || dist % 12 == 11 {
        // "Clash" penalty in JS was -100000000, here we return 0.0 or handle it in mapping
        // JS returned -100000000.
        return -100000000.0;
    }


    let score: f64 = match dist % 12 {
        0 => 1.0,   // Unison
        1 => -1.0,   // Min 2nd
        2 => -1.0,   // Maj 2nd
        3 => 0.6,   // Min 3rd
        4 => 0.8,   // Maj 3rd
        5 => 0.7,   // P4
        6 => -100.0,   // Tritone
        7 => 1.0,   // P5
        8 => 0.7,   // Min 6th
        9 => 0.8,   // Maj 6th
        10 => -1.0,  // Min 7th
        11 => -1.0,  // Maj 7th
        _ => 0.0,
    };
    return score;

    // let score: f64 = match effective_dist {
    //     0 => 1.0,   // Unison
    //     1 => -1.0,   // Min 2nd
    //     2 => 0.2,   // Maj 2nd
    //     3 => 0.6,   // Min 3rd
    //     4 => 0.8,   // Maj 3rd
    //     5 => 0.7,   // P4
    //     6 => 0.0,   // Tritone
    //     7 => 1.0,   // P5
    //     8 => 0.7,   // Min 6th
    //     9 => 0.8,   // Maj 6th
    //     10 => 0.3,  // Min 7th
    //     11 => 0.4,  // Maj 7th
    //     12 => 1.0,  // Octave
    //     13 => -1.0,  // Min 9th
    //     14 => 0.85, // Maj 9th
    //     15 => 0.7,  // Min 10th
    //     16 => 0.9,  // Maj 10th
    //     17 => 0.7,  // P11
    //     18 => 0.2,  // #11
    //     19 => 1.0,  // P12
    //     20 => 0.7,  // Min 13th
    //     21 => 0.85, // Maj 13th
    //     22 => 0.5,  // Min 14th
    //     23 => 0.6,  // Maj 14th
    //     _ => 0.0,
    // };

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

    //score.clamp(0.0, 1.0)
}


/// Pitch classes of `mode` (0 = Ionian … 6 = Locrian) built on `root`, returned
/// in SCALE ORDER: index 0 is the tonic, index n is scale degree n+1.
///
/// The result is deliberately NOT sorted numerically. Sorting it (as this used
/// to) rotates the degree mapping by however many pitch classes wrap past 12:
/// for root = 2 Ionian the notes 2,4,6,7,9,11,1 sorted to [1,2,4,6,7,9,11], so
/// `scale[0]` was C# rather than D and every degree index downstream — chord
/// roots in `gen_schillinger_progression`, `notes[0..2]` in the `use_resolve`
/// bass logic — pointed at the wrong scale degree. In other words, changing the
/// key silently changed the mode. Callers that want ascending MIDI pitches get
/// them from `gen_scale`, which sorts the realised pitches anyway.
pub fn generate_mode_from_steps(root: i32, mode: &i32) -> Vec<i32> {
    let step_pattern = vec![2, 2, 1, 2, 2, 2, 1];

    // rotate
    let steps_rot = if *mode > 0 {
         let split_idx = *mode as usize  % step_pattern.len();
         let (left, right) = step_pattern.split_at(split_idx);
         [right, left].concat()
    } else {
        step_pattern.clone()
    };

    // remove last? JS: modePattern.pop();
    // Actually we need to walk it.
    let mut mode_pattern = steps_rot;
    mode_pattern.pop();

    let mut current_note = root.rem_euclid(12);
    let mut mode_notes = vec![current_note];

    for step in mode_pattern {
        current_note = (current_note + step).rem_euclid(12);
        mode_notes.push(current_note);
    }
    mode_notes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_starts_on_the_tonic_for_every_root() {
        for root in 0..12 {
            for mode in 0..7 {
                let s = generate_mode_from_steps(root, &mode);
                assert_eq!(s.len(), 7);
                assert_eq!(s[0], root, "root {root} mode {mode} does not start on the tonic");
            }
        }
    }

    #[test]
    fn mode_is_a_transposition_of_the_same_mode_at_c() {
        // Degree-for-degree, D dorian is C dorian transposed up two semitones.
        let at_c = generate_mode_from_steps(0, &1);
        let at_d = generate_mode_from_steps(2, &1);
        for i in 0..7 {
            assert_eq!(at_d[i], (at_c[i] + 2).rem_euclid(12), "degree {i}");
        }
    }

    #[test]
    fn ionian_and_aeolian_have_the_expected_degrees() {
        assert_eq!(generate_mode_from_steps(0, &0), vec![0, 2, 4, 5, 7, 9, 11]);
        assert_eq!(generate_mode_from_steps(0, &5), vec![0, 2, 3, 5, 7, 8, 10]);
    }
}
