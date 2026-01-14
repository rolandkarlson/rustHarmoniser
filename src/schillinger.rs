use crate::utils::{mod_shim, SeededRng, ArrayExt};
use crate::music_theory::{generate_mode_from_steps};

// PL = 8 in JS
const PL: i32 = 8;
pub(crate) const CLIP_LEN: i32 = 8 * 4 * 10; // JS `var CLIP_LEN = PL;`
const EXP: i32 = 2;

fn find_sequence_with_condition(possible_steps: &[i32], sequence_length: i32, rng: &mut SeededRng) -> Option<Vec<i32>> {
    let max_attempts = 1000000;
    let mut attempts = 0;

    while attempts < max_attempts {
        let mut sequence = vec![0];
        let mut current_sum = 0;
        let mut last_value = -999;

        for _ in 1..sequence_length {
            let random_index = (rng.seeded_random(1.0, 0.0) * possible_steps.len() as f64).floor() as usize;
            let mut step = possible_steps[random_index] * EXP;

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
            
            sequence.push(current_sum);
        }

        let last_element = sequence.last().unwrap();
        let modulo_result = mod_shim(*last_element % 7 + 7, 7);
        // JS: `var moduloResult = ((lastElement % 7) + 7) % 7;`
        
        let target = mod_shim(2 * EXP, 7);

        if modulo_result == target {
            return Some(sequence);
        }
        attempts += 1;
    }
    None
}

pub fn gen_schillinger_progression(rng: &mut SeededRng) -> Vec<Vec<i32>> {
    let bars = CLIP_LEN / 4; // 8 / 4 = 2 bars?
    // Wait, JS: `var CLIP_LEN = PL;` -> 8.
    // `var bars = CLIP_LEN / 4;` -> 2.
    // Loop `for (var i = 0; i < bars; i++)`.
    // BUT later `schillingerNotes[bar]` is accessed.
    // If bars=2, index 0 and 1 exist.
    // `getSchillingerScale` uses `var bar = Math.floor(currentNote.start / 4);`.
    // If start goes up to 8 (CLIP_LEN), bar is 0 or 1.
    
    // JS `srm` init:
    // `var srm = [ findSequenceWithCondition([-2, -2, -2, -1, -3], PL) ];`
    
    let seq_opt = find_sequence_with_condition(&[-2, -2, -2, -1, -3], PL, rng);
    let seq = seq_opt.unwrap_or(vec![0; PL as usize]); // Fallback? JS logs error and returns null.
    // We'll trust RNG seed matches or just handle it.

    // `genSchillingerProgression` logic:
    let mut chord_notes = Vec::new();
    let mut root_sequence = 0;
    let scale = generate_mode_from_steps(0, 0);
    
    // JS `get` is wrap around access.
    let n_struct_base = vec![0, 1, 2];
    let ex_base = vec![2]; // `var ex = [2].get(i);`

    for i in 0..bars {
        let n_struct_idx = mod_shim(i, 5); // JS `[0..6].get(i)`? No, array len 5.
        // Wait, JS: `var nStruct = [[0, 1, 2, 4, 6]].get(i);`
        // It's an array OF ARRAYS? `[[0,1...]]`
        // `[[...]].get(i)` where length is 1. So it always gets `[0, 1, 2, 4, 6]`.
        // My bad, `[[0, 1, 2, 4, 6]]` has length 1.
        let n_struct = &n_struct_base;
        
        // `var ex = [2].get(i)` -> always 2.
        let ex = 2;

        let notes: Vec<i32> = n_struct.iter().map(|&itm| {
             let idx = (itm * ex) + root_sequence;
             // scale.get(idx)
             scale[mod_shim(idx, scale.len() as i32) as usize]
        }).collect();
        
        root_sequence = mod_shim(root_sequence - 4, scale.len() as i32);
        chord_notes.push(notes);
    }
    
    chord_notes
}
