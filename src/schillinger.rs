use crate::utils::{mod_shim, SeededRng, ArrayExt};
use crate::music_theory::{generate_mode_from_steps};

// PL = 8 in JS
const PL: i32 = 4;
const EXP: i32 = 2;

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

pub fn gen_schillinger_progression(config_seq: &[i32]) -> Vec<Vec<i32>> {
    let bars = PL;

    let seq_opt = find_sequence_with_condition(&[-4, -4, -4, -2, -2,-6], PL);
    //let seq = seq_opt.unwrap_or(vec![0; PL as usize]); // Fallback? JS logs error and returns null.
    let seq = config_seq;//

    // We'll trust RNG seed matches or just handle it.

    // `genSchillingerProgression` logic:
    let mut chord_notes = Vec::new();
    let mut root_sequence = 0;
    let scale = generate_mode_from_steps(0, 5);
    
    // JS `get` is wrap around access.
    let n_struct_base = vec![0, 1, 2];
    let ex_base = vec![2]; // `var ex = [2].get(i);`

    for i in 0..bars {
        let n_struct = &n_struct_base;
        let ex = 2;
        let notes: Vec<i32> = n_struct.iter().map(|&itm| {
             let idx = (itm * ex) + seq[i as usize%seq.len()];
             // scale.get(idx)
             scale[mod_shim(idx, scale.len() as i32) as usize]
        }).collect();

        chord_notes.push(notes);
    }
    
    chord_notes
}
