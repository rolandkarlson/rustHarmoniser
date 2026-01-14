use crate::utils::SeededRng;

pub fn rotate_seq(seq: &[i32], steps: i32, rotate: i32) -> Vec<i32> {
    let len = seq.len() as i32;
    let val = len - rotate;
    let mut output = vec![0; seq.len()];
    for i in 0..seq.len() {
        let idx = ((i as i32 + val).abs() % len) as usize;
        output[i] = seq[idx];
    }
    output
}

pub fn euclid(steps: i32, pulses: i32, rotate: i32) -> Vec<i32> {
    let mut stored_rhythm = Vec::new();
    let mut bucket = 0;
    for _ in 0..steps {
        bucket += pulses;
        if bucket >= steps {
            bucket -= steps;
            stored_rhythm.push(1);
        } else {
            stored_rhythm.push(0);
        }
    }
    
    // In JS: rotate += 1; rotate % steps; 
    let r = (rotate + 1) % steps; // simplistic port
    if r > 0 {
        stored_rhythm = rotate_seq(&stored_rhythm, steps, r);
    }
    stored_rhythm
}


pub fn transform_rhythm(rhythm: &[f64], split_chance: f64, merge_chance: f64, rng: &mut SeededRng) -> Vec<f64> {
    let mut new_rhythm = Vec::new();
    let mut skip_next = false;
    let max_duration = 4.0;
    let min_duration = 0.25;

    for i in 0..rhythm.len() {
        if skip_next {
            skip_next = false;
            continue;
        }

        let note = rhythm[i];
        
        // Merge Logic
        if i < rhythm.len() - 1 {
             if (rhythm[i+1] - note).abs() < 0.001 { // compare floats equality
                 if (note * 2.0) <= max_duration {
                     if rng.seeded_random(1.0, 0.0) < merge_chance {
                         new_rhythm.push(note * 2.0);
                         skip_next = true;
                         continue;
                     }
                 }
             }
        }

        // Split Logic
        if (note / 2.0) >= min_duration {
            if rng.seeded_random(1.0, 0.0) < split_chance {
                let half = note / 2.0;
                new_rhythm.push(half);
                new_rhythm.push(half);
                continue;
            }
        }

        new_rhythm.push(note);
    }
    new_rhythm
}

pub fn gen_rythm2(len: f64, pn: &Vec<f64>, rng: &mut SeededRng) -> Vec<f64> {
    let mut ret = Vec::new();
    let mut current_len = len;
    for _ in 0..300 {
        let r_idx = rng.random_int(10) as usize; 
        // pn.get(randomInt) -> checks bound? JS array.get is mod.
        // We need to handle mod access manually or use a helper
        let r_val = pn[ (r_idx as i32 % pn.len() as i32) as usize ]; // simplistic mod
        
        let tl = current_len - r_val;
        if tl <= 0.0 {
            if current_len > 0.0 {
                ret.push(current_len);
            }
            return ret;
        } else {
            ret.push(r_val);
            current_len = tl;
        }
    }
    ret
}
