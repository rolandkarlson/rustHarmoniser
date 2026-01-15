use crate::model::{Note, Config};
use crate::utils::{SeededRng, ArrayExt, mod_shim, sin};
use crate::music_theory::{get_harmonic_score_adjusted, gen_scale};
use std::collections::HashMap;
use crate::schillinger;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

// Helper to get permutations of notes
pub fn get_permutations(notes: &[Note]) -> Vec<Vec<Note>> {
    let mut results = Vec::new();
    let mut notes = notes.to_vec();
    
    fn backtrack(current: Vec<Note>, remaining: Vec<Note>, results: &mut Vec<Vec<Note>>) {
        if remaining.is_empty() {
            results.push(current);
            return;
        }
        
        for i in 0..remaining.len() {
            let mut next_current = current.clone();
            next_current.push(remaining[i]);
            
            let mut next_remaining = remaining.clone();
            next_remaining.remove(i);
            
            backtrack(next_current, next_remaining, results);
        }
    }
    
    backtrack(Vec::new(), notes, &mut results);
    results
}

pub fn get_distance_score(prev_note: i32, current_note: i32) -> f64 {
    let dist = (prev_note - current_note).abs() as f64;
    if dist == 0.0 {
        return 1.0;
    }
    let max_jump = 7.0;
    if dist > max_jump {
        return -(dist * 10.0);
    }
    
    let score = 1.0 - (dist / max_jump);
    score.max(0.0)
}

#[derive(Clone)]
pub struct NoteScore {
    pub note: i32,
    pub score: f64,
    pub distance: f64,
    pub crossing: bool,
}

fn get_chords_on_position(notes: &[Note], position: f64) -> Vec<i32> {
    notes.iter()
        .filter(|n| n.start <= position && n.duration + n.start > position && n.muted == 0)
        .map(|n| n.pitch)
        .collect()
}

fn get_chords_on_exact_position(notes: &[Note], position: f64) -> Vec<Note> {
    // JS `getChordsOnExactPosition` returns `Note` objects.
     notes.iter()
        .filter(|n| (n.start - position).abs() < 0.001 && n.muted == 0)
        .cloned()
        .collect()
}

fn get_chords_on_exact_end_position(notes: &[Note], position: f64) -> Vec<Note> {
    notes.iter()
        .filter(|n| (n.start + n.duration - position).abs() < 0.001 && n.muted == 0)
        .cloned()
        .collect()
}


fn get_last_on_channel(notes: &[Note], position: f64, channel: i32, count: i32) -> Vec<i32> {
    let mut relevant: Vec<&Note> = notes.iter()
        .filter(|n| n.start < position && n.channel == channel && n.muted == 0)
        .collect();
    
    relevant.sort_by(|a, b| b.start.partial_cmp(&a.start).unwrap());
    
    let take_count = if count <= 0 { 1 } else { count as usize };
    relevant.into_iter().take(take_count).map(|n| n.pitch).collect()
}

struct Boundries {
    min: i32,
    max: i32,
}

fn get_chords_on_position_boundries(notes: &[Note], position: f64, channel: i32) -> Boundries {
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    for n in notes {
        if n.start <= position && n.duration + n.start > position && n.muted == 0 {
             if n.channel < channel {
                 upper.push(n.pitch);
             } else if n.channel > channel {
                 lower.push(n.pitch);
             }
        }
    }
    
    let min = if lower.is_empty() { 24 } else { *lower.iter().max().unwrap() }; // JS: max of lower is bound? Wait.
    // JS: `min = lower.length === 0 ? 24 : Math.max.apply(null, lower);`
    // Yes, max of lower notes acts as MINIMUM bound for current note (to avoid crossing below).
    
    let max = if upper.is_empty() { 90 } else { *upper.iter().min().unwrap() };
    
    Boundries { min, max }
}


pub struct HarmonizerState {
     pub schillinger_notes: Vec<Vec<i32>>,
}

fn get_schillinger_scale(current_note: &Note, state: &HarmonizerState) -> Vec<i32> {
    // JS: `var bar = Math.floor(currentNote.start / 4);`
    let bar = (current_note.start / 4.0).floor() as i32;
    
    // JS: `var notes = schillingerNotes[bar];`
    let safe_bar = mod_shim(bar, state.schillinger_notes.len() as i32) as usize;
    let notes = &state.schillinger_notes[safe_bar];

    if notes.len() < 2 {
        println!("Schillinger scale < 2: {:?}", notes);
    }
    
    // if current_note.channel > 2 {
    //     // JS: `return [notes[0], notes[2], notes[2]];`
    //     if notes.len() >= 3 {
    //          return vec![notes[0], notes[2], notes[2]];
    //     }
    // }
    notes.clone()
}

fn is_harmony_moving_to_same_direction(last: &[Note], current: &[Note], going_down: bool) -> bool {
    // Simplified port of `isHarmonyMovingToSameDirection`
    // notesToChannelPitchMap
     let mut last_map = HashMap::new();
     for n in last { last_map.insert(n.channel, n.pitch); }
     
     let mut cur_map = HashMap::new();
     for n in current { cur_map.insert(n.channel, n.pitch); }
     
     let mut up = 0;
     let mut down = 0;
     let mut compared = 0;
     
     for (ch, last_p) in last_map {
         if let Some(cur_p) = cur_map.get(&ch) {
             compared += 1;
             if cur_p > &last_p { up += 1; }
             else if cur_p < &last_p { down += 1; }
         }
     }
     
     if compared == 0 { return false; }
     
     if going_down { down > up } else { up > down }
}


pub fn get_harmony_scores(current_note: &Note, notes: &[Note], no_same_note_penalty: bool, config: &Config, state: &HarmonizerState) -> Vec<NoteScore> {
    let current_harmony = get_chords_on_position(notes, current_note.start);
    let current_on_same_start_harmony = get_chords_on_exact_position(notes, current_note.start);
    let current_on_same_end_harmony = get_chords_on_exact_end_position(notes, current_note.start);
    let last_harmony = get_chords_on_position(notes, current_note.start - 1.0); // -1 time
    let mut current_lasts = get_last_on_channel(notes, current_note.start, current_note.channel, 6);
    let bounds_p = get_chords_on_position_boundries(notes, current_note.start - 0.1, current_note.channel);
    let is_outer_voice = current_note.channel == 0 || current_note.channel == 3;

    if current_lasts.is_empty() {
        current_lasts.push(current_note.pitch);
    }

    let mut no_same_note_penalty = no_same_note_penalty;
    if current_lasts.len() >= 5 {
        let first_val = current_lasts[0];
        if current_lasts.iter().all(|&x| x == first_val) {
            no_same_note_penalty = false;
        }
    }


    // JS `lastHarmonySum` unused really?
    
    let sch_scale =[0,1,2,3,4,5,6,7,8,9,10,11];// get_schillinger_scale(current_note, state);
    let center_octave = (current_lasts[0] as f64 / 12.0).floor() as i32;
    let sc = gen_scale(&sch_scale, center_octave);
    
    let mut scores = Vec::new();

    for note_candidate in sc {
        let mut score = 0.0;
        let distance_sum = 0.0;
        let mut distance_score = 0.0;
        let mut harmony_score = 0.0;
        let mut crossing = false;

        // Same direction check
        if !current_on_same_end_harmony.is_empty() && !current_on_same_start_harmony.is_empty() && !current_lasts.is_empty() {
             if is_outer_voice {
                 // checking direction of harmony movement
                 // JS: `isHarmonyMovingToSameDirection(..., currentLasts[0] > noteCandidate)`
                 // `goingDown` arg is `currentLasts[0] > noteCandidate`
                 if is_harmony_moving_to_same_direction(&current_on_same_end_harmony, &current_on_same_start_harmony, current_lasts[0] > note_candidate) {
                     score -= config.same_direction;
                 }
             }
        }
        
        // Last harmony intervals check
        if !last_harmony.is_empty() {
             let mut last_harmony_intervals = Vec::new();
             for i in 0..last_harmony.len() {
                 for j in (i+1)..last_harmony.len() {
                     last_harmony_intervals.push((last_harmony[i] - last_harmony[j]).abs() % 12);
                 }
             }
             
             // Check consecutive optimization
             // `getScoreForConsecutive(7)` and `0`
             for &check_val in &[7, 0] {
                 if last_harmony_intervals.contains(&check_val) {
                     for ch_pitch in &current_harmony {
                        let md = (note_candidate - ch_pitch).abs() % 12;
                        if md == check_val {
                            score -= config.consecutive_octav_fift;
                        }
                     }
                 }
             }
        }
        
        let mut current_harmony_intervals = Vec::new();
        for i in 0..current_harmony.len() {
             for j in (i+1)..current_harmony.len() {
                 current_harmony_intervals.push((current_harmony[i] - current_harmony[j]).abs() % 12);
             }
        }

        if current_harmony.contains(&note_candidate) {
            score += -10000.0;
        }
        
        // if mod exists and < 4 notes
        if current_harmony.len() < 3 {
             // JS: `currentHarmony.modExists` logic
             for ch in &current_harmony {
                 if ch % 12 == note_candidate % 12 {
                     score += -10000.0;
                 }
             }
        }
        
        // Harmonic score loop
        let mut harm_sum = 0.0;
        for ch_pitch in &current_harmony {
             if current_harmony.len() < 4 {
                 let dif = (ch_pitch - note_candidate).abs() % 12;
                 if current_harmony_intervals.contains(&dif) {
                     score -= config.interval_exists_in_harmony;
                 }
             }
             harm_sum += get_harmonic_score_adjusted(note_candidate, *ch_pitch);
        }
        if !current_harmony.is_empty() {
            harmony_score += harm_sum / current_harmony.len() as f64;
        }

        // Boundries
        let d = bounds_p.max - note_candidate;
        let channel_boundry_max = [2,2,2,2,7][current_note.channel as usize];
        if d < channel_boundry_max {
             score -= config.no_crossing;
             crossing = true;
        }
        let d2 = note_candidate - bounds_p.min;
        let channel_boundry_min = [2,2,2,7,1][current_note.channel as usize];
        if d2 < channel_boundry_min {
            score -= config.no_crossing;
            crossing = true;
        }
        
        // Distance score with last note
        if !current_lasts.is_empty() {
             let last_note = current_lasts[0];
             
             if current_lasts.contains(&note_candidate) && !no_same_note_penalty {
                 // count occurrences
                 let count = current_lasts.iter().filter(|&&x| x == note_candidate).count();
                 score -= config.last_note_exist_in_voice * count as f64;
             }
             
             if last_note == note_candidate && !no_same_note_penalty {
                  score -= config.last_note_same;
             }
             
             // let base_dist = (note_candidate - current_note.pitch).abs() as f64;
             // score -= (base_dist / 7.0).powf(3.0);
             
             distance_score = get_distance_score(last_note, note_candidate);
        }
        let r = 0.0;
        let w_harmony = 0.5+r;
        let w_smooth = 0.5-r;
        let sum_score = (harmony_score * w_harmony) + (distance_score * w_smooth) + score;
        
        scores.push(NoteScore {
            note: note_candidate,
            score: sum_score,
            distance: distance_sum,
            crossing
        });
    }

    scores
}


pub fn gen_voice(base: i32, rhythm_data: &Vec<f64>, pitch_shifts: &[i32], channel: i32, muted: i32, rng: &mut SeededRng) -> Vec<Note> {
    let mut ar = Vec::new();
    let clip_len = schillinger::CLIP_LEN as f64; 
    let mut pos = 0.0;
    let mut counter = 0;
    let sf = (rng.random_int(60) + 1) as f64;
    
    while pos < clip_len {
        let n = base + pitch_shifts[mod_shim(counter, pitch_shifts.len() as i32) as usize];
        let d = rhythm_data[mod_shim(counter, rhythm_data.len() as i32) as usize];
        let v = 1 + rng.random_int(10) + sin(counter as f64, sf, 10.0) as i32;
        ar.push(Note::new(n, pos, d, v, muted, channel));
        pos += d;
        counter += 1;
    }
    
    ar
}

fn group_by_start_array(notes: Vec<Note>) -> Vec<Vec<Note>> {
    let mut map: HashMap<String, Vec<Note>> = HashMap::new(); 
    let quantize = |f: f64| format!("{:.4}", f);
    
    for n in notes {
        let key = quantize(n.start);
        map.entry(key).or_insert(Vec::new()).push(n);
    }
    
    let mut groups: Vec<Vec<Note>> = map.into_values().collect();
    groups.sort_by(|a, b| a[0].start.partial_cmp(&b[0].start).unwrap());
    
    for g in &mut groups {
        g.sort_by(|a, b| b.pitch.cmp(&a.pitch));
    }
    
    groups
}

#[derive(Clone)]
struct BeamCandidate {
    notes: Vec<Note>,
    score: f64,
}

fn score_note_group(
    current_notes_in: &[Note],
    trimmed_notes: &[Note],
    temp_group_notes: &mut Vec<Note>,
    no_same_note_penalty: bool,
    config: &Config,
    state: &HarmonizerState,
) -> f64 {
    let mut group_score = 0.0;
    let mut current_notes = current_notes_in.to_vec();
    let permu_first_channel = current_notes.last().unwrap().channel; // JS: currentNotes[currentNotes.length - 1].channel

    for j in 0..current_notes.len() {
        let skip_penalty = no_same_note_penalty || (permu_first_channel != current_notes[j].channel);
        
        // Context is trimmed_notes + temp_group_notes
        let mut context = trimmed_notes.to_vec();
        context.extend(temp_group_notes.iter().cloned());
        
        let mut scores = get_harmony_scores(&current_notes[j], &context, skip_penalty, config, state);
        scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        if let Some(best) = scores.first() {
            current_notes[j].pitch = best.note;
            current_notes[j].muted = 0;
            temp_group_notes.push(current_notes[j]);
            group_score += best.score;
        } else {
             // Fallback if no scores (shouldn't happen with correct logic)
             temp_group_notes.push(current_notes[j]);
        }
    }
    group_score
}


fn score_lookahead(
    grouped_notes: &[Vec<Note>],
    start_idx: usize,
    depth: i32,
    context: &[Note],
    config: &Config,
    state: &HarmonizerState,
    cache: &mut HashMap<u64, f64>,
) -> f64 {
    if depth == 0 || start_idx >= grouped_notes.len() {
        return 0.0;
    }

    // Cache key generation using Hash
    let context_len = context.len();
    let suffix_len = if context_len > 4 { 4 } else { context_len };
    let suffix = &context[context_len - suffix_len..];
    
    let mut hasher = DefaultHasher::new();
    start_idx.hash(&mut hasher);
    depth.hash(&mut hasher);
    for n in suffix {
        n.pitch.hash(&mut hasher);
        n.channel.hash(&mut hasher);
    }
    let key = hasher.finish();
    
    if let Some(&val) = cache.get(&key) {
        return val;
    }

    let current_group = &grouped_notes[start_idx];
    let permutations = get_permutations(current_group);
    
    // 1. Calculate local scores for all permutations
    let mut candidates = Vec::with_capacity(permutations.len());
    
    // Optimization: Limit lookahead check to a subset of permutations to avoid factorial explosion
    // Taking first 20 permutations is enough to get a heuristic estimate
    for perm in permutations.iter().take(20) {
        let mut temp_notes = Vec::new();
        let score = score_note_group(perm, context, &mut temp_notes, false, config, state);
        candidates.push((score, temp_notes));
    }

    // 2. Prune: Sort and take top K
    // Sort descending
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    
    let k = 2; // Pruning width for lookahead (Reduced from 5 for performance)
    let top_candidates = candidates.into_iter().take(k);

    let mut best_score = -f64::INFINITY;

    // 3. Recurse only on top K
    for (local_score, temp_notes) in top_candidates {
        let mut next_context = context.to_vec();
        next_context.extend(temp_notes);
        
        // Recurse
        let total_score = local_score + score_lookahead(grouped_notes, start_idx + 1, depth - 1, &next_context, config, state, cache);

        if total_score > best_score {
            best_score = total_score;
        }
    }

    cache.insert(key, best_score);
    best_score
}

fn score_group_beam(income: Vec<Note>, config: &Config, state: &HarmonizerState) -> Vec<Note> {
    let grouped_notes = group_by_start_array(income);
    let beam_width = 5;
    let lookahead = 5; // Lookahead depth

    let mut beam = vec![BeamCandidate {
        notes: Vec::new(),
        score: 0.0,
    }];
    let mut ccc = 0.0;
    // JS: candidates loop through groups
    for (i, current_group) in grouped_notes.iter().enumerate() {
        let permutations = get_permutations(current_group);
        let grouped_notes_ref = &grouped_notes;
        let mut candidates = Vec::new();
        
        // Cache for this step's lookahead to share across beam candidates if they converge?
        // Actually lookahead cache depends on context.
        // We can share a cache across the whole beam step or even persistent?
        // "start_idx" is fixed per step.
        // So we can create a new cache for each group step to avoid memory explosion, 
        // OR share it?
        // Contexts will be different for each beam candidate.
        // But if they converge (same last notes), we hit cache.
        let mut cache: HashMap<u64, f64> = HashMap::new();

        for beam_state in &beam {
            // trim notes to last 30
            let start = if beam_state.notes.len() > 30 { beam_state.notes.len() - 30 } else { 0 };
            let trimmed_notes = &beam_state.notes[start..];

            for perm in &permutations {
                 let mut temp_notes = Vec::new();
                 // score_note_group requires mutable temp_notes to push results
                 let group_score = score_note_group(perm, trimmed_notes, &mut temp_notes, false, config, state);

                 let mut next_context = trimmed_notes.to_vec();
                 next_context.extend(temp_notes.clone());
                 
                 let lookahead_score = score_lookahead(&grouped_notes, i + 1, lookahead, &next_context, config, state, &mut cache);
                 
                 let mut new_notes = beam_state.notes.clone();
                 new_notes.extend(temp_notes);
                 
                 candidates.push(BeamCandidate {
                     notes: new_notes,
                     score: beam_state.score + group_score + lookahead_score,
                 });
            }
        }
        
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        beam = candidates.into_iter().take(beam_width).collect();
        
        println!("Processed group {}/{}, best score: {}", i, grouped_notes.len(), beam[0].score - ccc);
        ccc = beam[0].score;
    }
    
    if beam.is_empty() {
        return Vec::new();
    }
    println!("Final Score: {}", beam[0].score);
    beam[0].notes.clone()
}

pub fn harmonise2(income: Vec<Note>, config: &Config, state: &HarmonizerState) -> Vec<Note> {
    score_group_beam(income, config, state)
}
