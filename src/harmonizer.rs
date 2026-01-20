use crate::model::{Note, Config};
use crate::utils::{SeededRng, ArrayExt, mod_shim, sin};
use crate::music_theory::{get_harmonic_score_adjusted, gen_scale};
use crate::schillinger;
use dashmap::DashMap;
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use fxhash::FxHasher64;

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

#[derive(Clone, Debug)]
pub struct Boundries {
    pub min: i32,
    pub max: i32,
}

pub struct HarmonizerState {
    pub schillinger_notes: Vec<Vec<i32>>,
}

fn get_schillinger_scale(current_note: &Note, state: &HarmonizerState) -> Vec<i32> {
    let bar = (current_note.start / 4.0).floor() as i32;
    let safe_bar = mod_shim(bar, state.schillinger_notes.len() as i32) as usize;
    let notes = &state.schillinger_notes[safe_bar];
    notes.clone()
}

fn is_harmony_moving_to_same_direction(last: &[Note], current: &[Note], going_down: bool) -> bool {
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

pub struct PrecomputedHarmonyData {
    pub last_harmony: Vec<i32>,
    pub last_harmony_intervals: Vec<i32>,
    pub sustaining_notes: Vec<i32>,
    pub boundries_by_channel: Vec<Boundries>,
    pub last_notes_by_channel: Vec<Vec<i32>>,
    pub notes_ending_at_start: Vec<Note>,
}

fn build_precomputed_data(context: &[Note], start_time: f64) -> PrecomputedHarmonyData {
    let mut last_harmony = Vec::new();
    let mut sustaining_notes = Vec::new();
    let mut notes_ending_at_start = Vec::new();
    let mut sustaining_at_minus_0_1 = Vec::new();
    let mut last_notes_by_channel: Vec<Vec<i32>> = vec![Vec::new(); 16];

    for n in context {
        // last_harmony: start <= start-1.0 && end > start-1.0
        if n.start <= start_time - 1.0 && n.start + n.duration > start_time - 1.0 && n.muted == 0 {
            last_harmony.push(n.pitch);
        }

        // sustaining_notes: start <= start && end > start
        if n.start <= start_time && n.start + n.duration > start_time && n.muted == 0 {
            sustaining_notes.push(n.pitch);
        }

        // sustaining_at_minus_0_1 (for boundaries): start <= start-0.1 && end > start-0.1
        if n.start <= start_time - 0.1 && n.start + n.duration > start_time - 0.1 && n.muted == 0 {
            sustaining_at_minus_0_1.push(n);
        }

        // notes_ending_at_start: end == start
        if (n.start + n.duration - start_time).abs() < 0.001 && n.muted == 0 {
            notes_ending_at_start.push(n.clone());
        }

        // last_notes: start < start
        if n.start < start_time && n.muted == 0 {
            let ch = n.channel as usize;
            if ch < 16 {
                last_notes_by_channel[ch].push(n.pitch);
            }
        }
    }

    for notes in &mut last_notes_by_channel {
        if notes.len() > 5 {
            let len = notes.len();
            *notes = notes[len-5..].to_vec();
        }
        notes.reverse();
    }

    let mut boundries_by_channel = Vec::with_capacity(16);
    for ch in 0..16 {
        let mut lower = Vec::new();
        let mut upper = Vec::new();
        for n in &sustaining_at_minus_0_1 {
            if n.channel < ch as i32 {
                upper.push(n.pitch);
            } else if n.channel > ch as i32 {
                lower.push(n.pitch);
            }
        }
        let min = if lower.is_empty() { 24 } else { *lower.iter().max().unwrap() };
        let max = if upper.is_empty() { 90 } else { *upper.iter().min().unwrap() };
        boundries_by_channel.push(Boundries { min, max });
    }

    let last_harmony_intervals = {
        let len = last_harmony.len();
        let mut intervals = Vec::with_capacity(if len > 0 { len * (len - 1) / 2 } else { 0 });
        for i in 0..last_harmony.len() {
            for j in (i+1)..last_harmony.len() {
                intervals.push((last_harmony[i] - last_harmony[j]).abs() % 12);
            }
        }
        intervals
    };

    PrecomputedHarmonyData {
        last_harmony,
        last_harmony_intervals,
        sustaining_notes,
        boundries_by_channel,
        last_notes_by_channel,
        notes_ending_at_start,
    }
}

pub fn get_harmony_scores(
    current_note: &Note,
    current_on_same_start_harmony: &[Note],
    no_same_note_penalty: bool,
    config: &Config,
    state: &HarmonizerState,
    precomputed: &PrecomputedHarmonyData
) -> Vec<NoteScore> {

    let last_harmony = &precomputed.last_harmony;
    let last_harmony_intervals = &precomputed.last_harmony_intervals;
let channel_idx = current_note.channel as usize;
    let mut current_harmony = precomputed.sustaining_notes.clone();
    for n in current_on_same_start_harmony {
         current_harmony.push(n.pitch);
    }

    let current_on_same_end_harmony = &precomputed.notes_ending_at_start;

    let mut current_lasts = if channel_idx < precomputed.last_notes_by_channel.len() {
        precomputed.last_notes_by_channel[channel_idx].clone()
    } else {
        Vec::new()
    };

    let bounds_p = if channel_idx < precomputed.boundries_by_channel.len() {
        &precomputed.boundries_by_channel[channel_idx]
    } else {
        &Boundries { min: 24, max: 90 }
    };

    let is_outer_voice = current_note.channel == 0 || current_note.channel == 3;

    if current_lasts.is_empty() {
        current_lasts.push(current_note.pitch);
    }

    let mut no_same_note_penalty = no_same_note_penalty;
    if current_lasts.len() >= 4 {
        let first_val = current_lasts[0];
        if current_lasts.iter().all(|&x| x == first_val) {
            no_same_note_penalty = false;
        }
    }

    // Inline seq array
    let seq_arr = [
        [0,3,12,1], [0,3,-5,1], [0,3,4,1], [0,3,4,1],[0,3,4,1]
    ];
    let seq_row = seq_arr.get_wrapped(channel_idx);
    let seq = seq_row.get_wrapped((current_note.start / (4.0*8.0)) as usize);

    let last_note = if !current_lasts.is_empty() { current_lasts[0] } else { current_note.pitch };
    let range = 3;
    let min_pitch = (last_note - range).max(24);
    let max_pitch = (last_note + range).min(96);

    let current_harmony_intervals: Vec<i32> = {
        let len = current_harmony.len();
        let mut intervals = Vec::with_capacity(if len > 0 { len * (len - 1) / 2 } else { 0 });
        for i in 0..current_harmony.len() {
            for j in (i+1)..current_harmony.len() {
                intervals.push((current_harmony[i] - current_harmony[j]).abs() % 12);
            }
        }
        intervals
    };

    let has_interval_7 = last_harmony_intervals.contains(&7);
    let has_interval_0 = last_harmony_intervals.contains(&0);

    let channel_boundry_max = [2,2,2,2,7].get_wrapped(channel_idx);
    let channel_boundry_min = [2,2,2,7,1].get_wrapped(channel_idx);

    let current_harmony_len = current_harmony.len();
    let mut scores = Vec::with_capacity((max_pitch - min_pitch + 1) as usize);

    for note_candidate in min_pitch..=max_pitch {
        let mut score = 0.0;
        let mut distance_score = 0.0;
        let mut harmony_score = 0.0;
        let mut crossing = false;

        // Same direction check
        if !current_on_same_end_harmony.is_empty() && !current_on_same_start_harmony.is_empty() && !current_lasts.is_empty() {
            if is_outer_voice {
                if is_harmony_moving_to_same_direction(current_on_same_end_harmony, current_on_same_start_harmony, current_lasts[0] > note_candidate) {
                    score -= config.same_direction;
                }
            }
        }

        if !last_harmony.is_empty() {
            for &check_val in &[7, 0] {
                let has_interval = if check_val == 7 { has_interval_7 } else { has_interval_0 };
                if has_interval {
                    for ch_pitch in &current_harmony {
                        let md = (note_candidate - ch_pitch).abs() % 12;
                        if md == check_val {
                            score -= config.consecutive_octav_fift;
                        }
                    }
                }
            }
        }

        if current_harmony.contains(&note_candidate) {
            score += -10000.0;
        }

        if current_harmony_len < 3 {
            for ch in &current_harmony {
                if ch % 12 == note_candidate % 12 {
                    score += -10000.0;
                }
            }
        }

        let mut harm_sum = 0.0;
        for ch_pitch in &current_harmony {
            if current_harmony_len < 3 {
                let dif = (ch_pitch - note_candidate).abs() % 12;
                if current_harmony_intervals.contains(&dif) {
                    score -= config.interval_exists_in_harmony;
                }
            }
            harm_sum += get_harmonic_score_adjusted(note_candidate, *ch_pitch);
        }
        if current_harmony_len > 0 {
            harmony_score += harm_sum / current_harmony_len as f64;
        }

        let d = bounds_p.max - note_candidate;
        if d < *channel_boundry_max {
            score -= config.no_crossing;
            crossing = true;
        }
        let d2 = note_candidate - bounds_p.min;
        if d2 < *channel_boundry_min {
            score -= config.no_crossing;
            crossing = true;
        }

        if !current_lasts.is_empty() {
            if current_lasts.contains(&note_candidate) && !no_same_note_penalty {
                let count = current_lasts.iter().filter(|&&x| x == note_candidate).count();
                if count >= 2 {
                    score -= config.last_note_exist_in_voice
                }
            }

            if last_note == note_candidate && !no_same_note_penalty {
                score -= config.last_note_same;
            }

            let base_dist = (note_candidate - current_note.pitch + seq).abs() as f64;
            let normalized = base_dist / 8.0;
            score -= normalized * normalized * normalized;

            distance_score = get_distance_score(last_note, note_candidate);
        }
        let r = 0.1;

        let w_harmony = 0.5+r;
        let w_smooth = 0.5-r;
        let sum_score = (harmony_score * w_harmony) + (distance_score * w_smooth) + score;
        

        scores.push(NoteScore {
            note: note_candidate,
            score: sum_score,
            distance: 0.0,
            crossing
        });
    }

    scores
}


pub fn gen_voice(base: i32, rhythm_data: &Vec<f64>, pitch_shifts: &[i32], channel: i32, muted: i32) -> Vec<Note> {
    let mut ar = Vec::new();
    let clip_len = schillinger::CLIP_LEN as f64;
    let mut pos = 0.0;
    let mut counter = 0;
    let sf = (SeededRng::random_int(60) + 1) as f64;

    while pos < clip_len {
        let n = base + pitch_shifts[mod_shim(counter, pitch_shifts.len() as i32) as usize];
        let d = rhythm_data[mod_shim(counter, rhythm_data.len() as i32) as usize];
        let v = 1 + SeededRng::random_int(10) + sin(counter as f64, sf, 10.0) as i32;
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

struct IntermediateCandidate {
    parent_idx: usize,
    added_notes: Vec<Note>,
    score: f64,
}

fn score_note_group(
    current_notes_in: &[Note],
    temp_group_notes: &mut Vec<Note>,
    no_same_note_penalty: bool,
    config: &Config,
    state: &HarmonizerState,
    precomputed: &PrecomputedHarmonyData
) -> f64 {
    let mut group_score = 0.0;
    let mut current_notes = current_notes_in.to_vec();
    let permu_first_channel = current_notes.last().unwrap().channel;

    for j in 0..current_notes.len() {
        let skip_penalty = no_same_note_penalty || (permu_first_channel != current_notes[j].channel);

        // temp_group_notes acts as current_on_same_start_harmony
        let mut scores = get_harmony_scores(&current_notes[j], temp_group_notes, skip_penalty, config, state, precomputed);
        scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        if let Some(best) = scores.first() {
            current_notes[j].pitch = best.note;
            current_notes[j].muted = 0;
            temp_group_notes.push(current_notes[j]);
            group_score += best.score;
        } else {
            temp_group_notes.push(current_notes[j]);
        }
    }
    group_score
}


fn score_lookahead(
    all_permutations: &[Vec<Vec<Note>>],
    start_idx: usize,
    depth: i32,
    context: &[Note],
    config: &Config,
    state: &HarmonizerState,
    cache: &DashMap<u64, f64>,
) -> f64 {
    if depth == 0 || start_idx >= all_permutations.len() {
        return 0.0;
    }

    let context_len = context.len();
    let suffix_len = if context_len > 10 { 10 } else { context_len };
    let suffix = &context[context_len - suffix_len..];

    let mut hasher = FxHasher64::default();
    start_idx.hash(&mut hasher);
    depth.hash(&mut hasher);
    for n in suffix {
        n.pitch.hash(&mut hasher);
        n.channel.hash(&mut hasher);
    }
    let key = hasher.finish();

    if let Some(val) = cache.get(&key) {
        return *val;
    }

    let permutations = &all_permutations[start_idx];

    let start_time = permutations[0][0].start;
    let precomputed = build_precomputed_data(context, start_time);

    let candidates: Vec<(f64, Vec<Note>)> = permutations.par_iter()
        .map(|perm| {
            let mut temp_notes = Vec::new();
            let score = score_note_group(perm, &mut temp_notes, false, config, state, &precomputed);
            (score, temp_notes)
        })
        .collect();

    let mut sorted_candidates = candidates;
    sorted_candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let k = 2;
    let top_candidates: Vec<_> = sorted_candidates.into_iter().take(k).collect();

    let best_score = top_candidates.par_iter()
        .map(|(local_score, temp_notes)| {
            let mut next_context = context.to_vec();
            next_context.extend(temp_notes.iter().cloned());

            *local_score + score_lookahead(all_permutations, start_idx + 1, depth - 1, &next_context, config, state, cache)
        })
        .reduce(|| -f64::INFINITY, |a, b| a.max(b));

    cache.insert(key, best_score);
    best_score
}

fn score_group_beam(income: Vec<Note>, config: &Config, state: &HarmonizerState) -> Vec<Note> {
    let grouped_notes = group_by_start_array(income);

    let all_permutations: Vec<Vec<Vec<Note>>> = grouped_notes.par_iter()
        .map(|g| get_permutations(g))
        .collect();

    let beam_width = 5;
    let lookahead = 3;

    let mut beam = vec![BeamCandidate {
        notes: Vec::new(),
        score: 0.0,
    }];
    let mut ccc = 0.0;

    for (i, _) in grouped_notes.iter().enumerate() {
        let permutations = &all_permutations[i];

        let cache: DashMap<u64, f64> = DashMap::new();
        let all_permutations_ref = &all_permutations;
        let cache_ref = &cache;

        let current_beam = &beam;

        let mut candidates: Vec<IntermediateCandidate> = current_beam
            .par_iter()
            .enumerate()
            .flat_map(|(parent_idx, beam_state)| {
                let start = if beam_state.notes.len() > 30 {
                    beam_state.notes.len() - 30
                } else {
                    0
                };
                let trimmed_notes = &beam_state.notes[start..];

                let start_time = permutations[0][0].start;
                let precomputed = build_precomputed_data(trimmed_notes, start_time);

                permutations.par_iter().map(move |perm| {
                    let mut temp_notes = Vec::new();
                    let group_score =
                        score_note_group(perm, &mut temp_notes, false, config, state, &precomputed);

                    let mut next_context = trimmed_notes.to_vec();
                    next_context.extend(temp_notes.clone());

                    let lookahead_score = score_lookahead(
                        all_permutations_ref, i + 1, lookahead, &next_context, config, state, cache_ref,
                    );

                    IntermediateCandidate {
                        parent_idx,
                        added_notes: temp_notes,
                        score: beam_state.score + group_score + lookahead_score,
                    }
                })
            })
            .collect();

        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        beam = candidates.into_iter().take(beam_width).map(|c| {
            let mut new_notes = current_beam[c.parent_idx].notes.clone();
            new_notes.extend(c.added_notes);
            BeamCandidate {
                notes: new_notes,
                score: c.score,
            }
        }).collect();

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
