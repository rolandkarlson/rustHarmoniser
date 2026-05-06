use crate::model::{Note, Config};
use crate::utils::{SeededRng, ArrayExt, mod_shim, sin};
use crate::music_theory::{gen_scale};

use dashmap::DashMap;
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use fxhash::FxHasher64;

// --- Vectorized scoring infrastructure ---

/// Static consonance lookup table indexed by interval mod 12.
/// Replaces get_harmonic_score_adjusted() function calls.
/// 12x9 Harmony Matrix — each row is a style/context profile.
/// Rows: 0=Strict Classical, 1=Jazz/Extended, 2=Suspense/Tension, 3=Ethereal/Open,
///        4=Dark/Melancholic, 5=Bright/Lydian, 6=Aggressive/Brutal, 7=Ancient/Fifth-Based,
///        8=Neutral/Zero (no preference)
/// Columns: interval 0-11 in semitones.
const HARMONY_MATRIX: [[f64; 12]; 9] = [
    // 0: STRICT CLASSICAL (Pure consonance, heavy penalties for clashes)
    [1.0, -100.0, -0.5, 0.6, 0.8, 0.5, -100.0, 1.0, 0.5, 0.7, -0.8, -100.0],
    // 1: JAZZ & COLOR (7ths and 9ths are loved, clusters are okay)
    [1.0, -0.2, 0.5, 0.7, 0.9, 0.4, 0.3, 1.0, 0.4, 0.8, 0.9, 0.6],
    // 2: TENSION/RESOLUTION (High value on tritones and leading tones)
    [0.0, 0.8, 0.2, -0.5, -0.5, -0.2, 1.0, 0.1, -0.4, -0.4, 0.3, 0.9],
    // 3: ETHEREAL & OPEN (Perfect 4ths, 5ths, and Major 2nds/9ths)
    [1.0, -10.0, 0.7, -0.2, 0.3, 0.9, -1.0, 1.0, -0.1, 0.4, 0.2, -0.5],
    // 4: DARK & MELANCHOLIC (Bias toward minor 3rd and minor 6th)
    [1.0, -0.5, -0.2, 1.0, -0.4, 0.3, -0.2, 0.8, 0.9, -0.3, 0.4, -0.8],
    // 5: BRIGHT & LYDIAN (Major 3rd, Major 6th, #4 tritone)
    [1.0, -0.8, 0.4, -0.3, 1.0, -0.2, 0.7, 0.9, -0.2, 1.0, -0.4, 0.5],
    // 6: AGGRESSIVE/BRUTAL (Dissonance rewarded, unisons boring)
    [-0.5, 1.0, 0.4, -0.8, -0.8, -0.5, 0.9, -0.5, -0.8, -0.8, 0.5, 1.0],
    // 7: ANCIENT/HOLLOW (Organum: only 1sts, 4ths, 5ths, 8ths)
    [1.0, -100.0, -100.0, -100.0, -100.0, 0.8, -100.0, 1.0, -100.0, -100.0, -100.0, -100.0],
    // 8: NEUTRAL/ZERO (no preference — every interval scores 0.0)
    [0.0; 12],
];

/// Get the interpolated consonance table for a fractional harmony context value.
/// Integer values select a row directly; fractional values LERP between adjacent rows.
fn get_harmony_row(ctx: f64) -> [f64; 12] {
    let clamped = ctx.clamp(0.0, 8.0);
    let lo = clamped.floor() as usize;
    let hi = (lo + 1).min(8);
    let t = clamped - lo as f64;
    let mut row = [0.0f64; 12];
    for i in 0..12 {
        row[i] = HARMONY_MATRIX[lo][i] * (1.0 - t) + HARMONY_MATRIX[hi][i] * t;
    }
    row
}

type PitchSet = u128;
type IntervalSet = u16;

#[inline(always)]
fn pitch_set_contains(set: PitchSet, pitch: i32) -> bool {
    if pitch < 0 || pitch > 127 { return false; }
    (set >> pitch as u32) & 1 != 0
}

#[inline(always)]
fn pitch_set_from_slice(pitches: &[i32]) -> PitchSet {
    let mut set: PitchSet = 0;
    for &p in pitches {
        if p >= 0 && p <= 127 {
            set |= 1u128 << p as u32;
        }
    }
    set
}

#[inline(always)]
fn pitch_class_set_from_slice(pitches: &[i32]) -> IntervalSet {
    let mut set: IntervalSet = 0;
    for &p in pitches {
        let pc = ((p % 12) + 12) % 12;
        set |= 1u16 << pc as u16;
    }
    set
}

#[inline(always)]
fn interval_set_from_slice(intervals: &[i32]) -> IntervalSet {
    let mut set: IntervalSet = 0;
    for &iv in intervals {
        let iv_mod = ((iv % 12) + 12) % 12;
        set |= 1u16 << iv_mod as u16;
    }
    set
}
/// Heuristic channel-priority orderings. Each row is a voice-processing order.
/// The voice scored first has fewest constraints (most freedom), last has most.
const SMART_ORDERINGS: [[i32; 5]; 1] = [
    [0, 1, 2, 3, 4],  // Soprano-first
    // [4, 3, 2, 1, 0],  // Bass-first
    // [0, 4, 1, 3, 2],  // Outer-voices-first
    // [2, 1, 3, 0, 4],  // Inner-voices-first
    // [4, 0, 3, 1, 2],  // Alternating outer
    // [3, 2, 1, 0, 4],  // Tenor-first
    // [1, 0, 2, 4, 3],  // Alto-first
    // [0, 4, 2, 1, 3],  // Outer + middle
    // [4, 2, 0, 3, 1],  // Spread pattern
    // [2, 0, 4, 1, 3],  // Middle-out
];

/// Smart permutation selection: ~10 heuristic orderings instead of N!
/// For small groups (≤ 3 notes), falls back to all permutations.
pub fn get_permutations(notes: &[Note]) -> Vec<Vec<Note>> {
    let n = notes.len();
    // if n <= 3 {
    //     return get_all_permutations(notes);
    // }

    let mut results = Vec::with_capacity(SMART_ORDERINGS.len());
    let mut seen_channel_orders: Vec<Vec<i32>> = Vec::new();

    for ordering in &SMART_ORDERINGS {
        // Map channel ordering to notes present in this group
        let perm: Vec<Note> = ordering.iter()
            .filter_map(|&ch| notes.iter().find(|note| note.channel == ch))
            .copied()
            .collect();

        // Skip if not all notes were matched (group might not have all 5 channels)
        if perm.len() != n {
            continue;
        }

        // Deduplicate: skip if we already have this channel ordering
        let ch_order: Vec<i32> = perm.iter().map(|note| note.channel).collect();
        if seen_channel_orders.contains(&ch_order) {
            continue;
        }
        seen_channel_orders.push(ch_order);
        results.push(perm);
    }

    // Fallback: if heuristics produced fewer than 3 results (unusual channel layouts),
    // fall back to all permutations
    // if results.len() < 3 {
    //     return get_all_permutations(notes);
    // }

    results
}
// Helper to get permutations of notes
pub fn get_all_permutations(notes: &[Note]) -> Vec<Vec<Note>> {
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
        return 30.0;
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
    pub schillinger_notes: Vec<Vec<Vec<i32>>>,
    pub voice_contour: Option<Vec<Vec<i32>>>,
    pub contour_resolution: f64,
    pub harmony_contour: Option<Vec<f64>>,
    pub harmony_contour_resolution: f64,
    pub harmony_matrix_contour: Option<Vec<f64>>,
}


#[derive(Clone, Copy)]
enum BoundMode { Ceiling, Floor }

fn get_modular_bound(n: i32, anchors: &[i32], m: i32, mode: BoundMode) -> i32 {
    let mut sorted: Vec<i32> = anchors.iter().map(|&a| mod_shim(a, m)).collect();
    sorted.sort();
    sorted.dedup();
    if sorted.is_empty() { return n; }
    let r = mod_shim(n, m);
    match mode {
        BoundMode::Ceiling => sorted.iter().find(|&&x| x > r).copied().unwrap_or(sorted[0]),
        BoundMode::Floor => sorted.iter().rev().find(|&&x| x < r).copied().unwrap_or(*sorted.last().unwrap()),
    }
}

fn apply_bound(notes: Vec<i32>, anchors: &[i32], config: &Config, current_lasts_lead: Vec<i32>, i: i32) -> Vec<i32> {
    if !config.use_ceiling && !config.use_floor { return notes; }
    if current_lasts_lead.is_empty() || anchors.is_empty() { return notes; }
    let mode = if config.use_ceiling { BoundMode::Ceiling } else { BoundMode::Floor };
    current_lasts_lead.into_iter()
        .map(|n| get_modular_bound(n+i-1, anchors, 12, mode))
        .collect()
}

fn get_schillinger_scale(current_note: &Note, state: &HarmonizerState, config: &Config, current_lasts_lead: Vec<i32>) -> Vec<i32> {
    let bar_duration = 4.0;
    let bar = (current_note.start / bar_duration).floor() as i32;
    let num_voices = state.schillinger_notes.len() as i32;
    let voice_idx = if num_voices > 0 {
        mod_shim(current_note.channel, num_voices) as usize
    } else {
        0
    };
    let voice_bars = &state.schillinger_notes[voice_idx];
    let safe_bar = mod_shim(bar, voice_bars.len() as i32) as usize;
    let notes = &voice_bars[safe_bar];

    if(current_note.muted == 0){
        return notes.clone();
    }

    let result = if(bar % config.pl == 0 || bar % config.pl ==  config.pl - 1){
        if(current_note.channel == 4){
             vec![notes[0]]
        } else if(current_note.channel == 0){
            vec![notes[2]]
        } else {
            vec![notes[0], notes[1], notes[2]]
        }
    } else if(current_note.channel == 4){
        vec![notes[0]]
    } else {
        notes.clone()
    };

    apply_bound(result, notes, config, current_lasts_lead, current_note.channel )
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
    pub lead_pitch: Option<i32>,
    // Vectorized scoring fields
    pub last_harmony_interval_set: IntervalSet,
    pub has_parallel_fifth: bool,
    pub has_parallel_unison: bool,
    pub last_notes_bitset_by_channel: Vec<PitchSet>,
}

fn build_precomputed_data(context: &[Note], current_group: &[Note], start_time: f64) -> PrecomputedHarmonyData {
    let mut last_harmony = Vec::new();
    let mut sustaining_notes = Vec::new();
    let mut notes_ending_at_start = Vec::new();
    let mut sustaining_at_minus_0_1 = Vec::new();
    let mut last_notes_by_channel: Vec<Vec<i32>> = vec![Vec::new(); 16];
    let mut sustaining_lead_pitch: Option<i32> = None;
    let mut latest_past_lead: Option<(f64, i32)> = None;

    for n in context {
        // last_harmony: start <= start-1.0 && end > start-1.0
        if n.start <= start_time - 1.0 && n.start + n.duration > start_time - 1.0 && n.muted == 0 {
            last_harmony.push(n.pitch);
        }

        // sustaining_notes: start <= start && end > start
        if n.start <= start_time && n.start + n.duration > start_time && n.muted == 0 {
            sustaining_notes.push(n.pitch);
            if n.channel == 0 {
                sustaining_lead_pitch = Some(n.pitch);
            }
        }

        // most recent past channel-0 note (even if no longer sounding)
        if n.channel == 0 && n.muted == 0 && n.start < start_time {
            if latest_past_lead.map_or(true, |(s, _)| n.start > s) {
                latest_past_lead = Some((n.start, n.pitch));
            }
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

    let last_harmony_interval_set = interval_set_from_slice(&last_harmony_intervals);
    let has_parallel_fifth = last_harmony_intervals.contains(&7);
    let has_parallel_unison = last_harmony_intervals.contains(&0);
    let last_notes_bitset_by_channel: Vec<PitchSet> = last_notes_by_channel
        .iter()
        .map(|notes| pitch_set_from_slice(notes))
        .collect();

    // Lead pitch priority: current group > sustaining from context > most recent past lead
    let lead_pitch = current_group.iter()
        .find(|n| n.channel == 0 && n.muted == 0)
        .map(|n| n.pitch)
        .or(sustaining_lead_pitch)
        .or(latest_past_lead.map(|(_, p)| p));

    PrecomputedHarmonyData {
        last_harmony,
        last_harmony_intervals,
        sustaining_notes,
        boundries_by_channel,
        last_notes_by_channel,
        notes_ending_at_start,
        lead_pitch,
        last_harmony_interval_set,
        has_parallel_fifth,
        has_parallel_unison,
        last_notes_bitset_by_channel,
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

    // === Phase 1: Build context ===

    if current_note.muted == 0 {
        let candidate: i32 = if config.schillinger_progression {
            let sch_scale = get_schillinger_scale(current_note, state, config, Vec::new());
            let center_octave = (current_note.pitch as f64 / 12.0).floor() as i32;
            gen_scale(&sch_scale, center_octave)
                .into_iter()
                .min_by_key(|&p| (p - current_note.pitch).abs())
                .unwrap_or(current_note.pitch)
        } else {
            current_note.pitch
        };

        return vec![NoteScore {
            note: candidate,
            score: 0.0,
            distance: 0.0,
            crossing: false,
        }];
    }

    let channel_idx = current_note.channel as usize;

    // Merge sustaining + same-start into current harmony
    let mut harmony_pitches: Vec<i32> = Vec::with_capacity(precomputed.sustaining_notes.len() + current_on_same_start_harmony.len());
    harmony_pitches.extend_from_slice(&precomputed.sustaining_notes);
    for n in current_on_same_start_harmony {
        harmony_pitches.push(n.pitch);
    }
    let harmony_len = harmony_pitches.len();

    // Build bitsets for O(1) membership tests
    let harmony_pitch_set = pitch_set_from_slice(&harmony_pitches);
    let harmony_pitch_class_set = pitch_class_set_from_slice(&harmony_pitches);
    let harmony_interval_set = {
        let mut intervals = Vec::with_capacity(if harmony_len > 0 { harmony_len * (harmony_len - 1) / 2 } else { 0 });
        for i in 0..harmony_len {
            for j in (i+1)..harmony_len {
                intervals.push((harmony_pitches[i] - harmony_pitches[j]).abs() % 12);
            }
        }
        interval_set_from_slice(&intervals)
    };

    let current_on_same_end_harmony = &precomputed.notes_ending_at_start;

    let mut current_lasts = if channel_idx < precomputed.last_notes_by_channel.len() {
        precomputed.last_notes_by_channel[channel_idx].clone()
    } else {
        Vec::new()
    };

    let current_lasts_lead: Vec<i32> = if current_note.channel == 0 {
        Vec::new()
    } else {
        precomputed.lead_pitch.map(|p| vec![p]).unwrap_or_default()
    };

    let last_notes_set: PitchSet = if channel_idx < precomputed.last_notes_bitset_by_channel.len() {
        precomputed.last_notes_bitset_by_channel[channel_idx]
    } else { 0 };

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

    let mut target_offset: i32 = 0;
    let use_contour = if let Some(ref contours) = state.voice_contour {
        if !contours.is_empty() {
            let contour = &contours[mod_shim(channel_idx as i32, contours.len() as i32) as usize];
            if !contour.is_empty() {
                let idx = (current_note.start / state.contour_resolution).floor() as usize;
                target_offset = *contour.get_wrapped(idx);
            }
        }
        false // use_contour is always set to false in original
    } else {
        false
    };

    let seq = 0;

    let last_note = if !current_lasts.is_empty() { current_lasts[0] } else { current_note.pitch };
    let range = 3;
    let min_pitch = (last_note - range).max(24);
    let max_pitch = (last_note + range).min(96);

    // Generate candidates
    let sp = 0.0;
    let candidates: Vec<i32> = if config.schillinger_progression {
        let sch_scale = get_schillinger_scale(current_note, state, config, current_lasts_lead);
        let center_octave = (current_lasts[0] as f64 / 12.0).floor() as i32;
        gen_scale(&sch_scale, center_octave)
    } else {
        (min_pitch..=max_pitch).collect()
    };
    let n = candidates.len();

    // Precompute weights (loop-invariant)
    let r = if let Some(ref contour) = state.harmony_contour {
        if !contour.is_empty() {
            let idx = (current_note.start / state.harmony_contour_resolution).floor() as usize;
            *contour.get_wrapped(idx)
        } else {
            config.harmony_distance_balance
        }
    } else {
        config.harmony_distance_balance
    };
    let w_harmony = 0.5 + r;
    let w_smooth = 0.5 - r;

    let channel_boundry_max = [2,2,2,2,7].get_wrapped(channel_idx);
    let channel_boundry_min = [2,2,2,7,1].get_wrapped(channel_idx);

    let has_lasts = !current_lasts.is_empty();
    let input_pitch_f = current_note.pitch;

    // Precompute direction check: call only 2x instead of Nx
    let (dir_penalty_down, dir_penalty_up) = if is_outer_voice
        && !current_on_same_end_harmony.is_empty()
        && !current_on_same_start_harmony.is_empty()
        && has_lasts
    {
        (
            is_harmony_moving_to_same_direction(current_on_same_end_harmony, current_on_same_start_harmony, true),
            is_harmony_moving_to_same_direction(current_on_same_end_harmony, current_on_same_start_harmony, false),
        )
    } else {
        (false, false)
    };
    let check_direction = dir_penalty_down || dir_penalty_up;
    let last0 = if has_lasts { current_lasts[0] } else { 0 };

    // === Phase 2: Separate scoring passes ===

    let mut penalty_scores = vec![0.0f64; n];
    let mut consonance_scores = vec![0.0f64; n];
    let mut distance_scores = vec![0.0f64; n];
    let mut crossing_flags = vec![false; n];

    // Pass A: Consonance — harmony matrix lookup averaged over harmony notes
    // Select row from harmony matrix contour (default row 0 = Strict Classical)
    let harmony_ctx = if let Some(ref contour) = state.harmony_matrix_contour {
        if !contour.is_empty() {
            let idx = (current_note.start / state.harmony_contour_resolution).floor() as usize;
            *contour.get_wrapped(idx)
        } else {
            0.0
        }
    } else {
        0.0
    };
    let consonance_row = get_harmony_row(harmony_ctx);

    if harmony_len > 0 {
        let inv_len = 1.0 / harmony_len as f64;
        for &h in &harmony_pitches {
            for (i, &c) in candidates.iter().enumerate() {
                let interval = ((c - h).abs() % 12) as usize;
                consonance_scores[i] += consonance_row[interval];
            }
        }
        for s in consonance_scores.iter_mut() {
            *s *= inv_len;
        }
    }

    // Pass B: Duplicate & interval penalties (bitset O(1))
    for (i, &c) in candidates.iter().enumerate() {
        // Exact duplicate in harmony
        if pitch_set_contains(harmony_pitch_set, c) {
            penalty_scores[i] -= 10000.0;
        }
        // Octave-equivalent duplicate (only when harmony < 3 notes)
        if harmony_len < 3 {
            let pc = ((c % 12) + 12) % 12;
            if (harmony_pitch_class_set >> pc as u16) & 1 != 0 {
                penalty_scores[i] -= 10000.0;
            }
        }
        // Interval already in harmony (only when harmony < 3)
        if harmony_len < 3 {
            for &h in &harmony_pitches {
                let dif = ((h - c).abs() % 12) as u16;
                if (harmony_interval_set >> dif) & 1 != 0 {
                    penalty_scores[i] -= config.interval_exists_in_harmony;
                }
            }
        }
    }

    // Pass C: Parallel 5ths/unisons (precomputed booleans gate the loop)
    if !precomputed.last_harmony.is_empty() {
        if precomputed.has_parallel_fifth {
            for (i, &c) in candidates.iter().enumerate() {
                for &h in &harmony_pitches {
                    if (c - h).abs() % 12 == 7 {
                        penalty_scores[i] -= config.consecutive_octav_fift;
                    }
                }
            }
        }
        if precomputed.has_parallel_unison {
            for (i, &c) in candidates.iter().enumerate() {
                for &h in &harmony_pitches {
                    if (c - h).abs() % 12 == 0 {
                        penalty_scores[i] -= config.consecutive_octav_fift;
                    }
                }
            }
        }
    }

    // Pass D: Voice crossing
    for (i, &c) in candidates.iter().enumerate() {
        if bounds_p.max - c < *channel_boundry_max {
            penalty_scores[i] -= config.no_crossing;
            crossing_flags[i] = true;
        }
        if c - bounds_p.min < *channel_boundry_min {
            penalty_scores[i] -= config.no_crossing;
            crossing_flags[i] = true;
        }
    }

    // Pass E-G: Distance, cubic, and history penalties
    if has_lasts {
        for (i, &c) in candidates.iter().enumerate() {
            // E: Distance score
            distance_scores[i] = get_distance_score(last_note, c);

            // F: Cubic pitch distance
            let base_dist = if use_contour {
                (c - input_pitch_f + target_offset).abs()
            } else {
                (c - current_note.pitch + seq).abs()
            };
            let normalized = base_dist as f64 / 8.0;
            penalty_scores[i] -= normalized * normalized * normalized;

            // G: History penalties (bitset gates the expensive count)
            if !no_same_note_penalty {
                if c == last_note {
                    penalty_scores[i] -= config.last_note_same;
                }
                if pitch_set_contains(last_notes_set, c) {
                    let count = current_lasts.iter().filter(|&&x| x == c).count();
                    if count >= 2 {
                        penalty_scores[i] -= config.last_note_exist_in_voice;
                    }
                }
            }
        }
    }

    // Pass H: Same direction penalty (precomputed for both directions)
    if check_direction {
        for (i, &c) in candidates.iter().enumerate() {
            let going_down = last0 > c;
            if (going_down && dir_penalty_down) || (!going_down && dir_penalty_up) {
                penalty_scores[i] -= config.same_direction;
            }
        }
    }

    // === Phase 3: Combine — element-wise vectorized ===

    let mut scores = Vec::with_capacity(n);
    for i in 0..n {
        let sum_score = (consonance_scores[i] - sp) * w_harmony
            + distance_scores[i] * w_smooth
            + penalty_scores[i];
        scores.push(NoteScore {
            note: candidates[i],
            score: sum_score,
            distance: 0.0,
            crossing: crossing_flags[i],
        });
    }

    scores
}


pub fn gen_voice_from_notes(pattern: &[Note], source_length: f64, config: &Config) -> Vec<Note> {
    let clip_len = (config.pl * 4 * config.render_length) as f64;
    if pattern.is_empty() || source_length <= 0.0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut offset = 0.0;
    while offset < clip_len {
        for n in pattern {
            let s = n.start + offset;
            if s >= clip_len { continue; }
            let mut d = n.duration;
            if s + d > clip_len { d = clip_len - s; }
            if d < 0.001 { continue; }
            out.push(Note {
                pitch: n.pitch,
                start: s,
                duration: d,
                velocity: n.velocity,
                muted: 0,
                channel: 0,
                probability: n.probability,
            });
        }
        offset += source_length;
    }
    out
}

pub fn gen_voice(base: i32, rhythm_data: &Vec<f64>, pitch_shifts: &[i32], channel: i32, muted: i32, config: &Config) -> Vec<Note> {
    let mut ar = Vec::new();
    let clip_len = (config.pl * 4 * config.render_length) as f64;
    let mut pos = 0.0;
    let mut counter = 0;
    let sf = (SeededRng::random_int(60) + 1) as f64;

    while pos < clip_len {
        let n = base + pitch_shifts[mod_shim(counter, pitch_shifts.len() as i32) as usize];
        let mut d = if let Some(vrc) = &config.voice_rhythm_contour {
            if channel >= 0 && channel < vrc.len() as i32 {
                let track = &vrc[channel as usize];
                if !track.is_empty() {
                    let idx = (pos / config.voice_contour_resolution).floor() as usize;
                    track[mod_shim(idx as i32, track.len() as i32) as usize]
                } else {
                    rhythm_data[mod_shim(counter, rhythm_data.len() as i32) as usize]
                }
            } else {
                rhythm_data[mod_shim(counter, rhythm_data.len() as i32) as usize]
            }
        } else {
            rhythm_data[mod_shim(counter, rhythm_data.len() as i32) as usize]
        };

        // Clamp to bar boundary (bar = 4 beats) — notes must not cross bar lines
        let bar_len = 4.0;
        let next_bar = ((pos / bar_len).floor() + 1.0) * bar_len;
        if pos + d > next_bar {
            d = next_bar - pos;
        }

        if pos + d > clip_len {
            d = clip_len - pos;
        }

        if d < 0.001 {
            pos = next_bar.min(clip_len);
            continue;
        }

        let v = 1 + SeededRng::random_int(10) + sin(counter as f64, sf, 10.0) as i32;
        ar.push(Note::new(n, pos, d, v, muted, channel));
        pos += d;

        // Snap to bar boundary to avoid float drift
        if (pos - next_bar).abs() < 0.001 {
            pos = next_bar;
        }

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
    let precomputed = build_precomputed_data(context, &permutations[0], start_time);

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

use std::sync::mpsc::Sender;

fn score_group_beam(income: Vec<Note>, config: &Config, state: &HarmonizerState, progress_sender: Option<&Sender<(usize, usize)>>) -> Vec<Note> {
    let grouped_notes = group_by_start_array(income);

    let all_permutations: Vec<Vec<Vec<Note>>> = grouped_notes.par_iter()
        .map(|g| get_permutations(g))
        .collect();

    let beam_width = 5;
    let lookahead = config.lookahead_depth;

    let mut beam = vec![BeamCandidate {
        notes: Vec::new(),
        score: 0.0,
    }];
    let mut ccc = 0.0;

    for (i, _) in grouped_notes.iter().enumerate() {
        if let Some(sender) = progress_sender {
            let _ = sender.send((i, grouped_notes.len()));
        }

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
                let trimmed_notes = &beam_state.notes;

                let start_time = permutations[0][0].start;
                let precomputed = build_precomputed_data(trimmed_notes, &permutations[0], start_time);

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

        //  println!("Processed group {}/{}, best score: {}", i, grouped_notes.len(), beam[0].score - ccc);
        ccc = beam[0].score;
    }

    if beam.is_empty() {
        return Vec::new();
    }
    //   println!("Final Score: {}", beam[0].score);
    beam[0].notes.clone()
}

pub fn harmonise2(income: Vec<Note>, config: &Config, state: &HarmonizerState, progress_sender: Option<&Sender<(usize, usize)>>) -> Vec<Note> {
    score_group_beam(income, config, state, progress_sender)
}
