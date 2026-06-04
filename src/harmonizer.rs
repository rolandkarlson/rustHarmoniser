use crate::model::{Note, Config};
use crate::utils::{SeededRng, ArrayExt, mod_shim, sin};
use crate::music_theory::{gen_scale};

use dashmap::DashMap;
use rayon::prelude::*;
use serde::Serialize;
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
// Soft preferences live in [-1, 1] (higher = more favoured); any cell <= -5 is a
// HARD "forbidden" constraint (see FORBIDDEN_THRESHOLD). Physical roughness/spacing
// (m2 vs m9, register) is handled separately by pair_roughness, so these rows
// encode interval-class *character* per style, not raw acoustic dissonance.
// Columns: P1 m2 M2 m3 M3 P4 TT P5 m6 M6 m7 M7
pub const HARMONY_MATRIX: [[f64; 12]; 9] = [
    // 0: STRICT CLASSICAL — consonance-first; m2 / tritone / M7 forbidden
    [1.0, -100.0, -0.4, 0.8, 0.9, 0.5, -100.0, 1.0, 0.7, 0.8, -0.3, -100.0],
    // 1: JAZZ & COLOR — 7ths/9ths/tritone embraced; bare unison/octave duller
    [0.6, 0.0, 0.7, 0.8, 0.9, 0.5, 0.6, 0.9, 0.5, 0.8, 1.0, 0.8],
    // 2: TENSION/RESOLUTION — tritones and leading tones prized, triads dull
    [-0.2, 0.8, 0.2, -0.3, -0.3, -0.2, 1.0, 0.0, -0.3, -0.3, 0.5, 0.9],
    // 3: ETHEREAL & OPEN — quartal/quintal and open 2nds/9ths; m2 forbidden
    [1.0, -100.0, 0.8, -0.2, 0.2, 1.0, -0.5, 1.0, 0.0, 0.5, 0.4, -0.4],
    // 4: DARK & MELANCHOLIC — minor 3rd/6th favoured, major color avoided
    [1.0, -0.5, -0.1, 1.0, -0.4, 0.3, -0.2, 0.8, 1.0, -0.3, 0.5, -0.6],
    // 5: BRIGHT & LYDIAN — major 3rd/6th and the #4 tritone
    [1.0, -0.7, 0.5, -0.3, 1.0, -0.2, 0.8, 0.9, -0.2, 1.0, -0.3, 0.6],
    // 6: AGGRESSIVE/BRUTAL — dissonance rewarded, consonance dull
    [-0.5, 1.0, 0.4, -0.6, -0.6, -0.4, 1.0, -0.5, -0.6, -0.6, 0.5, 1.0],
    // 7: ANCIENT/HOLLOW — organum: only 1sts, 4ths, 5ths, 8ths
    [1.0, -100.0, -100.0, -100.0, -100.0, 1.0, -100.0, 1.0, -100.0, -100.0, -100.0, -100.0],
    // 8: NEUTRAL/ZERO — no preference; every interval scores 0.0
    [0.0; 12],
];

// ===================== Harmony scoring tuning =====================
// Matrix cells at or below this are treated as a HARD constraint ("forbidden")
// rather than a numeric preference. Replaces the old -100 / -10 sentinels, which
// (a) were diluted by averaging and (b) produced meaningless values when LERPed
// between style rows. Soft preferences now live in a clean [-1, 1] band.
const FORBIDDEN_THRESHOLD: f64 = -5.0;
// Score for candidates that hit any forbidden interval — large enough to lose to
// every other term even at the lowest harmony weight, so it acts as a reject.
const HARD_REJECT: f64 = -1.0e6;
// How much register-aware sensory roughness (Layer 3) modulates the pitch-class
// style preference. 0 = pure style/pitch-class; 1 = pure psychoacoustic roughness.
const ROUGHNESS_WEIGHT: f64 = 0.35;
// Chord-level aggregation weights (Layer 2): overall mean, worst single clash,
// and the interval against the bass (root/inversion sensitivity). Sum ≈ 1.
const AGG_MEAN: f64 = 0.40;
const AGG_WORST: f64 = 0.35;
const AGG_BASS: f64 = 0.25;

/// A resolved consonance profile for one fractional harmony-context value.
/// `soft` holds bounded style preferences per interval class (0..11); `forbidden`
/// marks hard-rejected interval classes (kept out of the soft LERP).
struct HarmonyRow {
    soft: [f64; 12],
    forbidden: [bool; 12],
}

/// Resolve the consonance profile for a fractional harmony context value.
/// Integer values select a row directly; fractional values LERP between adjacent
/// rows — but only over the *clamped* soft values, so sentinel "forbidden" cells
/// no longer poison the interpolation. Forbiddenness is decided by the dominant
/// adjacent row. `custom` overrides the built-in HARMONY_MATRIX when supplied and
/// well-formed (9 rows × 12 columns); otherwise the default matrix is used.
fn get_harmony_row(ctx: f64, custom: Option<&Vec<Vec<f64>>>) -> HarmonyRow {
    let clamped = ctx.clamp(0.0, 8.0);
    let lo = clamped.floor() as usize;
    let hi = (lo + 1).min(8);
    let t = clamped - lo as f64;
    let valid = custom.map_or(false, |m| m.len() == 9 && m.iter().all(|r| r.len() == 12));
    let cell = |row: usize, i: usize| -> f64 {
        if valid { custom.unwrap()[row][i] } else { HARMONY_MATRIX[row][i] }
    };
    let mut soft = [0.0f64; 12];
    let mut forbidden = [false; 12];
    for i in 0..12 {
        let lo_v = cell(lo, i);
        let hi_v = cell(hi, i);
        let lo_forb = lo_v <= FORBIDDEN_THRESHOLD;
        let hi_forb = hi_v <= FORBIDDEN_THRESHOLD;
        forbidden[i] = (lo_forb && (1.0 - t) >= 0.5) || (hi_forb && t >= 0.5);
        // Clamp to [-1, 1] before blending so a forbidden cell contributes at most
        // -1 to the soft surface instead of -100.
        let lo_s = lo_v.clamp(-1.0, 1.0);
        let hi_s = hi_v.clamp(-1.0, 1.0);
        soft[i] = lo_s * (1.0 - t) + hi_s * t;
    }
    HarmonyRow { soft, forbidden }
}

#[inline]
fn midi_to_hz(p: i32) -> f64 {
    440.0 * 2f64.powf((p as f64 - 69.0) / 12.0)
}

/// Plomp–Levelt / Sethares sensory dissonance between two fundamentals,
/// normalized to roughly [0, 1]. Roughness peaks ~1 semitone apart and decays to
/// ~0 at unison and at wide spacing — and because it works on absolute frequency,
/// it is register-aware: a literal minor 2nd is far rougher than a minor 9th, and
/// the same pitch-class clash is rougher low than high. (Fundamental-only model.)
fn pair_roughness(p1: i32, p2: i32) -> f64 {
    if p1 == p2 { return 0.0; }
    let f_low = midi_to_hz(p1.min(p2));
    let f_high = midi_to_hz(p1.max(p2));
    let s = 0.24 / (0.0207 * f_low + 18.96);
    let fdiff = f_high - f_low;
    let r = (-3.5 * s * fdiff).exp() - (-5.75 * s * fdiff).exp();
    // The curve e^{-3.5x} - e^{-5.75x} peaks at ≈0.1813; normalize to ~[0, 1].
    (r / 0.1813).clamp(0.0, 1.0)
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
const SMART_ORDERINGS: [[i32; 5]; 10] = [
    [0, 1, 2, 3, 4],  // Soprano-first
    [4, 3, 2, 1, 0],  // Bass-first
    [0, 4, 1, 3, 2],  // Outer-voices-first
    [2, 1, 3, 0, 4],  // Inner-voices-first
    [4, 0, 3, 1, 2],  // Alternating outer
    [3, 2, 1, 0, 4],  // Tenor-first
    [1, 0, 2, 4, 3],  // Alto-first
    [0, 4, 2, 1, 3],  // Outer + middle
    [4, 2, 0, 3, 1],  // Spread pattern
    [2, 0, 4, 1, 3],  // Middle-out
];

/// Smart permutation selection: ~10 heuristic orderings instead of N!
/// For small groups (≤ 3 notes), falls back to all permutations.
pub fn get_permutations(notes: &[Note], use_leading_voice: bool) -> Vec<Vec<Note>> {
    let n = notes.len();

    if use_leading_voice {
        let mut perm = notes.to_vec();
        perm.sort_by_key(|n| n.channel);
        return vec![perm];
    }

    if n <= 3 {
        return get_all_permutations(notes);
    }

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
    if results.len() < 3 {
        return get_all_permutations(notes);
    }

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

/// Voice-leading smoothness, normalized to roughly [-1, 1] so it trades off on an
/// equal footing with the (also bounded) consonance term via w_smooth / w_harmony.
/// Previously this returned +30 for a unison and -(10·dist) for leaps — a dynamic
/// range that swamped the harmony matrix entirely.
///
/// Holding a pitch (a common tone, dist 0) is the SMOOTHEST option and scores
/// highest; small steps are next; leaps decay toward -1. Excessive holding /
/// repetition is curbed separately by the repeat penalties (last_note_*) and the
/// group-level common-tone control — NOT here — so that common tones can occur.
/// 0 -> 1.0, 1 -> 0.86, 2 -> 0.71, 7 -> 0.0, 12 (octave) -> ~-0.71, then clamps.
pub fn get_distance_score(prev_note: i32, current_note: i32) -> f64 {
    let dist = (prev_note - current_note).abs() as f64;
    (1.0 - dist / 7.0).clamp(-1.0, 1.0)
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ScoreBreakdown {
    pub harmony: f64,
    pub distance: f64,
    pub repeat: f64,
    pub crossing: f64,
    pub parallel: f64,
    pub contour: f64,
    pub same_direction: f64,
    pub total: f64,
}

#[derive(Clone)]
pub struct NoteScore {
    pub note: i32,
    pub score: f64,
    pub distance: f64,
    pub crossing: bool,
    pub breakdown: ScoreBreakdown,
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
    pub harmony_matrix: Option<Vec<Vec<f64>>>,
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

    if current_note.muted == 0 {
        return notes.clone();
    }

    let result = if config.use_resolve {
        if bar % config.pl == 0 || bar % config.pl == config.pl - 1 {
            if current_note.channel == 4 {
                vec![notes[0]]
            } else if current_note.channel == 0 {
                vec![notes[2]]
            } else {
                vec![notes[0], notes[1], notes[2]]
            }
        } else if current_note.channel == 4 {
            vec![notes[0]]
        } else {
            notes.clone()
        }
    } else {
        notes.clone()
    };

    apply_bound(result, notes, config, current_lasts_lead, current_note.channel)
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
    pub sustaining_notes: Vec<i32>,
    pub sustaining_note_events: Vec<Note>,
    pub boundries_by_channel: Vec<Boundries>,
    pub last_notes_by_channel: Vec<Vec<i32>>,
    pub notes_ending_at_start: Vec<Note>,
    pub lead_pitch: Option<i32>,
    pub last_notes_bitset_by_channel: Vec<PitchSet>,
}

fn build_precomputed_data(context: &[Note], current_group: &[Note], start_time: f64) -> PrecomputedHarmonyData {
    let mut sustaining_notes = Vec::new();
    let mut sustaining_note_events = Vec::new();
    let mut notes_ending_at_start = Vec::new();
    let mut sustaining_at_minus_0_1 = Vec::new();
    let mut last_notes_by_channel: Vec<Vec<i32>> = vec![Vec::new(); 16];
    let mut sustaining_lead_pitch: Option<i32> = None;
    let mut latest_past_lead: Option<(f64, i32)> = None;

    for n in context {
        // sustaining_notes: start <= start && end > start
        if n.start <= start_time && n.start + n.duration > start_time && n.muted == 0 {
            sustaining_notes.push(n.pitch);
            sustaining_note_events.push(*n);
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
        sustaining_notes,
        sustaining_note_events,
        boundries_by_channel,
        last_notes_by_channel,
        notes_ending_at_start,
        lead_pitch,
        last_notes_bitset_by_channel,
    }
}

fn motion_direction(from: i32, to: i32) -> i32 {
    (to - from).signum()
}

fn parallel_motion_contribution(
    candidate_pitch: i32,
    candidate_channel: i32,
    candidate_previous_pitch: i32,
    harmony_events: &[Note],
    last_notes_by_channel: &[Vec<i32>],
    penalty_weight: f64,
) -> f64 {
    if penalty_weight == 0.0 {
        return 0.0;
    }

    let candidate_motion = motion_direction(candidate_previous_pitch, candidate_pitch);
    if candidate_motion == 0 {
        return 0.0;
    }

    let mut contribution = 0.0;
    for other in harmony_events {
        if other.muted != 0 || other.channel == candidate_channel {
            continue;
        }

        let other_channel = other.channel as usize;
        if other_channel >= last_notes_by_channel.len() {
            continue;
        }

        let Some(&other_previous_pitch) = last_notes_by_channel[other_channel].first() else {
            continue;
        };

        let other_motion = motion_direction(other_previous_pitch, other.pitch);
        if other_motion == 0 || other_motion != candidate_motion {
            continue;
        }

        let previous_interval = (candidate_previous_pitch - other_previous_pitch).abs() % 12;
        let current_interval = (candidate_pitch - other.pitch).abs() % 12;
        if previous_interval == current_interval && (current_interval == 0 || current_interval == 7) {
            contribution -= penalty_weight;
        }
    }

    contribution
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

    if current_note.channel == 0 && config.use_leading_voice {
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
            breakdown: ScoreBreakdown::default(),
        }];
    }

    let channel_idx = current_note.channel as usize;

    // Merge sustaining + same-start into current harmony
    let mut harmony_pitches: Vec<i32> = Vec::with_capacity(precomputed.sustaining_notes.len() + current_on_same_start_harmony.len());
    harmony_pitches.extend_from_slice(&precomputed.sustaining_notes);
    let mut harmony_events: Vec<Note> = Vec::with_capacity(precomputed.sustaining_note_events.len() + current_on_same_start_harmony.len());
    harmony_events.extend(precomputed.sustaining_note_events.iter().copied());
    for n in current_on_same_start_harmony {
        harmony_pitches.push(n.pitch);
        harmony_events.push(*n);
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
        let from_ending: Vec<i32> = current_on_same_start_harmony.iter()
            .filter(|n| n.channel == 0)
            .map(|n| n.pitch)
            .collect();
        if from_ending.is_empty() {
            precomputed.lead_pitch.map(|p| vec![p]).unwrap_or_default()
        } else {
            from_ending
        }
    };

    let last_notes_set: PitchSet = if channel_idx < precomputed.last_notes_bitset_by_channel.len() {
        precomputed.last_notes_bitset_by_channel[channel_idx]
    } else { 0 };

    let bounds_p = if channel_idx < precomputed.boundries_by_channel.len() {
        &precomputed.boundries_by_channel[channel_idx]
    } else {
        &Boundries { min: 24, max: 90 }
    };

    // Outer voices = soprano (ch0) and bass (ch4) — the pair contrary motion matters for.
    let is_outer_voice = current_note.channel == 0 || current_note.channel == 4;

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
        true // use_contour is always set to false in original
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

    let mut consonance_scores = vec![0.0f64; n];
    let mut distance_scores = vec![0.0f64; n];
    let mut harmony_contribs = vec![0.0f64; n];
    let mut distance_contribs = vec![0.0f64; n];
    let mut repeat_contribs = vec![0.0f64; n];
    let mut crossing_contribs = vec![0.0f64; n];
    let mut parallel_contribs = vec![0.0f64; n];
    let mut contour_contribs = vec![0.0f64; n];
    let mut same_direction_contribs = vec![0.0f64; n];
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
    let consonance_row = get_harmony_row(harmony_ctx, state.harmony_matrix.as_ref());

    if harmony_len > 0 {
        // Bass = lowest sounding pitch; its interval is weighted extra so chord
        // root/inversion identity is respected rather than averaged away.
        let bass_pitch = *harmony_pitches.iter().min().unwrap();
        for (i, &c) in candidates.iter().enumerate() {
            let mut sum = 0.0;
            let mut worst = f64::INFINITY;
            let mut bass_pair = 0.0;
            let mut hard = false;
            for &h in &harmony_pitches {
                let ic = ((c - h).abs() % 12) as usize;
                if consonance_row.forbidden[ic] {
                    hard = true;
                }
                // Layer 3: blend pitch-class style preference with register-aware
                // roughness, so m2 ≠ m9 and the same chord clusters worse down low.
                let style = consonance_row.soft[ic];
                let rough = pair_roughness(c, h);
                let pair = (1.0 - ROUGHNESS_WEIGHT) * style - ROUGHNESS_WEIGHT * rough;
                sum += pair;
                if pair < worst { worst = pair; }
                if h == bass_pitch { bass_pair = pair; }
            }
            consonance_scores[i] = if hard {
                HARD_REJECT
            } else {
                // Layer 2: chord-level aggregation — overall fit, worst single
                // clash (so one harsh interval can't hide behind consonant ones),
                // and the bass interval.
                let mean = sum / harmony_len as f64;
                AGG_MEAN * mean + AGG_WORST * worst + AGG_BASS * bass_pair
            };
        }
    }
    for i in 0..n {
        harmony_contribs[i] += (consonance_scores[i] - sp) * w_harmony;
    }

    // Pass B: Duplicate & interval penalties (bitset O(1))
    for (i, &c) in candidates.iter().enumerate() {
        // Exact duplicate in harmony
        if pitch_set_contains(harmony_pitch_set, c) {
            harmony_contribs[i] -= 10000.0;
        }
        // Octave-equivalent duplicate (only when harmony < 3 notes)
        if harmony_len < 3 {
            let pc = ((c % 12) + 12) % 12;
            if (harmony_pitch_class_set >> pc as u16) & 1 != 0 {
                harmony_contribs[i] -= 10000.0;
            }
        }
        // Interval already in harmony (only when harmony < 3)
        if harmony_len < 3 {
            for &h in &harmony_pitches {
                let dif = ((h - c).abs() % 12) as u16;
                if (harmony_interval_set >> dif) & 1 != 0 {
                    harmony_contribs[i] -= config.interval_exists_in_harmony;
                }
            }
        }
    }

    // Pass C: Parallel 5ths/octaves by channel-pair motion
    if config.consecutive_octav_fift != 0.0 && !harmony_events.is_empty() {
        for (i, &c) in candidates.iter().enumerate() {
            parallel_contribs[i] += parallel_motion_contribution(
                c,
                current_note.channel,
                last_note,
                &harmony_events,
                &precomputed.last_notes_by_channel,
                config.consecutive_octav_fift,
            );
        }
    }

    // Pass D: Voice crossing
    for (i, &c) in candidates.iter().enumerate() {
        if bounds_p.max - c < *channel_boundry_max {
            crossing_contribs[i] -= config.no_crossing;
            crossing_flags[i] = true;
        }
        if c - bounds_p.min < *channel_boundry_min {
            crossing_contribs[i] -= config.no_crossing;
            crossing_flags[i] = true;
        }
    }

    // Pass E-G: Distance, cubic, and history penalties
    if has_lasts {
        for (i, &c) in candidates.iter().enumerate() {
            // E: Distance score
            distance_scores[i] = get_distance_score(last_note, c);
            distance_contribs[i] = distance_scores[i] * w_smooth;

            // F: Quartic pull toward the voice's pitch-contour target.
            // Scaled by config.voice_contour_weight (0 = contour has no effect).
            if config.voice_contour_weight != 0.0 {
                let base_dist = if use_contour {
                    (c - (input_pitch_f + target_offset)).abs()
                } else {
                    (c - current_note.pitch + seq).abs()
                };
                let normalized = base_dist as f64 / 24.0;
                contour_contribs[i] -= config.voice_contour_weight * normalized * normalized * normalized * normalized;
            }

            // G: History penalties (leader) / hold bonus (non-leader).
            // `no_same_note_penalty` is true for every voice except the
            // permutation leader. Non-leaders get a "stickiness" bonus for keeping
            // their previous pitch (common tone) so they hold unless moving is
            // clearly better; the leader is penalized for holding so it moves.
            if no_same_note_penalty {
                if c == last_note {
                    repeat_contribs[i] += config.same_note_bonus;
                }
            } else {
                if c == last_note {
                    repeat_contribs[i] -= config.last_note_same;
                }
                if pitch_set_contains(last_notes_set, c) {
                    let count = current_lasts.iter().filter(|&&x| x == c).count();
                    if count >= 2 {
                        repeat_contribs[i] -= config.last_note_exist_in_voice;
                    }
                }
            }
        }
    }

    // Pass H: Same direction penalty (precomputed for both directions).
    // A held note (c == last0) is oblique motion, not similar motion — exempt it.
    if check_direction {
        for (i, &c) in candidates.iter().enumerate() {
            if c == last0 { continue; }
            let going_down = last0 > c;
            if (going_down && dir_penalty_down) || (!going_down && dir_penalty_up) {
                same_direction_contribs[i] -= config.same_direction;
            }
        }
    }

    // === Phase 3: Combine — element-wise vectorized ===

    let mut scores = Vec::with_capacity(n);
    for i in 0..n {
        let sum_score = harmony_contribs[i]
            + distance_contribs[i]
            + repeat_contribs[i]
            + crossing_contribs[i]
            + parallel_contribs[i]
            + contour_contribs[i]
            + same_direction_contribs[i];
        let breakdown = ScoreBreakdown {
            harmony: harmony_contribs[i],
            distance: distance_contribs[i],
            repeat: repeat_contribs[i],
            crossing: crossing_contribs[i],
            parallel: parallel_contribs[i],
            contour: contour_contribs[i],
            same_direction: same_direction_contribs[i],
            total: sum_score,
        };
        scores.push(NoteScore {
            note: candidates[i],
            score: sum_score,
            distance: distance_scores[i],
            crossing: crossing_flags[i],
            breakdown,
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
    actual_score: f64,
}

struct IntermediateCandidate {
    parent_idx: usize,
    added_notes: Vec<Note>,
    actual_score: f64,
    rank_score: f64,
}

fn score_note_group(
    current_notes_in: &[Note],
    temp_group_notes: &mut Vec<Note>,
    no_same_note_penalty: bool,
    config: &Config,
    state: &HarmonizerState,
    precomputed: &PrecomputedHarmonyData
) -> f64 {
    let current_notes = current_notes_in.to_vec();
    let n = current_notes.len();
    let permu_first_channel = current_notes.last().unwrap().channel;
    let base_len = temp_group_notes.len();

    let prev_pitch = |ch: i32| -> Option<i32> {
        precomputed.last_notes_by_channel.get(ch as usize).and_then(|v| v.first()).copied()
    };
    let skip_for = |ch: i32| no_same_note_penalty || (permu_first_channel != ch);

    // ---- Pass 1: greedy best pitch per voice; record the "hold previous pitch"
    // alternative so we know each voice's benefit of moving. ----
    let mut chosen = vec![0i32; n];
    let mut best_score = vec![0.0f64; n];
    let mut hold_pitch = vec![None::<i32>; n];   // previous pitch, if this voice can hold it
    let mut hold_score = vec![f64::NEG_INFINITY; n];
    let mut group_score = 0.0;

    for j in 0..n {
        let mut scores = get_harmony_scores(&current_notes[j], temp_group_notes, skip_for(current_notes[j].channel), config, state, precomputed);
        scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // The leading voice's pitch is fixed by the input melody — not eligible.
        let is_lead = current_notes[j].channel == 0 && config.use_leading_voice;
        if !is_lead {
            if let Some(hp) = prev_pitch(current_notes[j].channel) {
                if let Some(h) = scores.iter().find(|s| s.note == hp) {
                    hold_pitch[j] = Some(hp);
                    hold_score[j] = h.score;
                }
            }
        }

        if let Some(best) = scores.first() {
            chosen[j] = best.note;
            best_score[j] = best.score;
            let mut note = current_notes[j];
            note.pitch = best.note;
            note.muted = 0;
            temp_group_notes.push(note);
            group_score += best.score;
        } else {
            chosen[j] = current_notes[j].pitch;
            temp_group_notes.push(current_notes[j]);
        }
    }

    // ---- Voice-change budget: cap/floor how many eligible voices change pitch
    // between consecutive chords, by forcing the least-worthwhile movers to hold
    // (or the most-worthwhile holders to move) and re-scoring. ----
    let max_changed = config.max_voices_changed;
    let min_changed = config.min_voices_changed;
    if max_changed >= 0 || min_changed >= 0 {
        let movers: Vec<usize> = (0..n)
            .filter(|&j| hold_pitch[j].is_some() && chosen[j] != hold_pitch[j].unwrap())
            .collect();
        let benefit = |j: usize| best_score[j] - hold_score[j];

        let mut force_hold: Vec<usize> = Vec::new();
        let mut force_move: Vec<usize> = Vec::new();

        if max_changed >= 0 && movers.len() as i32 > max_changed {
            // Keep the highest-benefit movers; hold the rest on their old pitch.
            let mut m = movers.clone();
            m.sort_by(|&a, &b| benefit(a).partial_cmp(&benefit(b)).unwrap_or(std::cmp::Ordering::Equal));
            let to_hold = m.len() - max_changed as usize;
            force_hold = m.into_iter().take(to_hold).collect();
        } else if min_changed >= 0 && (movers.len() as i32) < min_changed {
            // Force the highest-benefit holders to move.
            let mut h: Vec<usize> = (0..n)
                .filter(|&j| hold_pitch[j].is_some() && chosen[j] == hold_pitch[j].unwrap())
                .collect();
            h.sort_by(|&a, &b| benefit(b).partial_cmp(&benefit(a)).unwrap_or(std::cmp::Ordering::Equal));
            let to_move = ((min_changed as usize).saturating_sub(movers.len())).min(h.len());
            force_move = h.into_iter().take(to_move).collect();
        }

        if !force_hold.is_empty() || !force_move.is_empty() {
            // Pass 2: rebuild the chord. Forced holds are placed first so the
            // remaining voices are scored against them.
            temp_group_notes.truncate(base_len);
            group_score = 0.0;
            let mut order: Vec<usize> = force_hold.clone();
            order.extend((0..n).filter(|j| !force_hold.contains(j)));

            for &j in &order {
                let skip = skip_for(current_notes[j].channel);
                let mut scores = get_harmony_scores(&current_notes[j], temp_group_notes, skip, config, state, precomputed);
                scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

                let pick = if force_hold.contains(&j) {
                    let hp = hold_pitch[j].unwrap();
                    scores.iter().find(|s| s.note == hp).cloned()
                        .unwrap_or(NoteScore { note: hp, score: 0.0, distance: 0.0, crossing: false, breakdown: ScoreBreakdown::default() })
                } else if force_move.contains(&j) {
                    // Must not pick the held pitch.
                    let hp = hold_pitch[j];
                    scores.iter().find(|s| Some(s.note) != hp).or_else(|| scores.first()).cloned()
                        .unwrap_or(NoteScore { note: current_notes[j].pitch, score: 0.0, distance: 0.0, crossing: false, breakdown: ScoreBreakdown::default() })
                } else {
                    match scores.first() {
                        Some(b) => b.clone(),
                        None => NoteScore { note: current_notes[j].pitch, score: 0.0, distance: 0.0, crossing: false, breakdown: ScoreBreakdown::default() },
                    }
                };

                let mut note = current_notes[j];
                note.pitch = pick.note;
                note.muted = 0;
                temp_group_notes.push(note);
                group_score += pick.score;
            }
        }
    }

    // Optional soft per-common-tone penalty (in addition to the hard budget above).
    if config.common_tone_penalty != 0.0 {
        let common = temp_group_notes[base_len..].iter()
            .filter(|nnote| prev_pitch(nnote.channel) == Some(nnote.pitch))
            .count();
        group_score -= config.common_tone_penalty * common as f64;
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
        .map(|g| get_permutations(g, config.use_leading_voice))
        .collect();

    let beam_width = 5;
    let lookahead = config.lookahead_depth;

    let mut beam = vec![BeamCandidate {
        notes: Vec::new(),
        actual_score: 0.0,
    }];
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

                    let actual_score = beam_state.actual_score + group_score;
                    IntermediateCandidate {
                        parent_idx,
                        added_notes: temp_notes,
                        actual_score,
                        rank_score: actual_score + lookahead_score,
                    }
                })
            })
            .collect();

        candidates.sort_by(|a, b| b.rank_score.partial_cmp(&a.rank_score).unwrap());

        beam = candidates.into_iter().take(beam_width).map(|c| {
            let mut new_notes = current_beam[c.parent_idx].notes.clone();
            new_notes.extend(c.added_notes);
            BeamCandidate {
                notes: new_notes,
                actual_score: c.actual_score,
            }
        }).collect();
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
