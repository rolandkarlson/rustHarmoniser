use crate::model::{Note, Config};
use crate::utils::{SeededRng, ArrayExt, mod_shim, sin};
use crate::music_theory::{gen_scale};

use dashmap::DashMap;
use rayon::prelude::*;
use std::collections::{HashMap, BTreeMap};
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
// How much register-aware sensory roughness (Layer 3) modulates the pitch-class
// style preference. 0 = pure style/pitch-class; 1 = pure psychoacoustic roughness.
const ROUGHNESS_WEIGHT: f64 = 0.35;
// Chord-level aggregation weights (Layer 2): overall mean, worst single clash,
// and the interval against the bass (root/inversion sensitivity). Sum ≈ 1.
const AGG_MEAN: f64 = 0.40;
const AGG_WORST: f64 = 0.35;
const AGG_BASS: f64 = 0.25;

// Candidate-generation register limits (MIDI pitch).
const PITCH_MIN: i32 = 24; // search-window floor for the simple candidate path
const PITCH_MAX: i32 = 96; // search-window ceiling for the simple candidate path

// Soft penalty on holding a pitch that is outside the current candidate scale
// (Schillinger mode). Sized to beat the max smoothness edge a hold can have
// (distance score 1.0) so off-scale holds lose unless the voice-change budget
// forces them.
const OFF_SCALE_HOLD_PENALTY: f64 = 2.0;
const DEFAULT_BOUND_MIN: i32 = 24; // register floor when no voice below constrains
const DEFAULT_BOUND_MAX: i32 = 90; // register ceiling when no voice above constrains
// Per-channel crossing buffers (soprano→bass): how many semitones a candidate must
// stay clear of the bounding voice above (`UPPER`) and below (`LOWER`).
const CROSSING_BUFFER_UPPER: [i32; 5] = [2, 2, 2, 2, 7];
const CROSSING_BUFFER_LOWER: [i32; 5] = [2, 2, 2, 7, 1];
// Pitch span (≈ two octaves) normalizing the quartic voice-contour pull to ~[0, 1].
const CONTOUR_PITCH_SPAN: f64 = 24.0;

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

/// Melody force (config.melody_force): line-shaping pressure applied to EVERY
/// voice, unlike the leader-only repeat penalties (which decide who moves THIS
/// chord). Penalizes a candidate by how recently/often it appeared in the
/// voice's last 5 notes — recency-decayed (1.0, 0.8, 0.6, 0.4, 0.2), so
/// circling patterns like A-B-A-B are caught, not just immediate repeats — and
/// rewards stepwise motion (1-2 semitones) over both holds and leaps.
/// `lasts` is most-recent-first. Output ≈ [-3, 0.25] × weight.
fn melody_force_term(c: i32, lasts: &[i32], weight: f64) -> f64 {
    if weight == 0.0 {
        return 0.0;
    }
    let mut t = 0.0;
    for (k, &p) in lasts.iter().take(5).enumerate() {
        if p == c {
            t -= 1.0 - 0.2 * k as f64;
        }
    }
    if let Some(&last) = lasts.first() {
        let d = (c - last).abs();
        if d == 1 || d == 2 {
            t += 0.25;
        }
    }
    t * weight
}

#[derive(Clone, Debug)]
pub struct Boundaries {
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
    pub melody_force_contour: Option<Vec<Vec<f64>>>,
}

/// Effective melody-force weight for voice `channel_idx` at beat position
/// `start`: sample that voice's row of melody_force_contour if present and
/// non-empty (per-voice, like voice_contour), otherwise fall back to the
/// scalar `config.melody_force`.
fn melody_force_at(state: &HarmonizerState, config: &Config, channel_idx: usize, start: f64) -> f64 {
    if let Some(ref contours) = state.melody_force_contour {
        if !contours.is_empty() {
            let contour = &contours[mod_shim(channel_idx as i32, contours.len() as i32) as usize];
            if !contour.is_empty() {
                let idx = (start / state.harmony_contour_resolution).floor() as usize;
                return *contour.get_wrapped(idx);
            }
        }
    }
    config.melody_force
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

/// Tintinnabuli-style bound (original "p2rt" semantics, kept by preference over
/// the strict Pärt T−n/T+n positions): each lower voice is confined to the
/// anchor pitch class nearest above (Ceiling) / below (Floor) the lead's pitch
/// offset by `i - 1` semitones (i = channel), in pitch-class space. Adjacent
/// channels often land on the SAME class, giving the characteristic parallel
/// octave stacks. The octave stays free (candidates span ±1 octave; smoothness
/// + crossing pick the register). The lead itself must be given an empty
/// `current_lasts_lead` so it stays free.
fn apply_bound(notes: Vec<i32>, anchors: &[i32], config: &Config, current_lasts_lead: Vec<i32>, i: i32) -> Vec<i32> {
    if !config.use_ceiling && !config.use_floor { return notes; }
    if current_lasts_lead.is_empty() || anchors.is_empty() { return notes; }
    let mode = if config.use_ceiling { BoundMode::Ceiling } else { BoundMode::Floor };
    current_lasts_lead.into_iter()
        .map(|n| get_modular_bound(n + i - 1, anchors, 12, mode))
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

pub struct PrecomputedHarmonyData {
    pub sustaining_notes: Vec<i32>,
    pub boundaries_by_channel: Vec<Boundaries>,
    pub last_notes_by_channel: Vec<Vec<i32>>,
    pub notes_ending_at_start: Vec<Note>,
    pub lead_pitch: Option<i32>,
}

fn build_precomputed_data(context: &[Note], current_group: &[Note], start_time: f64) -> PrecomputedHarmonyData {
    let mut sustaining_notes = Vec::new();
    let mut notes_ending_at_start = Vec::new();
    let mut sustaining_at_minus_0_1 = Vec::new();
    let mut last_notes_by_channel: Vec<Vec<i32>> = vec![Vec::new(); 16];
    let mut sustaining_lead_pitch: Option<i32> = None;
    let mut latest_past_lead: Option<(f64, i32)> = None;

    for n in context {
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

    let mut boundaries_by_channel = Vec::with_capacity(16);
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
        boundaries_by_channel.push(Boundaries { min, max });
    }

    // Lead pitch priority: current group > sustaining from context > most recent past lead
    let lead_pitch = current_group.iter()
        .find(|n| n.channel == 0 && n.muted == 0)
        .map(|n| n.pitch)
        .or(sustaining_lead_pitch)
        .or(latest_past_lead.map(|(_, p)| p));

    PrecomputedHarmonyData {
        sustaining_notes,
        boundaries_by_channel,
        last_notes_by_channel,
        notes_ending_at_start,
        lead_pitch,
    }
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
    // Quantize start to 1e-4-beat ticks (matches the old "{:.4}" key) and group on
    // the integer tick — a BTreeMap keeps groups in ascending-start order for free,
    // with no per-note string allocation.
    let mut map: BTreeMap<i64, Vec<Note>> = BTreeMap::new();

    for n in notes {
        let key = (n.start * 10_000.0).round() as i64;
        map.entry(key).or_default().push(n);
    }

    let mut groups: Vec<Vec<Note>> = map.into_values().collect();

    // Channel order within a group; the beam re-sorts by channel anyway, so this
    // just makes the grouping output match that ordering directly.
    for g in &mut groups {
        g.sort_by_key(|n| n.channel);
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

// ===================== Joint chord scoring (config.use_joint_scoring) =========
// Replaces permutation orderings + greedy per-voice picks with a joint search
// over COMPLETE chords: every voice's candidate list is enumerated together and
// each finished chord is scored exactly once — full pairwise consonance with the
// mean/worst/bass aggregation applied to the actual final sonority, the
// voice-change budget as a filter instead of a repair pass, and the leader
// chosen per chord (equivalent to what the permutation sweep approximated).
// Hard constraints (forbidden intervals, unison collisions, budget excess) are
// counted as violations and compared lexicographically BEFORE the soft score,
// so -1e6 sentinels never leak into accumulated beam totals.

/// Per hard violation, subtracted from the group score returned to the outer
/// beam. Large enough to dominate soft terms, small enough that totals stay
/// readable and one forced violation doesn't erase all downstream soft scoring.
const VIOLATION_PENALTY: f64 = 1000.0;
/// Full Cartesian enumeration up to this many chords (covers 7^5 and ~15^4);
/// above it, a within-chord beam (bass-first) keeps cost bounded.
const JOINT_ENUM_CAP: usize = 60_000;
const JOINT_BEAM_WIDTH: usize = 64;

struct JointCand {
    pitch: i32,
    /// Melodic terms independent of the leader role and of other voices' NEW
    /// pitches: smoothness, contour pull, crossing.
    soft_base: f64,
    /// Repeat/history terms if this voice is the leader (hold penalties).
    lead_term: f64,
    /// Repeat/history terms if it is not (hold bonus / stickiness).
    nonlead_term: f64,
}

struct JointVoice {
    note: Note,
    prev: Option<i32>,
    is_fixed_lead: bool,
    cands: Vec<JointCand>,
}

/// Lead-voice pitch fixed by the input melody (same snap as the legacy
/// get_harmony_scores early return).
fn fixed_lead_pitch(note: &Note, state: &HarmonizerState, config: &Config) -> i32 {
    if config.schillinger_progression {
        let sch_scale = get_schillinger_scale(note, state, config, Vec::new());
        let center_octave = (note.pitch as f64 / 12.0).floor() as i32;
        gen_scale(&sch_scale, center_octave)
            .into_iter()
            .min_by_key(|&p| (p - note.pitch).abs())
            .unwrap_or(note.pitch)
    } else {
        note.pitch
    }
}

fn build_joint_voices(
    group: &[Note],
    w_smooth: f64,
    config: &Config,
    state: &HarmonizerState,
    precomputed: &PrecomputedHarmonyData,
) -> Vec<JointVoice> {
    // The lead pitch anchoring the ceil/floor tintinnabuli bounds: the melody
    // pitch when the lead is fixed, otherwise the lead's most recent HARMONIZED
    // pitch (the legacy sequential scorer bound lower voices to the lead note
    // placed in the same chord — the raw input lead pitch is a constant seed, so
    // anchoring to it would freeze every lower voice on one pitch class).
    let lead_fixed: Option<i32> = group.iter()
        .find(|n| n.channel == 0 && config.use_leading_voice)
        .map(|n| fixed_lead_pitch(n, state, config));
    let lead_for_bounds: Vec<i32> = lead_fixed
        .or_else(|| precomputed.last_notes_by_channel.first().and_then(|v| v.first()).copied())
        .or(precomputed.lead_pitch)
        .map(|p| vec![p])
        .unwrap_or_default();

    group.iter().map(|note| {
        let channel_idx = note.channel as usize;
        let prev = precomputed.last_notes_by_channel
            .get(channel_idx)
            .and_then(|v| v.first())
            .copied();

        if note.channel == 0 && config.use_leading_voice {
            return JointVoice {
                note: *note,
                prev,
                is_fixed_lead: true,
                cands: vec![JointCand {
                    pitch: lead_fixed.unwrap_or(note.pitch),
                    soft_base: 0.0,
                    lead_term: 0.0,
                    nonlead_term: 0.0,
                }],
            };
        }

        let mut current_lasts = precomputed.last_notes_by_channel
            .get(channel_idx)
            .cloned()
            .unwrap_or_default();
        if current_lasts.is_empty() {
            current_lasts.push(note.pitch);
        }
        let last_note = current_lasts[0];
        // Anti-stagnation: a voice frozen on one pitch for its whole history
        // window loses the hold bonus and takes the repeat penalties instead.
        let stagnant = current_lasts.len() >= 4
            && current_lasts.iter().all(|&x| x == current_lasts[0]);

        let range = config.candidate_range.max(1);
        let mut candidates: Vec<i32> = if config.schillinger_progression {
            // The lead voice is never bound to itself (legacy: its lead-anchor
            // list was empty), so ceil/floor confines only the lower voices.
            let bounds_lead = if note.channel == 0 { Vec::new() } else { lead_for_bounds.clone() };
            let sch_scale = get_schillinger_scale(note, state, config, bounds_lead);
            let center_octave = (last_note as f64 / 12.0).floor() as i32;
            gen_scale(&sch_scale, center_octave)
        } else {
            ((last_note - range).max(PITCH_MIN)..=(last_note + range).min(PITCH_MAX)).collect()
        };
        if candidates.is_empty() {
            candidates.push(note.pitch);
        }
        // Always offer the hold (common tone), even when the previous pitch has
        // left the current scale — the voice-change budget needs the hold option
        // to exist to act on this voice. But an off-scale hold is second-class:
        // it gets no stickiness bonus and takes a fixed penalty, so it only
        // survives when the budget genuinely forces this voice to hold.
        let off_scale_hold = !candidates.contains(&last_note);
        if off_scale_hold {
            candidates.push(last_note);
        }

        let mut target_offset: i32 = 0;
        let use_contour = if let Some(ref contours) = state.voice_contour {
            if !contours.is_empty() {
                let contour = &contours[mod_shim(channel_idx as i32, contours.len() as i32) as usize];
                if !contour.is_empty() {
                    let idx = (note.start / state.contour_resolution).floor() as usize;
                    target_offset = *contour.get_wrapped(idx);
                }
            }
            true
        } else {
            false
        };

        let default_bounds = Boundaries { min: DEFAULT_BOUND_MIN, max: DEFAULT_BOUND_MAX };
        let bounds = precomputed.boundaries_by_channel
            .get(channel_idx)
            .unwrap_or(&default_bounds);
        let cb_max = *CROSSING_BUFFER_UPPER.get_wrapped(channel_idx);
        let cb_min = *CROSSING_BUFFER_LOWER.get_wrapped(channel_idx);

        let eff_melody_force = melody_force_at(state, config, channel_idx, note.start);
        let cands = candidates.into_iter().map(|c| {
            let mut soft_base = get_distance_score(last_note, c) * w_smooth;
            soft_base += melody_force_term(c, &current_lasts, eff_melody_force);
            if c == last_note && off_scale_hold {
                soft_base -= OFF_SCALE_HOLD_PENALTY;
            }

            if config.voice_contour_weight != 0.0 {
                let base_dist = if use_contour {
                    (c - (note.pitch + target_offset)).abs()
                } else {
                    (c - note.pitch).abs()
                };
                let normalized = base_dist as f64 / CONTOUR_PITCH_SPAN;
                soft_base -= config.voice_contour_weight
                    * normalized * normalized * normalized * normalized;
            }

            if bounds.max - c < cb_max {
                soft_base -= config.no_crossing;
            }
            if c - bounds.min < cb_min {
                soft_base -= config.no_crossing;
            }

            let mut lead_term = 0.0;
            if c == last_note {
                lead_term -= config.last_note_same;
            }
            if current_lasts.iter().filter(|&&x| x == c).count() >= 2 {
                lead_term -= config.last_note_exist_in_voice;
            }
            let nonlead_term = if stagnant {
                lead_term
            } else if c == last_note && !off_scale_hold {
                config.same_note_bonus
            } else {
                0.0
            };

            JointCand { pitch: c, soft_base, lead_term, nonlead_term }
        }).collect();

        JointVoice { note: *note, prev, is_fixed_lead: false, cands }
    }).collect()
}

const BEAM_WIDTH: usize = 5;

/// Group-level harmony/smoothness weights and the consonance-matrix context for a
/// group, resolved from the optional contours or the scalar config fall-backs.
/// Returns `(w_harmony, w_smooth, harmony_ctx)`.
fn group_weights(group_start: f64, config: &Config, state: &HarmonizerState) -> (f64, f64, f64) {
    let idx = (group_start / state.harmony_contour_resolution).floor() as usize;
    let r = match &state.harmony_contour {
        Some(c) if !c.is_empty() => *c.get_wrapped(idx),
        _ => config.harmony_distance_balance,
    };
    let harmony_ctx = match &state.harmony_matrix_contour {
        Some(c) if !c.is_empty() => *c.get_wrapped(idx),
        _ => 0.0,
    };
    (0.5 + r, 0.5 - r, harmony_ctx)
}

/// Precomputed pairwise consonance over every pitch that can appear in the
/// sonority, so the chord-scoring hot loop is pure index lookups. `soft`/`forbidden`
/// are flat `m × m` tables indexed by `idx_of[pitch]`.
struct PairTable {
    pitches: Vec<i32>,
    idx_of: HashMap<i32, usize>,
    m: usize,
    soft: Vec<f64>,
    forbidden: Vec<bool>,
    sustaining_idx: Vec<usize>,
}

impl PairTable {
    fn build(voices: &[JointVoice], precomputed: &PrecomputedHarmonyData, row: &HarmonyRow) -> Self {
        let mut pitches: Vec<i32> = voices.iter()
            .flat_map(|v| v.cands.iter().map(|c| c.pitch))
            .collect();
        pitches.extend_from_slice(&precomputed.sustaining_notes);
        pitches.sort_unstable();
        pitches.dedup();
        let idx_of: HashMap<i32, usize> = pitches.iter().enumerate().map(|(i, &p)| (p, i)).collect();
        let m = pitches.len();
        let mut soft = vec![0.0f64; m * m];
        let mut forbidden = vec![false; m * m];
        for a in 0..m {
            for b in 0..m {
                let ic = ((pitches[a] - pitches[b]).abs() % 12) as usize;
                forbidden[a * m + b] = row.forbidden[ic];
                soft[a * m + b] = (1.0 - ROUGHNESS_WEIGHT) * row.soft[ic]
                    - ROUGHNESS_WEIGHT * pair_roughness(pitches[a], pitches[b]);
            }
        }
        let sustaining_idx = precomputed.sustaining_notes.iter().map(|p| idx_of[p]).collect();
        Self { pitches, idx_of, m, soft, forbidden, sustaining_idx }
    }
}

/// Lexicographic order on `(hard violations, soft score)`: fewer violations wins;
/// ties broken by the higher soft score. Returns true if `a` is strictly better.
fn chord_better(a: (u32, f64), b: (u32, f64)) -> bool {
    a.0 < b.0 || (a.0 == b.0 && a.1 > b.1)
}

/// Interval-variety duplicates within ONE sonority (no relation to the previous
/// or next chord). Octave/unison doublings (interval class 0) count beyond the
/// first at ANY chord size — a chord that is mostly one pitch class in different
/// octaves racks up penalties fast. Other repeated interval classes (stacked
/// fifths, augmented stacks, ...) count only while the sonority has at most 3
/// notes, where each interval should be a distinct color; 4+ note chords are
/// exempt from that part.
fn duplicate_interval_classes(pitches: &[i32]) -> u32 {
    let mut ic_count = [0u32; 12];
    for i in 0..pitches.len() {
        for j in (i + 1)..pitches.len() {
            ic_count[((pitches[i] - pitches[j]).abs() % 12) as usize] += 1;
        }
    }
    let mut dups = ic_count[0].saturating_sub(1);
    if pitches.len() <= INTERVAL_VARIETY_MAX_NOTES {
        dups += ic_count[1..].iter().map(|&c| c.saturating_sub(1)).sum::<u32>();
    }
    dups
}

/// Above this sonority size only octave/unison doublings are penalized; the
/// general repeated-interval rule applies to chords this small or smaller
/// (4+ note chords are free to repeat interval colors).
const INTERVAL_VARIETY_MAX_NOTES: usize = 3;

/// Scores complete chords (one candidate index per voice) against a prebuilt
/// `PairTable`. Holds only borrows, so it is cheap to share across rayon workers.
struct ChordScorer<'a> {
    voices: &'a [JointVoice],
    n: usize,
    table: &'a PairTable,
    sustaining_notes: &'a [i32],
    ending_by_channel: &'a HashMap<i32, i32>,
    check_dir: bool,
    w_harmony: f64,
    config: &'a Config,
}

impl ChordScorer<'_> {
    /// `(hard violations, soft score)` for one complete chord.
    fn eval(&self, chosen: &[usize]) -> (u32, f64) {
        let voices = self.voices;
        let n = self.n;
        let t = self.table;
        let m = t.m;
        let mut hard: u32 = 0;
        let mut soft = 0.0;

        // Per-voice base terms + leader selection: total with leader L =
        // base + Σ nonlead + max_L(lead(L) - nonlead(L)).
        let mut nonlead_sum = 0.0;
        let mut best_delta = f64::NEG_INFINITY;
        for (v, &ci) in voices.iter().zip(chosen) {
            let c = &v.cands[ci];
            soft += c.soft_base;
            if !v.is_fixed_lead {
                nonlead_sum += c.nonlead_term;
                best_delta = best_delta.max(c.lead_term - c.nonlead_term);
            }
        }
        soft += nonlead_sum;
        if best_delta > f64::NEG_INFINITY {
            soft += best_delta;
        }

        // Pairwise consonance over the full sonority: every pair touching a new
        // note (new-new and new-sustaining; sustaining-sustaining pairs are the
        // same for every candidate chord and would only dilute the aggregate).
        let mut bass = i32::MAX;
        for (v, &ci) in voices.iter().zip(chosen) {
            bass = bass.min(v.cands[ci].pitch);
        }
        for &s in self.sustaining_notes {
            bass = bass.min(s);
        }
        let mut sum = 0.0;
        let mut worst = f64::INFINITY;
        let mut cnt = 0usize;
        let mut bass_sum = 0.0;
        let mut bass_cnt = 0usize;
        for i in 0..n {
            let pi = voices[i].cands[chosen[i]].pitch;
            let ii = t.idx_of[&pi];
            for j in (i + 1)..n {
                let pj = voices[j].cands[chosen[j]].pitch;
                if pi == pj {
                    hard += 1; // exact unison collision between two voices
                    continue;
                }
                let k = ii * m + t.idx_of[&pj];
                if t.forbidden[k] {
                    hard += 1;
                }
                let ps = t.soft[k];
                sum += ps;
                cnt += 1;
                if ps < worst {
                    worst = ps;
                }
                if pi == bass || pj == bass {
                    bass_sum += ps;
                    bass_cnt += 1;
                }
            }
            for &sj in &t.sustaining_idx {
                let pj = t.pitches[sj];
                if pi == pj {
                    hard += 1;
                    continue;
                }
                let k = ii * m + sj;
                if t.forbidden[k] {
                    hard += 1;
                }
                let ps = t.soft[k];
                sum += ps;
                cnt += 1;
                if ps < worst {
                    worst = ps;
                }
                if pi == bass || pj == bass {
                    bass_sum += ps;
                    bass_cnt += 1;
                }
            }
        }
        if cnt > 0 {
            let mean = sum / cnt as f64;
            let bass_term = if bass_cnt > 0 { bass_sum / bass_cnt as f64 } else { mean };
            soft += self.w_harmony * (AGG_MEAN * mean + AGG_WORST * worst + AGG_BASS * bass_term);
        }

        // Interval-variety pressure within this one sonority (new + sustaining):
        // repeated octave doublings are penalized at any size, other repeated
        // interval classes only while the chord has <= 3 notes — see
        // duplicate_interval_classes (config.interval_exists_in_harmony, 0 = off).
        // Skipped entirely in floor/ceiling mode: the bound confines every
        // non-lead voice to one anchor pitch class, so octave doublings are the
        // intended sonority there and the penalty would only push voices onto
        // off-scale holds instead.
        let total_notes = n + self.sustaining_notes.len();
        if self.config.interval_exists_in_harmony != 0.0
            && !self.config.use_floor
            && !self.config.use_ceiling
            && total_notes >= 3
        {
            let mut sonority: Vec<i32> = Vec::with_capacity(total_notes);
            sonority.extend(voices.iter().zip(chosen).map(|(v, &ci)| v.cands[ci].pitch));
            sonority.extend_from_slice(self.sustaining_notes);
            let dups = duplicate_interval_classes(&sonority);
            soft -= self.config.interval_exists_in_harmony * dups as f64;
        }

        // Parallel 5ths/octaves among the new voices (sustaining notes have zero
        // motion at this instant).
        if self.config.consecutive_octav_fift != 0.0 {
            for i in 0..n {
                let Some(qi) = voices[i].prev else { continue };
                let pi = voices[i].cands[chosen[i]].pitch;
                let mi = (pi - qi).signum();
                if mi == 0 {
                    continue;
                }
                for j in (i + 1)..n {
                    let Some(qj) = voices[j].prev else { continue };
                    let pj = voices[j].cands[chosen[j]].pitch;
                    if (pj - qj).signum() != mi {
                        continue;
                    }
                    let prev_int = (qi - qj).abs() % 12;
                    let cur_int = (pi - pj).abs() % 12;
                    if prev_int == cur_int && (cur_int == 0 || cur_int == 7) {
                        soft -= self.config.consecutive_octav_fift;
                    }
                }
            }
        }

        // Same-direction penalty for outer voices moving with the chord majority
        // (held notes are oblique motion and exempt).
        if self.check_dir {
            for (v, &ci) in voices.iter().zip(chosen) {
                if v.note.channel != 0 && v.note.channel != 4 {
                    continue;
                }
                let Some(q) = v.prev else { continue };
                let mv = (v.cands[ci].pitch - q).signum();
                if mv == 0 {
                    continue;
                }
                let mut up = 0;
                let mut down = 0;
                for (w, &cj) in voices.iter().zip(chosen) {
                    if w.note.channel == v.note.channel {
                        continue;
                    }
                    let Some(&from) = self.ending_by_channel.get(&w.note.channel) else { continue };
                    let d = (w.cands[cj].pitch - from).signum();
                    if d > 0 {
                        up += 1;
                    } else if d < 0 {
                        down += 1;
                    }
                }
                if (mv > 0 && up > down) || (mv < 0 && down > up) {
                    soft -= self.config.same_direction;
                }
            }
        }

        // Voice-change budget as a constraint, common tones as a soft term.
        let mut movers: i32 = 0;
        let mut common: i32 = 0;
        for (v, &ci) in voices.iter().zip(chosen) {
            if let Some(q) = v.prev {
                let held = v.cands[ci].pitch == q;
                if held {
                    common += 1;
                }
                if !v.is_fixed_lead && !held {
                    movers += 1;
                }
            }
        }
        if self.config.max_voices_changed >= 0 && movers > self.config.max_voices_changed {
            hard += (movers - self.config.max_voices_changed) as u32;
        }
        if self.config.min_voices_changed >= 0 && movers < self.config.min_voices_changed {
            hard += (self.config.min_voices_changed - movers) as u32;
        }
        if self.config.common_tone_penalty != 0.0 {
            soft -= self.config.common_tone_penalty * common as f64;
        }

        (hard, soft)
    }

    /// Exhaustive search over the candidate product — parallel across the first
    /// voice's candidates, odometer over the rest. Optimal within the window.
    fn enumerate(&self) -> Option<(Vec<usize>, (u32, f64))> {
        let n = self.n;
        (0..self.voices[0].cands.len()).into_par_iter()
            .map(|c0| {
                let mut chosen = vec![0usize; n];
                chosen[0] = c0;
                let mut local_best: Option<(Vec<usize>, (u32, f64))> = None;
                loop {
                    let sc = self.eval(&chosen);
                    if local_best.as_ref().map_or(true, |b| chord_better(sc, b.1)) {
                        local_best = Some((chosen.clone(), sc));
                    }
                    // Advance the odometer over voices 1..n.
                    let mut pos = n - 1;
                    loop {
                        if pos == 0 {
                            return local_best;
                        }
                        chosen[pos] += 1;
                        if chosen[pos] < self.voices[pos].cands.len() {
                            break;
                        }
                        chosen[pos] = 0;
                        pos -= 1;
                    }
                }
            })
            .reduce(|| None, |a, b| match (a, b) {
                (Some(x), Some(y)) => Some(if chord_better(x.1, y.1) { x } else { y }),
                (x, None) => x,
                (None, y) => y,
            })
    }

    /// Bass-first within-chord beam for groups whose candidate product exceeds
    /// `JOINT_ENUM_CAP`. Partial states are ranked by running
    /// `(violations, partial soft)`; survivors get the exact chord score.
    fn beam_search(&self) -> Option<(Vec<usize>, (u32, f64))> {
        let n = self.n;
        let t = self.table;
        let m = t.m;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(self.voices[i].note.channel));

        let mut states: Vec<(Vec<usize>, u32, f64)> = vec![(Vec::new(), 0, 0.0)];
        for &vi in &order {
            let mut next: Vec<(Vec<usize>, u32, f64)> =
                Vec::with_capacity(states.len() * self.voices[vi].cands.len());
            for (part, hard, soft) in &states {
                for (ci, cand) in self.voices[vi].cands.iter().enumerate() {
                    let mut h = *hard;
                    let mut s = *soft + cand.soft_base + cand.nonlead_term;
                    let ii = t.idx_of[&cand.pitch];
                    for (pos, &cj) in part.iter().enumerate() {
                        let pj = self.voices[order[pos]].cands[cj].pitch;
                        if pj == cand.pitch {
                            h += 1;
                            continue;
                        }
                        let k = ii * m + t.idx_of[&pj];
                        if t.forbidden[k] {
                            h += 1;
                        }
                        s += t.soft[k];
                    }
                    for &sj in &t.sustaining_idx {
                        if t.pitches[sj] == cand.pitch {
                            h += 1;
                            continue;
                        }
                        let k = ii * m + sj;
                        if t.forbidden[k] {
                            h += 1;
                        }
                        s += t.soft[k];
                    }
                    let mut cp = part.clone();
                    cp.push(ci);
                    next.push((cp, h, s));
                }
            }
            next.sort_by(|a, b| a.1.cmp(&b.1)
                .then(b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)));
            next.truncate(JOINT_BEAM_WIDTH);
            states = next;
        }

        states.into_iter()
            .map(|(part, _, _)| {
                let mut chosen = vec![0usize; n];
                for (pos, &ci) in part.iter().enumerate() {
                    chosen[order[pos]] = ci;
                }
                let sc = self.eval(&chosen);
                (chosen, sc)
            })
            .fold(None, |acc: Option<(Vec<usize>, (u32, f64))>, x| {
                match acc {
                    Some(a) if chord_better(a.1, x.1) => Some(a),
                    _ => Some(x),
                }
            })
    }
}

/// Score one rhythmic group: build each voice's candidate set, then search the
/// joint candidate space for the chord with the fewest hard violations and the
/// best soft score. The chosen pitches (one per input note) are appended to
/// `temp_group_notes`; the return value is the group's soft score net of the
/// hard-violation penalty.
fn score_group(
    group: &[Note],
    temp_group_notes: &mut Vec<Note>,
    config: &Config,
    state: &HarmonizerState,
    precomputed: &PrecomputedHarmonyData,
) -> f64 {
    if group.is_empty() {
        return 0.0;
    }

    let (w_harmony, w_smooth, harmony_ctx) = group_weights(group[0].start, config, state);
    let row = get_harmony_row(harmony_ctx, state.harmony_matrix.as_ref());

    let voices = build_joint_voices(group, w_smooth, config, state, precomputed);
    let n = voices.len();
    let table = PairTable::build(&voices, precomputed, &row);

    let ending_by_channel: HashMap<i32, i32> = precomputed.notes_ending_at_start.iter()
        .map(|nn| (nn.channel, nn.pitch))
        .collect();
    let check_dir = !ending_by_channel.is_empty() && config.same_direction != 0.0;

    let scorer = ChordScorer {
        voices: &voices,
        n,
        table: &table,
        sustaining_notes: &precomputed.sustaining_notes,
        ending_by_channel: &ending_by_channel,
        check_dir,
        w_harmony,
        config,
    };

    let product = voices.iter()
        .map(|v| v.cands.len())
        .try_fold(1usize, |acc, l| acc.checked_mul(l))
        .unwrap_or(usize::MAX);

    let best = if product <= JOINT_ENUM_CAP {
        scorer.enumerate()
    } else {
        scorer.beam_search()
    };

    let Some((chosen, (hard, soft))) = best else {
        // No candidates at all — pass the group through unchanged.
        for nn in group {
            let mut note = *nn;
            note.muted = 0;
            temp_group_notes.push(note);
        }
        return 0.0;
    };

    for (v, &ci) in voices.iter().zip(&chosen) {
        let mut note = v.note;
        note.pitch = v.cands[ci].pitch;
        note.muted = 0;
        temp_group_notes.push(note);
    }

    soft - hard as f64 * VIOLATION_PENALTY
}

/// Greedy look-ahead: score the remaining groups one at a time (each conditioned
/// on the running context), memoized on a hash of `(group index, depth, recent
/// context)`. Returns the summed soft score over the look-ahead horizon.
fn score_lookahead(
    groups: &[Vec<Note>],
    start_idx: usize,
    depth: i32,
    context: &[Note],
    config: &Config,
    state: &HarmonizerState,
    cache: &DashMap<u64, f64>,
) -> f64 {
    if depth == 0 || start_idx >= groups.len() {
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

    let group = &groups[start_idx];
    let start_time = group[0].start;
    let precomputed = build_precomputed_data(context, group, start_time);

    let mut temp_notes = Vec::new();
    let local_score = score_group(group, &mut temp_notes, config, state, &precomputed);

    let mut next_context = context.to_vec();
    next_context.extend(temp_notes);

    let best_score = local_score
        + score_lookahead(groups, start_idx + 1, depth - 1, &next_context, config, state, cache);

    cache.insert(key, best_score);
    best_score
}

use std::sync::mpsc::Sender;

fn score_group_beam(income: Vec<Note>, config: &Config, state: &HarmonizerState, progress_sender: Option<&Sender<(usize, usize)>>) -> Vec<Note> {
    // Each group's notes come back in channel order from group_by_start_array.
    let groups = group_by_start_array(income);
    let lookahead = config.lookahead_depth;

    let mut beam = vec![BeamCandidate {
        notes: Vec::new(),
        actual_score: 0.0,
    }];
    for (i, group) in groups.iter().enumerate() {
        if let Some(sender) = progress_sender {
            let _ = sender.send((i, groups.len()));
        }

        let cache: DashMap<u64, f64> = DashMap::new();
        let groups_ref = &groups;
        let cache_ref = &cache;
        let current_beam = &beam;

        let mut candidates: Vec<IntermediateCandidate> = current_beam
            .par_iter()
            .enumerate()
            .map(|(parent_idx, beam_state)| {
                let start_time = group[0].start;
                // Trim the scoring context to a recent window. 32 beats comfortably
                // covers the longest sustain (notes are clamped to one 4-beat bar)
                // and the 5-note-per-channel history.
                let cutoff = start_time - 32.0;
                let begin = beam_state.notes.partition_point(|n| n.start < cutoff);
                let trimmed_notes = &beam_state.notes[begin..];

                let precomputed = build_precomputed_data(trimmed_notes, group, start_time);

                let mut temp_notes = Vec::new();
                let group_score = score_group(group, &mut temp_notes, config, state, &precomputed);

                let mut next_context = trimmed_notes.to_vec();
                next_context.extend(temp_notes.iter().cloned());

                let lookahead_score = score_lookahead(
                    groups_ref, i + 1, lookahead, &next_context, config, state, cache_ref,
                );

                let actual_score = beam_state.actual_score + group_score;
                IntermediateCandidate {
                    parent_idx,
                    added_notes: temp_notes,
                    actual_score,
                    rank_score: actual_score + lookahead_score,
                }
            })
            .collect();

        candidates.sort_by(|a, b| {
            b.rank_score.partial_cmp(&a.rank_score).unwrap_or(std::cmp::Ordering::Equal)
        });

        beam = candidates.into_iter().take(BEAM_WIDTH).map(|c| {
            let mut new_notes = current_beam[c.parent_idx].notes.clone();
            new_notes.extend(c.added_notes);
            BeamCandidate {
                notes: new_notes,
                actual_score: c.actual_score,
            }
        }).collect();
    }

    beam.first().map(|b| b.notes.clone()).unwrap_or_default()
}

pub fn harmonise2(income: Vec<Note>, config: &Config, state: &HarmonizerState, progress_sender: Option<&Sender<(usize, usize)>>) -> Vec<Note> {
    score_group_beam(income, config, state, progress_sender)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Config, Note};

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    /// Minimal state with all optional contours/matrix unset — exercises the
    /// scalar fall-back paths only.
    fn test_state() -> HarmonizerState {
        HarmonizerState {
            schillinger_notes: vec![],
            voice_contour: None,
            contour_resolution: 4.0,
            harmony_contour: None,
            harmony_contour_resolution: 4.0,
            harmony_matrix_contour: None,
            harmony_matrix: None,
            melody_force_contour: None,
        }
    }

    /// Default config switched to the simple (non-Schillinger) candidate path so
    /// tests don't need to populate `state.schillinger_notes`.
    fn test_config() -> Config {
        Config {
            schillinger_progression: false,
            use_leading_voice: false,
            ..Config::default()
        }
    }

    // ----- get_modular_bound (tintinnabuli bound, pc-space semantics) -----

    #[test]
    fn modular_bound_snaps_to_nearest_anchor() {
        let anchors = [0, 4, 7];
        assert_eq!(get_modular_bound(60, &anchors, 12, BoundMode::Ceiling), 4); // strictly above
        assert_eq!(get_modular_bound(62, &anchors, 12, BoundMode::Ceiling), 4);
        assert_eq!(get_modular_bound(62, &anchors, 12, BoundMode::Floor), 0);
        assert_eq!(get_modular_bound(60, &anchors, 12, BoundMode::Floor), 7); // wraps below
    }

    // ----- get_distance_score -----

    #[test]
    fn distance_score_unison_is_max() {
        assert!(approx(get_distance_score(60, 60), 1.0));
    }

    #[test]
    fn distance_score_fifth_is_zero() {
        // dist 7 → 1 - 7/7 = 0
        assert!(approx(get_distance_score(60, 67), 0.0));
    }

    #[test]
    fn distance_score_large_leap_clamps_to_minus_one() {
        assert!(approx(get_distance_score(60, 90), -1.0));
    }

    #[test]
    fn distance_score_symmetric_and_monotonic() {
        assert!(approx(get_distance_score(60, 67), get_distance_score(67, 60)));
        assert!(get_distance_score(60, 61) > get_distance_score(60, 63));
        assert!(get_distance_score(60, 63) > get_distance_score(60, 67));
    }

    // ----- melody_force_term -----

    #[test]
    fn melody_force_zero_weight_is_noop() {
        assert_eq!(melody_force_term(60, &[60, 60, 60], 0.0), 0.0);
    }

    #[test]
    fn melody_force_penalizes_immediate_repeat() {
        // c == lasts[0], no stepwise reward (dist 0) → -1.0 * weight
        assert!(approx(melody_force_term(60, &[60], 1.0), -1.0));
    }

    #[test]
    fn melody_force_recency_decays_and_caps_at_five() {
        // six repeats; only the first five count: -(1 + .8 + .6 + .4 + .2) = -3.0
        let lasts = [60, 60, 60, 60, 60, 60];
        assert!(approx(melody_force_term(60, &lasts, 1.0), -3.0));
    }

    #[test]
    fn melody_force_rewards_stepwise_motion() {
        assert!(approx(melody_force_term(61, &[60], 1.0), 0.25)); // step of 1
        assert!(approx(melody_force_term(62, &[60], 1.0), 0.25)); // step of 2
        assert!(approx(melody_force_term(63, &[60], 1.0), 0.0)); // leap of 3: neither
    }

    #[test]
    fn melody_force_scales_with_weight() {
        assert!(approx(melody_force_term(60, &[60], 2.0), -2.0));
    }

    // ----- pair_roughness -----

    #[test]
    fn roughness_unison_is_zero() {
        assert_eq!(pair_roughness(60, 60), 0.0);
    }

    #[test]
    fn roughness_symmetric_and_bounded() {
        for &(a, b) in &[(60, 61), (48, 55), (36, 37), (72, 84)] {
            let r = pair_roughness(a, b);
            assert!((0.0..=1.0).contains(&r), "{r} out of range");
            assert!(approx(pair_roughness(a, b), pair_roughness(b, a)));
        }
    }

    #[test]
    fn roughness_minor_second_rougher_than_fifth() {
        assert!(pair_roughness(60, 61) > pair_roughness(60, 67));
    }

    #[test]
    fn roughness_decays_with_spacing() {
        // Same pitch class: a narrow minor 2nd is far rougher than a wide minor 9th.
        assert!(pair_roughness(60, 61) > pair_roughness(60, 73));
        assert!(pair_roughness(48, 49) > pair_roughness(48, 61));
        // Beyond ~an octave the fundamentals barely interfere → roughness ≈ 0.
        assert!(pair_roughness(60, 73) < 0.01);
    }

    // ----- duplicate_interval_classes -----

    #[test]
    fn interval_variety_distinct_triad_has_no_duplicates() {
        // Major triad: ics 4, 3, 7 — all distinct.
        assert_eq!(duplicate_interval_classes(&[60, 64, 67]), 0);
    }

    #[test]
    fn interval_variety_counts_repeated_classes() {
        // Augmented triad: ics 4, 4, 8 — one duplicate.
        assert_eq!(duplicate_interval_classes(&[60, 64, 68]), 1);
        // Stacked fifths: ics 7, 2, 7 — one duplicate.
        assert_eq!(duplicate_interval_classes(&[60, 67, 74]), 1);
    }

    #[test]
    fn interval_variety_two_notes_never_duplicate() {
        assert_eq!(duplicate_interval_classes(&[60, 67]), 0);
    }

    #[test]
    fn interval_variety_octave_doublings_penalized_at_any_size() {
        // All C in five octaves: 10 pairs, all class 0 → 9 duplicates.
        assert_eq!(duplicate_interval_classes(&[36, 48, 60, 72, 84]), 9);
        // Three C octaves inside a 5-note chord: classes 0,0,0 among the Cs.
        assert_eq!(duplicate_interval_classes(&[48, 60, 72, 64, 67]), 2);
    }

    #[test]
    fn interval_variety_four_note_chords_only_count_octaves() {
        // 4-note chords are exempt from the repeated-interval rule: a dim7
        // (ics 3,6,9,3,6,3) and a quartal stack (5 x3, 10 x2, 3 x1) both pass.
        assert_eq!(duplicate_interval_classes(&[60, 63, 66, 69]), 0);
        assert_eq!(duplicate_interval_classes(&[50, 55, 60, 65]), 0);
        // 5-note quartal stack likewise — no octave doublings, no penalty.
        assert_eq!(duplicate_interval_classes(&[50, 55, 60, 65, 70]), 0);
        // But octave doublings still count at 4 notes: C-G-C-G → two class-0
        // pairs → 1 duplicate.
        assert_eq!(duplicate_interval_classes(&[48, 55, 60, 67]), 1);
    }

    // ----- get_harmony_row -----

    #[test]
    fn harmony_row_default_is_strict_classical() {
        let row = get_harmony_row(0.0, None); // row 0 of HARMONY_MATRIX
        assert!(approx(row.soft[0], 1.0)); // P1
        assert!(approx(row.soft[7], 1.0)); // P5
        assert!(row.forbidden[1]); // m2 (-100)
        assert!(row.forbidden[6]); // tritone (-100)
        assert!(!row.forbidden[0]);
        // A forbidden cell contributes only -1 to the soft surface, not -100.
        assert!(approx(row.soft[1], -1.0));
    }

    #[test]
    fn harmony_row_clamps_context() {
        let hi = get_harmony_row(100.0, None);
        let cap = get_harmony_row(8.0, None);
        assert_eq!(hi.soft, cap.soft);
        assert_eq!(hi.forbidden, cap.forbidden);

        let lo = get_harmony_row(-5.0, None);
        let zero = get_harmony_row(0.0, None);
        assert_eq!(lo.soft, zero.soft);
    }

    #[test]
    fn harmony_row_custom_matrix_lerps() {
        // row 0 all 0.0, row 1 all 1.0 → ctx 0.5 yields 0.5 everywhere.
        let mut m = vec![vec![0.0; 12]; 9];
        for v in m[1].iter_mut() {
            *v = 1.0;
        }
        let row = get_harmony_row(0.5, Some(&m));
        for i in 0..12 {
            assert!(approx(row.soft[i], 0.5), "col {i} = {}", row.soft[i]);
        }
    }

    #[test]
    fn harmony_row_custom_forbidden_blend() {
        // row 0 col 0 forbidden (-100), row 1 col 0 = 0.5.
        let mut m = vec![vec![0.0; 12]; 9];
        m[0][0] = -100.0;
        m[1][0] = 0.5;
        let row = get_harmony_row(0.5, Some(&m));
        // t = 0.5: lo_forb && (1-t) >= 0.5 → forbidden.
        assert!(row.forbidden[0]);
        // soft = clamp(-100 → -1)*0.5 + 0.5*0.5 = -0.25.
        assert!(approx(row.soft[0], -0.25));
    }

    #[test]
    fn harmony_row_malformed_custom_falls_back_to_default() {
        let bad = vec![vec![0.0; 11]; 9]; // 11 columns ≠ 12 → invalid
        let row = get_harmony_row(0.0, Some(&bad));
        let def = get_harmony_row(0.0, None);
        assert_eq!(row.soft, def.soft);
        assert_eq!(row.forbidden, def.forbidden);
    }

    // ----- build_precomputed_data -----

    #[test]
    fn precomputed_history_is_most_recent_first() {
        let ctx = vec![
            Note::new(60, 0.0, 1.0, 100, 0, 1),
            Note::new(62, 1.0, 1.0, 100, 0, 1),
            Note::new(64, 2.0, 1.0, 100, 0, 1),
        ];
        let group = vec![Note::new(65, 3.0, 1.0, 100, 0, 1)];
        let pc = build_precomputed_data(&ctx, &group, 3.0);
        assert_eq!(pc.last_notes_by_channel[1], vec![64, 62, 60]);
    }

    #[test]
    fn precomputed_history_capped_at_five() {
        let ctx: Vec<Note> = (0..8i32)
            .map(|i| Note::new(60 + i, i as f64, 1.0, 100, 0, 1))
            .collect();
        let group = vec![Note::new(80, 8.0, 1.0, 100, 0, 1)];
        let pc = build_precomputed_data(&ctx, &group, 8.0);
        // Only the 5 most recent (starts 3..7, pitches 63..67), newest first.
        assert_eq!(pc.last_notes_by_channel[1], vec![67, 66, 65, 64, 63]);
    }

    #[test]
    fn precomputed_detects_sustaining_note() {
        let ctx = vec![Note::new(48, 0.0, 4.0, 100, 0, 4)]; // spans t=2
        let group = vec![Note::new(60, 2.0, 1.0, 100, 0, 0)];
        let pc = build_precomputed_data(&ctx, &group, 2.0);
        assert!(pc.sustaining_notes.contains(&48));
    }

    // ----- harmonise2 (end-to-end) -----

    fn two_chord_input() -> Vec<Note> {
        vec![
            Note::new(48, 0.0, 4.0, 100, 0, 4),
            Note::new(60, 0.0, 4.0, 100, 0, 0),
            Note::new(48, 4.0, 4.0, 100, 0, 4),
            Note::new(60, 4.0, 4.0, 100, 0, 0),
        ]
    }

    #[test]
    fn harmonise_empty_input_yields_empty_output() {
        assert!(harmonise2(vec![], &test_config(), &test_state(), None).is_empty());
    }

    #[test]
    fn harmonise_preserves_count_channels_and_starts() {
        let input = two_chord_input();
        let out = harmonise2(input.clone(), &test_config(), &test_state(), None);

        assert_eq!(out.len(), input.len());
        for n in &out {
            assert_eq!(n.muted, 0);
        }
        // Each (channel, start) slot is preserved; only the pitch is reassigned.
        let key = |n: &Note| (n.channel, (n.start * 1000.0).round() as i64);
        let mut in_keys: Vec<_> = input.iter().map(key).collect();
        let mut out_keys: Vec<_> = out.iter().map(key).collect();
        in_keys.sort();
        out_keys.sort();
        assert_eq!(in_keys, out_keys);
    }

    #[test]
    fn harmonise_first_chord_stays_within_candidate_range() {
        let cfg = test_config(); // candidate_range = 3, no prior history
        let input = vec![
            Note::new(48, 0.0, 4.0, 100, 0, 4),
            Note::new(60, 0.0, 4.0, 100, 0, 0),
        ];
        let out = harmonise2(input, &cfg, &test_state(), None);
        for n in &out {
            let base = if n.channel == 4 { 48 } else { 60 };
            assert!(
                (n.pitch - base).abs() <= cfg.candidate_range,
                "ch{} pitch {} outside ±{} of {}",
                n.channel,
                n.pitch,
                cfg.candidate_range,
                base
            );
        }
    }

    #[test]
    fn harmonise_is_deterministic() {
        let cfg = test_config();
        let input: Vec<Note> = (0..6i32)
            .flat_map(|g| {
                let s = g as f64 * 4.0;
                [
                    Note::new(48, s, 4.0, 100, 0, 4),
                    Note::new(60, s, 4.0, 100, 0, 0),
                ]
            })
            .collect();
        let a: Vec<i32> = harmonise2(input.clone(), &cfg, &test_state(), None)
            .iter()
            .map(|n| n.pitch)
            .collect();
        let b: Vec<i32> = harmonise2(input, &cfg, &test_state(), None)
            .iter()
            .map(|n| n.pitch)
            .collect();
        assert_eq!(a, b);
    }

    fn pitch_by_channel(out: &[Note], start: f64) -> HashMap<i32, i32> {
        out.iter()
            .filter(|n| (n.start - start).abs() < 1e-6)
            .map(|n| (n.channel, n.pitch))
            .collect()
    }

    #[test]
    fn budget_max_zero_forces_voices_to_hold() {
        let mut cfg = test_config();
        cfg.max_voices_changed = 0; // no voice may change pitch
        cfg.min_voices_changed = -1;
        let out = harmonise2(two_chord_input(), &cfg, &test_state(), None);
        let g1 = pitch_by_channel(&out, 0.0);
        let g2 = pitch_by_channel(&out, 4.0);
        // Second chord holds the first chord's pitches in every voice.
        assert_eq!(g1.get(&0), g2.get(&0));
        assert_eq!(g1.get(&4), g2.get(&4));
    }

    #[test]
    fn schillinger_off_scale_input_snaps_to_scale() {
        // The GUI "all penalties zero" scenario: with every penalty at 0 and the
        // default same_note_bonus, non-leader voices must not hold their
        // off-scale initial pitches — every output note belongs to the scale.
        let mut cfg = test_config();
        cfg.schillinger_progression = true;
        cfg.last_note_same = 0.0;
        cfg.last_note_exist_in_voice = 0.0;
        cfg.same_direction = 0.0;
        cfg.interval_exists_in_harmony = 0.0;
        cfg.no_crossing = 0.0;
        cfg.common_tone_penalty = 0.0;
        cfg.melody_force = 0.0;
        cfg.voice_contour_weight = 0.0;
        let mut state = test_state();
        // One scale entry (used by every channel), one bar, C-major triad.
        state.schillinger_notes = vec![vec![vec![0, 4, 7]]];
        // Off-scale inputs (C#4, C#3) held across two chords: the leader moves
        // (min_voices_changed = 1) but the non-leader must not keep its
        // off-scale common tone just because of the stickiness bonus.
        let input = vec![
            Note::new(61, 0.0, 4.0, 100, 0, 0),
            Note::new(49, 0.0, 4.0, 100, 0, 4),
            Note::new(61, 4.0, 4.0, 100, 0, 0),
            Note::new(49, 4.0, 4.0, 100, 0, 4),
        ];
        let out = harmonise2(input, &cfg, &state, None);
        for n in &out {
            assert!(
                [0, 4, 7].contains(&(n.pitch.rem_euclid(12))),
                "pitch {} at start {} (ch {}) is not on the Schillinger scale",
                n.pitch,
                n.start,
                n.channel
            );
        }
    }

    #[test]
    fn floor_bound_snaps_to_scale_despite_interval_penalty() {
        // use_floor confines every non-lead voice to one anchor pitch class, so
        // octave stacks are the INTENDED sonority. The interval-variety penalty
        // (interval_exists_in_harmony) must not make off-scale holds cheaper
        // than the bounded scale notes — it is skipped in floor/ceiling mode.
        let mut cfg = test_config();
        cfg.schillinger_progression = true;
        cfg.use_floor = true;
        cfg.use_leading_voice = true;
        cfg.interval_exists_in_harmony = 1.0;
        cfg.same_note_bonus = 0.0;
        let mut state = test_state();
        // Lead melody on pc 8 (its own richer scale), lower voices confined to
        // the triad — the floor bound then puts EVERY lower voice on pc 7, an
        // intentional octave stack.
        state.schillinger_notes = vec![
            vec![vec![0, 3, 7, 10, 2, 5, 8]],
            vec![vec![0, 3, 7]],
            vec![vec![0, 3, 7]],
            vec![vec![0, 3, 7]],
            vec![vec![0, 3, 7]],
        ];
        // Neutral consonance row + smoothness-heavy balance (the config that
        // exposed the bug): with no consonance signal, the octave-dup penalty
        // alone was enough to make off-scale holds win.
        state.harmony_matrix_contour = Some(vec![8.0]);
        state.harmony_contour = Some(vec![-0.2]);
        // Non-lead voices seeded mostly off-scale (muted=1 engages the bound
        // path, like generated voices), two chords.
        let mut input = Vec::new();
        for (ch, pitch) in [(0, 80), (1, 75), (2, 70), (3, 60), (4, 45)] {
            input.push(Note::new(pitch, 0.0, 4.0, 100, 1, ch));
            input.push(Note::new(pitch, 4.0, 4.0, 100, 1, ch));
        }
        let out = harmonise2(input, &cfg, &state, None);
        // With the lead fixed on 80 (pc 8) over the [0,3,7] triad, the pc-space
        // bound floor(80 + ch − 1) lands on pc 7 for every lower voice —
        // anything else means an off-scale hold won.
        for n in out.iter().filter(|n| n.channel != 0) {
            assert_eq!(
                n.pitch.rem_euclid(12),
                7,
                "pitch {} at start {} (ch {}) is not on the floor-bound anchor",
                n.pitch,
                n.start,
                n.channel
            );
        }
    }

    #[test]
    fn budget_min_overrides_hold_bias_and_forces_movement() {
        let mut cfg = test_config();
        cfg.min_voices_changed = 2; // both non-lead voices must move
        cfg.max_voices_changed = -1;
        cfg.same_note_bonus = 5.0; // strong stickiness the budget must override
        let out = harmonise2(two_chord_input(), &cfg, &test_state(), None);
        let g1 = pitch_by_channel(&out, 0.0);
        let g2 = pitch_by_channel(&out, 4.0);
        assert_ne!(g1.get(&0), g2.get(&0), "ch0 should have moved");
        assert_ne!(g1.get(&4), g2.get(&4), "ch4 should have moved");
    }
}
