use crate::contour::Contours;
use crate::trace::{GroupBreakdown, TraceCollector, VoiceBreakdown};
use crate::model::{Note, Config, ChordTemplate};
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
    // 0: STRICT CLASSICAL — consonance-first; m2 / M7 forbidden. The tritone is
    // strongly disfavoured but NOT forbidden: hard-forbidding it outlaws the
    // dominant seventh and the diminished triad, i.e. exactly the sonorities
    // tonal cadences are built from. Its resolution is handled as a tendency
    // (see tendency_term) rather than by banning the interval.
    [1.0, -100.0, -0.4, 0.8, 0.9, 0.5, -0.5, 1.0, 0.7, 0.8, -0.3, -100.0],
    // 1: JAZZ & COLOR — 7ths/9ths/tritone embraced; bare unison/octave duller
    [0.6, 0.0, 0.7, 0.8, 0.9, 0.5, 0.6, 0.9, 0.5, 0.8, 1.0, 0.8],
    // 2: TENSION/RESOLUTION — tritones and leading tones prized, triads dull
    [-0.2, 0.8, 0.2, -0.3, -0.3, -0.2, 1.0, 0.0, -0.3, -0.3, 0.5, 0.9],
    // 3: ETHEREAL & OPEN — quartal/quintal and open 2nds/9ths; m2 forbidden.
    // m7 sits high: two stacked 4ths (the row's signature sonority) span one.
    [1.0, -100.0, 0.8, -0.2, 0.2, 1.0, -0.5, 1.0, 0.0, 0.5, 0.7, -0.4],
    // 4: DARK & MELANCHOLIC — minor 3rd/6th favoured, major color avoided.
    // The 3rd/6th columns are kept only MILDLY asymmetric: m6 is the
    // inversion of M3 (and M6 of m3), so hard-opposed signs made the row
    // reward first-inversion MAJOR triads (3,5,8 pairwise) and punish
    // first-inversion minor ones — backwards. Pairwise cells can't separate
    // major from minor in root position anyway (identical {3,4,7} multiset);
    // the root-relative quality term (chord_quality_weight) is what steers
    // mode, and these cells only color voicings.
    [1.0, -0.5, -0.1, 1.0, -0.2, 0.3, -0.2, 0.8, 0.6, 0.2, 0.5, -0.6],
    // 5: BRIGHT & LYDIAN — major 3rd/6th and the #4 tritone; same mild
    // 3rd/6th asymmetry as row 4, mirrored, for the same reason.
    [1.0, -0.7, 0.5, -0.1, 1.0, -0.2, 0.8, 0.9, 0.2, 0.6, -0.3, 0.6],
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
// style preference is `config.roughness_weight` (default in model.rs):
// 0 = pure style/pitch-class; 1 = pure psychoacoustic roughness.
//
// Chord-level aggregation weights (Layer 2): overall mean, worst single clash,
// the interval against the bass (inversion sensitivity), and each pitch class
// measured from the CHORD ROOT (quality — see the root_quality note in
// ChordScorer). The active set is always renormalized to sum to exactly 1
// (see the `agg` construction in score_group_options), so neither a missing
// root nor a chord_quality_weight away from 1 rescales the harmony term as a
// whole — it only rebalances the mix.
const AGG_MEAN: f64 = 0.30;
const AGG_WORST: f64 = 0.30;
const AGG_BASS: f64 = 0.20;
const AGG_ROOT: f64 = 0.20;

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
// Scale of the quadratic voice-contour spring: one octave from the anchor costs the
// full weight, and the penalty keeps growing (unclamped) beyond that so drift is
// hard-capped. Small offsets stay nearly free ((3/12)² ≈ 0.06) so the voice can
// oscillate around the anchor without being pinned to it.
const CONTOUR_PITCH_SPAN: f64 = 12.0;

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

/// Plomp–Levelt roughness between two *sine* partials (Sethares' parameters).
/// Peaks about a quarter of a critical band apart and decays to 0 at unison and
/// at wide spacing.
#[inline]
fn partial_roughness(f_low: f64, f_high: f64) -> f64 {
    let s = 0.24 / (0.0207 * f_low + 18.96);
    let x = s * (f_high - f_low);
    (-3.5 * x).exp() - (-5.75 * x).exp()
}

/// Harmonic partials per note in the sensory-dissonance model (amplitude 1/n).
/// A fundamental-only model is monotone in interval width, which made it rank a
/// major third as rougher than a tritone and gave a minor 9th ≈ 0 roughness —
/// it was a "narrow spacing" penalty, not sensory dissonance. Summing over
/// partials restores the classic curve: coinciding partials make P8/P5 smooth,
/// near-misses make m2/M2/m9 rough, and the whole surface still scales with
/// register.
const N_PARTIALS: usize = 6;
/// Divisor mapping the raw partial sum onto ~[0, 1]: the largest value the sum
/// reaches anywhere in the MIDI range (a major 2nd near A0). Normalizing by the
/// global maximum rather than a mid-register reference keeps the low bass —
/// where every interval is genuinely rougher — from saturating at 1.0 and
/// losing all discrimination exactly where spacing matters most.
const ROUGHNESS_NORM: f64 = 0.92;

/// Sensory dissonance between two notes with harmonic spectra, normalized to
/// roughly [0, 1]. Register-aware: the same interval class is rougher low than
/// high, and a literal minor 2nd is much rougher than a minor 9th (which is in
/// turn clearly rougher than zero, unlike in the fundamental-only model).
///
/// At C4 this ranks m2 .46 > M2 .35 > m3 .26 > M3 .20 > M7 .18 > TT .17 >
/// m6 .15 > P4 .13 > m7 .13 > M6 .11 > P5 .06 > P8 .003. The tritone landing
/// mid-pack is not a bug: its bite is functional rather than sensory, and the
/// style rows of HARMONY_MATRIX are where that character is expressed.
fn pair_roughness_uncached(p1: i32, p2: i32) -> f64 {
    if p1 == p2 { return 0.0; }
    let f_low = midi_to_hz(p1.min(p2));
    let f_high = midi_to_hz(p1.max(p2));
    let mut total = 0.0;
    for i in 1..=N_PARTIALS {
        let a1 = 1.0 / i as f64;
        let fi = f_low * i as f64;
        for j in 1..=N_PARTIALS {
            let a2 = 1.0 / j as f64;
            let fj = f_high * j as f64;
            let (lo, hi) = if fi <= fj { (fi, fj) } else { (fj, fi) };
            total += a1.min(a2) * partial_roughness(lo, hi);
        }
    }
    (total / ROUGHNESS_NORM).clamp(0.0, 1.0)
}

/// Roughness for every MIDI pair, built once. 36 exponential pairs per lookup is
/// too much for the chord-scoring hot path, but the function only depends on the
/// two pitches, so the whole 128×128 surface is precomputed on first use.
fn roughness_table() -> &'static [f64; 128 * 128] {
    static TABLE: std::sync::OnceLock<Box<[f64; 128 * 128]>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = Box::new([0.0f64; 128 * 128]);
        for a in 0..128i32 {
            for b in a..128i32 {
                let r = pair_roughness_uncached(a, b);
                t[(a * 128 + b) as usize] = r;
                t[(b * 128 + a) as usize] = r;
            }
        }
        t
    })
}

fn pair_roughness(p1: i32, p2: i32) -> f64 {
    if (0..128).contains(&p1) && (0..128).contains(&p2) {
        roughness_table()[(p1 * 128 + p2) as usize]
    } else {
        pair_roughness_uncached(p1, p2)
    }
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

/// Floor for the bass's idiomatic leaps: between the small-step scores
/// (0.86 / 0.71) and a third (0.57) — a comfortable option, not a free one,
/// so the bass still prefers a step when one is harmonically available.
const BASS_LEAP_SCORE: f64 = 0.5;

/// Channel-aware smoothness: `get_distance_score`, except that the bass
/// (channel 4) has its idiomatic leaps — P4, P5, octave — floored at
/// `BASS_LEAP_SCORE`. Root movement by 4th/5th IS the bass line's job in a
/// progression; the raw ramp scored a cadential 5th leap at 0.0 and an octave
/// at −0.71, so the bass was pressured to creep stepwise through changes the
/// upper voices were free to sing over. Other channels keep the strict ramp.
fn melodic_distance_score(prev_note: i32, current_note: i32, channel: i32) -> f64 {
    let s = get_distance_score(prev_note, current_note);
    if channel == 4 && matches!((prev_note - current_note).abs(), 5 | 7 | 12) {
        s.max(BASS_LEAP_SCORE)
    } else {
        s
    }
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

/// Tendency-tone pressure (`config.tendency_weight`): the two resolutions that
/// carry most of tonal harmony's forward motion, applied per voice as a function
/// of (previous pitch → candidate).
///
/// * **Leading tone → tonic.** A previous pitch a semitone below the KEY tonic
///   is rewarded for stepping up onto it and penalized for leaping away.
///   Because the test is literally "one semitone below the tonic", it simply
///   never fires in modes that have no leading tone (Aeolian, Dorian, …), which
///   is the musically correct behaviour rather than a special case.
/// * **Chordal 7th → down by step.** A previous pitch a minor 7th above the
///   root of the chord it belonged to resolves downward. Major 7ths are
///   excluded: as a chord tone the M7 tends *up*, and when it is the key's
///   leading tone the rule above already covers it.
///
/// `prev_root` is the root of the chord `prev` was part of (the previous bar at
/// a bar line), not the chord being built now.
fn tendency_term(prev: i32, c: i32, tonic: i32, prev_root: Option<i32>, weight: f64) -> f64 {
    if weight == 0.0 {
        return 0.0;
    }
    let mut t = 0.0;
    let step = c - prev;
    if prev.rem_euclid(12) == (tonic + 11).rem_euclid(12) {
        if step == 1 {
            t += 1.0;
        } else if step.abs() > 2 {
            t -= 0.5;
        }
    }
    if let Some(r) = prev_root {
        if (prev - r).rem_euclid(12) == 10 {
            if step == -1 || step == -2 {
                t += 1.0;
            } else if step > 0 {
                t -= 0.5;
            }
        }
    }
    t * weight
}

/// Pitch class of the chord root for the bar containing `start`.
///
/// The Schillinger layer builds each bar's chord as scale degrees
/// `(itm * ex) + seq[bar]`, so the entry generated from `itm = 0` — the first
/// one, for every built-in chord structure — IS the root. Voice 0's chord is
/// used as the reference; `ex` and the chord-structure contour vary the stack
/// per voice but not its root. `None` when there is no Schillinger data (the
/// simple candidate path), which disables every root-aware term.
fn bar_root_pc(state: &HarmonizerState, start: f64) -> Option<i32> {
    let bars = state.schillinger_notes.first()?;
    if bars.is_empty() {
        return None;
    }
    let bar = (start / 4.0).floor() as i32;
    bars[mod_shim(bar, bars.len() as i32) as usize]
        .first()
        .map(|&p| p.rem_euclid(12))
}

#[derive(Clone, Debug)]
pub struct Boundaries {
    pub min: i32,
    pub max: i32,
}

pub struct HarmonizerState {
    pub schillinger_notes: Vec<Vec<Vec<i32>>>,
    pub contours: Contours,
    pub harmony_matrix: Option<Vec<Vec<f64>>>,
}

impl HarmonizerState {
    /// Assemble from a config plus already-resolved contours. Runs the
    /// Schillinger progression, so call it at the same point in the seeded RNG
    /// stream the progression was always drawn at (after voice generation).
    pub fn new(config: &Config, contours: Contours) -> Self {
        HarmonizerState {
            schillinger_notes: crate::schillinger::gen_schillinger_progression(config, &contours),
            contours,
            harmony_matrix: config.harmony_matrix.clone(),
        }
    }

    pub fn from_config(config: &Config) -> Self {
        Self::new(config, Contours::from_config(config))
    }
}

/// Effective melody-force weight for voice `channel_idx` at beat position
/// `start`: that voice's melody-force contour when it has one, otherwise the
/// scalar `config.melody_force`.
fn melody_force_at(state: &HarmonizerState, config: &Config, channel_idx: usize, start: f64) -> f64 {
    state.contours.melody_force.as_ref()
        .and_then(|vc| vc.at(channel_idx, start))
        .unwrap_or(config.melody_force)
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

/// Rotate a 12-bit pitch-class mask up by `k` semitones.
fn rotate_pc_mask(mask: u16, k: i32) -> u16 {
    let k = k.rem_euclid(12) as u32;
    ((mask << k) | (mask >> (12 - k))) & 0x0FFF
}

/// One chord structure in all 12 transpositions, as pitch-class bitmasks
/// (`1 << pc`), sorted and deduped for binary search. `None` if the template names
/// no pitch classes — a stray `[]` must not silently forbid every chord.
///
/// Transposing here is what makes the constraint root-agnostic: a single `[0,4,7]`
/// admits a major triad on any of the 12 roots, so the caller never has to know
/// which chord the bar is on.
fn template_masks(pcs: &[i32]) -> Option<Vec<u16>> {
    let base = pcs.iter().fold(0u16, |m, &pc| m | 1 << pc.rem_euclid(12));
    if base == 0 {
        return None;
    }
    let mut masks: Vec<u16> = (0..12).map(|k| rotate_pc_mask(base, k)).collect();
    masks.sort_unstable();
    masks.dedup();
    Some(masks)
}

/// Union of every usable template's masks: the set of sonorities allowed when no
/// per-group assignment is in force. `None` means nothing is constrained.
fn chord_template_masks(templates: &[ChordTemplate]) -> Option<Vec<u16>> {
    let mut masks: Vec<u16> = templates
        .iter()
        .filter_map(|t| template_masks(t.pcs()))
        .flatten()
        .collect();
    if masks.is_empty() {
        return None;
    }
    masks.sort_unstable();
    masks.dedup();
    Some(masks)
}

/// Apportion `groups` slots across `weights` by the Sainte-Laguë highest-averages
/// method: each slot goes to whichever weight is currently most under-served.
///
/// Chosen over a random draw because it is exact and well spread. Over four groups
/// a 0.75/0.25 split lands 3-and-1 rather than "probably about three", and the
/// minority structure is distributed through the render instead of clumping. It
/// also touches no RNG, so adding weights cannot shift any other seeded draw.
fn apportion(weights: &[f64], groups: usize) -> Vec<usize> {
    let mut assigned = vec![0usize; weights.len()];
    let mut seq = Vec::with_capacity(groups);
    for _ in 0..groups {
        let mut best: Option<(usize, f64)> = None;
        for (i, &w) in weights.iter().enumerate() {
            if w <= 0.0 {
                continue;
            }
            let q = w / (2 * assigned[i] + 1) as f64;
            // Strict `>` makes ties fall to the earlier template, so the sequence
            // is a pure function of the weights — no ordering nondeterminism.
            if best.is_none_or(|(_, bq)| q > bq) {
                best = Some((i, q));
            }
        }
        match best {
            Some((i, _)) => {
                assigned[i] += 1;
                seq.push(i);
            }
            // Every weight is zero: nothing to apportion.
            None => break,
        }
    }
    seq
}

/// Which chord structures each rhythmic group may build.
///
/// Weights are apportioned across groups rather than folded into the soft score.
/// A constant score bonus would simply make the heaviest template win every chord,
/// so "major 0.75 / minor 0.25" would render as 100% major; assigning per group is
/// what makes the ratio mean what it says.
struct ChordPlan {
    /// Masks per usable template, in `config.chord_templates` order.
    per_template: Vec<Vec<u16>>,
    /// Union of `per_template` — what a group may use when unassigned.
    any: Vec<u16>,
    /// Template index per group. Empty when no entry states a weight, in which
    /// case every group may use any listed structure.
    assignment: Vec<usize>,
}

impl ChordPlan {
    /// `None` when `templates` constrains nothing.
    fn build(templates: &[ChordTemplate], groups: usize) -> Option<ChordPlan> {
        let mut per_template = Vec::new();
        let mut weights = Vec::new();
        for t in templates {
            // Unusable entries are dropped here so they cannot take an
            // apportionment slot that no chord could ever satisfy.
            if let Some(masks) = template_masks(t.pcs()) {
                per_template.push(masks);
                weights.push(t.weight());
            }
        }
        if per_template.is_empty() {
            return None;
        }

        let mut any: Vec<u16> = per_template.iter().flatten().copied().collect();
        any.sort_unstable();
        any.dedup();

        // Bare entries mean "no stated preference", so a list without any weight
        // keeps the whole whitelist legal in every group — the unweighted
        // behaviour. Stating even one weight switches on apportionment.
        let assignment = if templates.iter().any(|t| t.explicit_weight().is_some()) {
            apportion(&weights, groups)
        } else {
            Vec::new()
        };

        Some(ChordPlan { per_template, any, assignment })
    }

    /// The masks the group at `idx` must satisfy.
    fn masks_for(&self, idx: usize) -> &[u16] {
        match self.assignment.get(idx) {
            Some(&t) => &self.per_template[t],
            None => &self.any,
        }
    }
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

pub fn gen_voice(base: i32, rhythm_data: &Vec<f64>, pitch_shifts: &[i32], channel: i32, muted: i32, config: &Config, contours: &Contours) -> Vec<Note> {
    let mut ar = Vec::new();
    let clip_len = (config.pl * 4 * config.render_length) as f64;
    let mut pos = 0.0;
    let mut counter = 0;
    let sf = (SeededRng::random_int(60) + 1) as f64;

    while pos < clip_len {
        let n = base + pitch_shifts[mod_shim(counter, pitch_shifts.len() as i32) as usize];
        // Duration from this voice's rhythm contour when it has one; the flat
        // voice_rhythm cycle otherwise.
        let mut d = contours.voice_rhythm.as_ref()
            .filter(|_| channel >= 0)
            .and_then(|vc| vc.at_strict(channel as usize, pos))
            .unwrap_or_else(|| rhythm_data[mod_shim(counter, rhythm_data.len() as i32) as usize]);

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

/// One partial chord in the bass-first within-chord beam.
struct PartialChord {
    chosen: Vec<usize>,
    hard: u32,
    soft: f64,
    /// Pitch classes used so far, including sustaining notes — the input to the
    /// chord-template feasibility prune.
    pc_mask: u16,
    /// `pc_mask` is already a subset of no whitelisted chord, so no completion of
    /// this state can satisfy `config.chord_templates`.
    dead: bool,
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

/// The named components of a candidate's `soft_base`, kept alongside the folded
/// sum so a traced re-score can attribute the score without recomputing.
/// Values are as-added (weights applied, penalties negative).
#[derive(Clone, Copy, Default)]
struct CandTerms {
    smoothness: f64,
    melody_force: f64,
    tendency: f64,
    off_scale_hold: f64,
    contour_spring: f64,
    crossing_penalty: f64,
}

struct JointCand {
    pitch: i32,
    /// Melodic terms independent of the leader role and of other voices' NEW
    /// pitches: smoothness, contour pull, crossing. Always equals the sum of
    /// `terms` (folded in the same order, so bit-identical to the legacy value).
    soft_base: f64,
    /// Repeat/history terms if this voice is the leader (hold penalties).
    lead_term: f64,
    /// Repeat/history terms if it is not (hold bonus / stickiness).
    nonlead_term: f64,
    terms: CandTerms,
}

/// Callback surface for a traced chord evaluation. The search path uses
/// `NoTrace`, whose methods are empty and inline away — `eval` stays exactly
/// the hot loop it was; the explain pass passes a `TraceCollector`.
trait ScoreSink {
    fn term(&mut self, name: &'static str, value: f64);
    fn voice_term(&mut self, voice: usize, name: &'static str, value: f64);
    fn hard(&mut self, name: &'static str, count: u32);
    fn leader(&mut self, voice: usize);
}

struct NoTrace;
impl ScoreSink for NoTrace {
    #[inline(always)]
    fn term(&mut self, _: &'static str, _: f64) {}
    #[inline(always)]
    fn voice_term(&mut self, _: usize, _: &'static str, _: f64) {}
    #[inline(always)]
    fn hard(&mut self, _: &'static str, _: u32) {}
    #[inline(always)]
    fn leader(&mut self, _: usize) {}
}

impl ScoreSink for TraceCollector {
    fn term(&mut self, name: &'static str, value: f64) {
        TraceCollector::term(self, name, value)
    }
    fn voice_term(&mut self, voice: usize, name: &'static str, value: f64) {
        TraceCollector::voice_term(self, voice, name, value)
    }
    fn hard(&mut self, name: &'static str, count: u32) {
        TraceCollector::hard(self, name, count)
    }
    fn leader(&mut self, voice: usize) {
        self.leader = Some(voice);
    }
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

    // Tendency tones are keyed to the tonic (`config.root` is scale degree 1 —
    // see generate_mode_from_steps) and to the root of the chord the voice's
    // PREVIOUS pitch belonged to: at a bar line that is the previous bar, so
    // step back a hair from this group's start before looking the root up.
    let tonic = config.root.rem_euclid(12);
    let prev_root = if config.schillinger_progression && config.tendency_weight != 0.0 {
        let group_start = group.first().map(|n| n.start).unwrap_or(0.0);
        bar_root_pc(state, (group_start - 0.001).max(0.0))
    } else {
        None
    };

    // Chromatic-mode scale constraint as a pitch-class bitmask: entries are
    // offsets from `root`, so [0,2,4,5,7,9,11] is the major scale on whatever
    // the root is. 0 = no constraint (the window stays truly chromatic).
    let chromatic_mask: u16 = if config.schillinger_progression {
        0
    } else {
        config.chromatic_scale.iter()
            .fold(0u16, |m, &pc| m | 1 << (pc + config.root).rem_euclid(12))
    };

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
                    terms: CandTerms::default(),
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
        } else if chromatic_mask != 0 {
            ((last_note - range).max(PITCH_MIN)..=(last_note + range).min(PITCH_MAX))
                .filter(|p| chromatic_mask & 1 << p.rem_euclid(12) != 0)
                .collect()
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

        // A voice-contour that exists switches the spring's anchor to
        // "input pitch + sampled offset"; a missing or empty row means offset 0,
        // NOT "no contour" (that distinction only exists when the whole family
        // is absent).
        let use_contour = state.contours.voice.is_some();
        let target_offset: i32 = state.contours.voice.as_ref()
            .and_then(|vc| vc.at(channel_idx, note.start))
            .map(|v| v as i32)
            .unwrap_or(0);

        let default_bounds = Boundaries { min: DEFAULT_BOUND_MIN, max: DEFAULT_BOUND_MAX };
        let bounds = precomputed.boundaries_by_channel
            .get(channel_idx)
            .unwrap_or(&default_bounds);
        let cb_max = *CROSSING_BUFFER_UPPER.get_wrapped(channel_idx);
        let cb_min = *CROSSING_BUFFER_LOWER.get_wrapped(channel_idx);

        let eff_melody_force = melody_force_at(state, config, channel_idx, note.start);
        let cands = candidates.into_iter().map(|c| {
            // Each named component is computed once and folded into soft_base in
            // the same order as always, so the sum stays bit-identical.
            let mut terms = CandTerms {
                smoothness: melodic_distance_score(last_note, c, note.channel) * w_smooth,
                melody_force: melody_force_term(c, &current_lasts, eff_melody_force),
                tendency: tendency_term(last_note, c, tonic, prev_root, config.tendency_weight),
                ..CandTerms::default()
            };
            let mut soft_base = terms.smoothness;
            soft_base += terms.melody_force;
            soft_base += terms.tendency;
            if c == last_note && off_scale_hold {
                soft_base -= OFF_SCALE_HOLD_PENALTY;
                terms.off_scale_hold = -OFF_SCALE_HOLD_PENALTY;
            }

            if config.voice_contour_weight != 0.0 {
                let base_dist = if use_contour {
                    (c - (note.pitch + target_offset)).abs()
                } else {
                    (c - note.pitch).abs()
                };
                let normalized = base_dist as f64 / CONTOUR_PITCH_SPAN;
                soft_base -= config.voice_contour_weight * normalized * normalized;
                terms.contour_spring = -(config.voice_contour_weight * normalized * normalized);
            }

            if bounds.max - c < cb_max {
                soft_base -= config.no_crossing;
                terms.crossing_penalty -= config.no_crossing;
            }
            if c - bounds.min < cb_min {
                soft_base -= config.no_crossing;
                terms.crossing_penalty -= config.no_crossing;
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

            JointCand { pitch: c, soft_base, lead_term, nonlead_term, terms }
        }).collect();

        JointVoice { note: *note, prev, is_fixed_lead: false, cands }
    }).collect()
}

// The outer beam's width lives in `config.beam_width` (see model.rs) — it doubles
// as the per-group branching factor, so it has to be tunable against runtime.

/// Group-level harmony/smoothness weights and the consonance-matrix context for a
/// group, resolved from the optional contours or the scalar config fall-backs.
/// Returns `(w_harmony, w_smooth, harmony_ctx)`.
fn group_weights(group_start: f64, config: &Config, state: &HarmonizerState) -> (f64, f64, f64) {
    // The balance is a MIX between harmony and smoothness, so it is clamped
    // to the range where both weights stay non-negative: a contour value
    // beyond ±0.5 used to flip a weight's sign, silently REWARDING leaps
    // (or dissonance) instead of merely ignoring the other term.
    let r = state.contours.harmony_distance.as_ref()
        .map(|c| c.at(group_start))
        .unwrap_or(config.harmony_distance_balance)
        .clamp(-0.5, 0.5);
    let harmony_ctx = state.contours.harmony_matrix.as_ref()
        .map(|c| c.at(group_start))
        .unwrap_or(0.0);
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
    fn build(
        voices: &[JointVoice],
        precomputed: &PrecomputedHarmonyData,
        row: &HarmonyRow,
        roughness_weight: f64,
    ) -> Self {
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
                soft[a * m + b] = (1.0 - roughness_weight) * row.soft[ic]
                    - roughness_weight * pair_roughness(pitches[a], pitches[b]);
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

/// A scored chord: the per-voice candidate indices plus its `(hard, soft)` score.
type ScoredChord = (Vec<usize>, (u32, f64));

/// Total order over scored chords, best first — `chord_better`, with the
/// candidate-index vector as a final tiebreak so an exact score tie always
/// resolves the same way regardless of the order rayon happens to merge in.
fn chord_cmp(a: &ScoredChord, b: &ScoredChord) -> std::cmp::Ordering {
    a.1.0.cmp(&b.1.0)
        .then_with(|| b.1.1.partial_cmp(&a.1.1).unwrap_or(std::cmp::Ordering::Equal))
        .then_with(|| a.0.cmp(&b.0))
}

/// True if `sc` could still make a top-`k` list currently holding `list`.
/// Score-only (no index tiebreak) so the enumeration hot loop can skip cloning
/// the index vector for the overwhelming majority of chords; ties are admitted
/// and settled by `push_topk`.
fn topk_admits(list: &[ScoredChord], sc: (u32, f64), k: usize) -> bool {
    list.len() < k || (k > 0 && !chord_better(list[k - 1].1, sc))
}

/// Insert into a list kept sorted best-first by `chord_cmp`, capped at `k`.
fn push_topk(list: &mut Vec<ScoredChord>, item: ScoredChord, k: usize) {
    if k == 0 {
        return;
    }
    let pos = list.partition_point(|e| chord_cmp(e, &item) == std::cmp::Ordering::Less);
    if pos >= k {
        return;
    }
    list.insert(pos, item);
    list.truncate(k);
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

/// True if the pair move (qi→pi against qj→pj) commits a forbidden parallel:
/// perfect 5ths/octaves reached in SIMILAR motion, or octaves reached in
/// CONTRARY motion (antiparallel octaves — octave collapsing to unison or
/// opening to a double octave still exposes the doubled pitch class).
/// Oblique motion (either voice holding) is never a violation, and
/// contrary-motion fifths are tolerated, as is usual in a multi-voice texture.
fn parallel_motion_violation(qi: i32, pi: i32, qj: i32, pj: i32) -> bool {
    let mi = (pi - qi).signum();
    let mj = (pj - qj).signum();
    if mi == 0 || mj == 0 {
        return false;
    }
    let prev_int = (qi - qj).abs() % 12;
    let cur_int = (pi - pj).abs() % 12;
    if mi == mj {
        prev_int == cur_int && (cur_int == 0 || cur_int == 7)
    } else {
        prev_int == 0 && cur_int == 0
    }
}

/// Doubling balance for the root-aware doubling term, in units of
/// `config.root_doubling_weight`: +1 for the first root doubling (piling more
/// voices on is the interval-variety term's business), −1 per extra copy of
/// the leading tone (its resolution would force parallel octaves), and −1 per
/// extra copy of the chordal minor 7th — the other classic error: both copies
/// must fall by step, so one either breaks its resolution or they fall in
/// octaves.
fn doubling_balance(pc_count: &[u8; 12], root: i32, leading_tone_pc: i32) -> i32 {
    let root_dup = (pc_count[root as usize] as i32 - 1).clamp(0, 1);
    let lt_dup = (pc_count[leading_tone_pc as usize] as i32 - 1).max(0);
    let seventh_pc = (root + 10).rem_euclid(12) as usize;
    let seventh_dup = (pc_count[seventh_pc] as i32 - 1).max(0);
    root_dup - lt_dup - seventh_dup
}

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
    /// Aggregation weights `(mean, worst, bass, root)` — already renormalized
    /// and already scaled by `config.chord_quality_weight`.
    agg: (f64, f64, f64, f64),
    /// Style preference for each pitch class measured from the bar's CHORD ROOT:
    /// `root_quality[pc] = row.soft[(pc - root) mod 12]`.
    ///
    /// The pairwise multiset cannot express chord quality — a major triad
    /// {0,4,7} and a minor triad {0,3,7} both yield interval classes {3,4,7}, so
    /// a style row that rewards the minor 3rd and punishes the major one steers
    /// neither toward minor nor away from major. Intervals measured from the
    /// root do carry quality (4+7 vs 3+7), and reusing the same style row means
    /// the H-Matrix now controls chord colour as well as interval colour.
    /// `None` on the non-Schillinger path, where no root is defined.
    root_quality: Option<[f64; 12]>,
    root_pc: Option<i32>,
    /// Pitch class a semitone below the key tonic — the doubling to avoid.
    leading_tone_pc: i32,
    /// Allowed pitch-class-set bitmasks from `config.chord_templates`, already
    /// expanded over all 12 transpositions. `None` = chord structure is free.
    chord_masks: Option<&'a [u16]>,
    config: &'a Config,
}

impl ChordScorer<'_> {
    /// `(hard violations, soft score)` for one complete chord — the search
    /// hot path, evaluated untraced.
    #[inline]
    fn eval(&self, chosen: &[usize]) -> (u32, f64) {
        self.eval_with(chosen, &mut NoTrace)
    }

    /// `eval` with a sink receiving every named contribution. `NoTrace` inlines
    /// to the original hot loop; `TraceCollector` records the breakdown. The
    /// returned numbers are identical either way.
    fn eval_with<S: ScoreSink>(&self, chosen: &[usize], sink: &mut S) -> (u32, f64) {
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
        let mut leader_idx = usize::MAX;
        for (i, (v, &ci)) in voices.iter().zip(chosen).enumerate() {
            let c = &v.cands[ci];
            soft += c.soft_base;
            sink.voice_term(i, "smoothness", c.terms.smoothness);
            sink.voice_term(i, "melody_force", c.terms.melody_force);
            sink.voice_term(i, "tendency", c.terms.tendency);
            sink.voice_term(i, "off_scale_hold", c.terms.off_scale_hold);
            sink.voice_term(i, "contour_spring", c.terms.contour_spring);
            sink.voice_term(i, "crossing_penalty", c.terms.crossing_penalty);
            if !v.is_fixed_lead {
                nonlead_sum += c.nonlead_term;
                let delta = c.lead_term - c.nonlead_term;
                if delta > best_delta {
                    best_delta = delta;
                    leader_idx = i;
                }
            }
        }
        soft += nonlead_sum;
        if best_delta > f64::NEG_INFINITY {
            soft += best_delta;
            sink.leader(leader_idx);
            for (i, (v, &ci)) in voices.iter().zip(chosen).enumerate() {
                if v.is_fixed_lead {
                    continue;
                }
                let c = &v.cands[ci];
                if i == leader_idx {
                    sink.voice_term(i, "leader_history", c.lead_term);
                } else {
                    sink.voice_term(i, "hold_stickiness", c.nonlead_term);
                }
            }
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
                    sink.hard("unison_collision", 1);
                    continue;
                }
                let k = ii * m + t.idx_of[&pj];
                if t.forbidden[k] {
                    hard += 1;
                    sink.hard("forbidden_interval", 1);
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
                    sink.hard("unison_collision", 1);
                    continue;
                }
                let k = ii * m + sj;
                if t.forbidden[k] {
                    hard += 1;
                    sink.hard("forbidden_interval", 1);
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
        // Pitch-class census over the whole sonority, for the root-relative
        // terms below (quality, inversion, doubling).
        let mut pc_count = [0u8; 12];
        for (v, &ci) in voices.iter().zip(chosen) {
            pc_count[v.cands[ci].pitch.rem_euclid(12) as usize] += 1;
        }
        for &s in self.sustaining_notes {
            pc_count[s.rem_euclid(12) as usize] += 1;
        }

        // Chord-structure whitelist. Counted as a hard violation so it composes
        // lexicographically with the forbidden-interval and voice-budget checks
        // instead of competing with the soft score.
        if let Some(masks) = self.chord_masks {
            let mut mask = 0u16;
            for (pc, &cnt_pc) in pc_count.iter().enumerate() {
                if cnt_pc > 0 {
                    mask |= 1 << pc;
                }
            }
            if masks.binary_search(&mask).is_err() {
                hard += 1;
                sink.hard("chord_template", 1);
            }
        }

        if cnt > 0 {
            let mean = sum / cnt as f64;
            let bass_term = if bass_cnt > 0 { bass_sum / bass_cnt as f64 } else { mean };
            let (wm, ww, wb, wr) = self.agg;
            let mut agg = wm * mean + ww * worst + wb * bass_term;
            sink.term("harmony_mean", self.w_harmony * wm * mean);
            sink.term("harmony_worst", self.w_harmony * ww * worst);
            sink.term("harmony_bass", self.w_harmony * wb * bass_term);
            if let Some(rq) = self.root_quality {
                // Mean over DISTINCT pitch classes: quality is about which
                // degrees are present, not how often they are doubled (that is
                // the doubling term's job).
                let mut q = 0.0;
                let mut qn = 0usize;
                for (pc, &n) in pc_count.iter().enumerate() {
                    if n > 0 {
                        q += rq[pc];
                        qn += 1;
                    }
                }
                if qn > 0 {
                    agg += wr * (q / qn as f64);
                    sink.term("harmony_quality", self.w_harmony * wr * (q / qn as f64));
                }
            }
            soft += self.w_harmony * agg;
        }

        // Inversion and doubling, both meaningless without a root.
        if let Some(root) = self.root_pc {
            if self.config.root_position_weight != 0.0 {
                // Root position is the stable one; a third in the bass is a
                // usable soft inversion; a fifth in the bass (six-four) is
                // unstable and needs preparation this system cannot express;
                // a non-chord tone in the bass blurs the harmony outright.
                let iv = (bass.rem_euclid(12) - root).rem_euclid(12);
                let bass_bonus = match iv {
                    0 => 1.0,
                    3 | 4 => 0.4,
                    7 => 0.0,
                    _ => -0.3,
                };
                soft += self.config.root_position_weight * bass_bonus;
                sink.term("root_position", self.config.root_position_weight * bass_bonus);
            }
            if self.config.root_doubling_weight != 0.0 {
                // Doubling the root is the default in part-writing; doubling
                // the leading tone or the chordal 7th forces a broken
                // resolution or parallel octaves — see doubling_balance.
                let balance = doubling_balance(&pc_count, root, self.leading_tone_pc) as f64;
                soft += self.config.root_doubling_weight * balance;
                sink.term("root_doubling", self.config.root_doubling_weight * balance);
            }
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
            sink.term("interval_variety", -(self.config.interval_exists_in_harmony * dups as f64));
        }

        // Parallel 5ths/octaves and antiparallel octaves among the new voices
        // (sustaining notes have zero motion at this instant) — see
        // parallel_motion_violation.
        if self.config.consecutive_octav_fift != 0.0 {
            for i in 0..n {
                let Some(qi) = voices[i].prev else { continue };
                let pi = voices[i].cands[chosen[i]].pitch;
                if pi == qi {
                    continue;
                }
                for j in (i + 1)..n {
                    let Some(qj) = voices[j].prev else { continue };
                    let pj = voices[j].cands[chosen[j]].pitch;
                    if parallel_motion_violation(qi, pi, qj, pj) {
                        soft -= self.config.consecutive_octav_fift;
                        sink.term("parallel_motion", -self.config.consecutive_octav_fift);
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
                    sink.term("same_direction", -self.config.same_direction);
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
            sink.hard("voice_budget", (movers - self.config.max_voices_changed) as u32);
        }
        if self.config.min_voices_changed >= 0 && movers < self.config.min_voices_changed {
            hard += (self.config.min_voices_changed - movers) as u32;
            sink.hard("voice_budget", (self.config.min_voices_changed - movers) as u32);
        }
        if self.config.common_tone_penalty != 0.0 {
            soft -= self.config.common_tone_penalty * common as f64;
            sink.term("common_tone_penalty", -(self.config.common_tone_penalty * common as f64));
        }

        (hard, soft)
    }

    /// Exhaustive search over the candidate product — parallel across the first
    /// voice's candidates, odometer over the rest. Optimal within the window.
    /// Returns the best `k` distinct chords, best first (empty if no candidates).
    fn enumerate_topk(&self, k: usize) -> Vec<ScoredChord> {
        let n = self.n;
        (0..self.voices[0].cands.len()).into_par_iter()
            .map(|c0| {
                let mut chosen = vec![0usize; n];
                chosen[0] = c0;
                let mut local: Vec<ScoredChord> = Vec::with_capacity(k);
                loop {
                    let sc = self.eval(&chosen);
                    if topk_admits(&local, sc, k) {
                        push_topk(&mut local, (chosen.clone(), sc), k);
                    }
                    // Advance the odometer over voices 1..n.
                    let mut pos = n - 1;
                    loop {
                        if pos == 0 {
                            return local;
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
            .reduce(Vec::new, |mut a, b| {
                for item in b {
                    push_topk(&mut a, item, k);
                }
                a
            })
    }

    /// True if `partial` can still grow into a whitelisted chord — i.e. it is a
    /// subset of at least one allowed mask. Vacuously true with no whitelist.
    ///
    /// The within-chord beam ranks partial states on pairwise score alone, which
    /// discards the partials leading to any sonority the interval matrix happens
    /// to dislike — augmented and diminished triads especially — long before
    /// `eval` can judge the finished chord. Carrying feasibility into the prune
    /// is what makes the whitelist actually bind on the beam path.
    fn template_feasible(&self, partial: u16) -> bool {
        match self.chord_masks {
            None => true,
            Some(masks) => masks.iter().any(|&m| partial & !m == 0),
        }
    }

    /// Bass-first within-chord beam for groups whose candidate product exceeds
    /// `JOINT_ENUM_CAP`. Partial states are ranked by running
    /// `(violations, partial soft)`; survivors get the exact chord score.
    /// Returns the best `k` of them, best first.
    fn beam_search_topk(&self, k: usize) -> Vec<ScoredChord> {
        let n = self.n;
        let t = self.table;
        let m = t.m;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(self.voices[i].note.channel));

        // Pitch classes already committed by sustaining notes — `eval` counts them
        // in the census, so the feasibility prune has to start from them too.
        let sust_mask = self.sustaining_notes.iter()
            .fold(0u16, |m, &p| m | 1 << p.rem_euclid(12));
        let mut states: Vec<PartialChord> = vec![PartialChord {
            chosen: Vec::new(),
            hard: 0,
            soft: 0.0,
            pc_mask: sust_mask,
            dead: false,
        }];
        for &vi in &order {
            let mut next: Vec<PartialChord> =
                Vec::with_capacity(states.len() * self.voices[vi].cands.len());
            for st in &states {
                let (part, hard, soft) = (&st.chosen, &st.hard, &st.soft);
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
                    let pc_mask = st.pc_mask | 1 << cand.pitch.rem_euclid(12);
                    next.push(PartialChord {
                        chosen: cp,
                        hard: h,
                        soft: s,
                        pc_mask,
                        dead: !self.template_feasible(pc_mask),
                    });
                }
            }
            // Template-infeasible states sort last so they are dropped first, but
            // are still kept when nothing feasible survives — a whitelist that
            // cannot be met degrades to "least bad", never to an empty beam.
            next.sort_by(|a, b| a.dead.cmp(&b.dead)
                .then(a.hard.cmp(&b.hard))
                .then(b.soft.partial_cmp(&a.soft).unwrap_or(std::cmp::Ordering::Equal)));
            next.truncate(JOINT_BEAM_WIDTH);
            states = next;
        }

        let mut best: Vec<ScoredChord> = Vec::with_capacity(k);
        for PartialChord { chosen: part, .. } in states {
            let mut chosen = vec![0usize; n];
            for (pos, &ci) in part.iter().enumerate() {
                chosen[order[pos]] = ci;
            }
            let sc = self.eval(&chosen);
            push_topk(&mut best, (chosen, sc), k);
        }
        best
    }
}

/// Root-aware scoring inputs for a group at `group_start`: the bar's chord-root
/// pitch class, the root-relative quality profile, and the renormalized
/// aggregation weights `(mean, worst, bass, root)`.
///
/// The bar's chord root turns the bare pitch-class set the Schillinger layer
/// hands over back into a rooted chord, which is what the quality, inversion
/// and doubling terms need.
///
/// The active weights are renormalized to sum to 1 in BOTH branches:
/// without a root the fourth slot has nothing to score, and with one a
/// chord_quality_weight away from 1 must rebalance the mix, not rescale
/// the whole harmony term (cqw = 0 used to shrink the rooted path to 0.8×
/// while the rootless path renormalized — quality off made harmony
/// WEAKER than having no root at all). Negative cqw is clamped: "prefer
/// bad quality" has no meaning here and would let the sum reach zero.
fn root_and_agg(
    group_start: f64,
    config: &Config,
    state: &HarmonizerState,
    row: &HarmonyRow,
) -> (Option<i32>, Option<[f64; 12]>, (f64, f64, f64, f64)) {
    let root_pc = if config.schillinger_progression {
        bar_root_pc(state, group_start)
    } else {
        None
    };
    let root_quality = root_pc.map(|r| {
        let mut q = [0.0f64; 12];
        for (pc, slot) in q.iter_mut().enumerate() {
            *slot = row.soft[(pc as i32 - r).rem_euclid(12) as usize];
        }
        q
    });
    let agg = if root_quality.is_some() {
        let wr = AGG_ROOT * config.chord_quality_weight.max(0.0);
        let k = 1.0 / (AGG_MEAN + AGG_WORST + AGG_BASS + wr);
        (AGG_MEAN * k, AGG_WORST * k, AGG_BASS * k, wr * k)
    } else {
        let k = 1.0 / (AGG_MEAN + AGG_WORST + AGG_BASS);
        (AGG_MEAN * k, AGG_WORST * k, AGG_BASS * k, 0.0)
    };
    (root_pc, root_quality, agg)
}

/// Score one rhythmic group: build each voice's candidate set, then search the
/// joint candidate space for the chords with the fewest hard violations and the
/// best soft score. Returns up to `k` complete voicings, best first — each the
/// chosen notes (one per input note) plus that voicing's soft score net of the
/// hard-violation penalty. The outer beam branches on these alternatives; `k = 1`
/// is the greedy "just give me the best chord" case.
///
/// Always returns at least one entry (a pass-through of the input group when the
/// candidate search comes up empty).
fn score_group_options(
    group: &[Note],
    config: &Config,
    state: &HarmonizerState,
    precomputed: &PrecomputedHarmonyData,
    k: usize,
    chord_masks: Option<&[u16]>,
) -> Vec<(Vec<Note>, f64)> {
    if group.is_empty() {
        return vec![(Vec::new(), 0.0)];
    }
    let k = k.max(1);

    let (w_harmony, w_smooth, harmony_ctx) = group_weights(group[0].start, config, state);
    let row = get_harmony_row(harmony_ctx, state.harmony_matrix.as_ref());

    let voices = build_joint_voices(group, w_smooth, config, state, precomputed);
    let n = voices.len();
    let table = PairTable::build(&voices, precomputed, &row, config.roughness_weight);

    let ending_by_channel: HashMap<i32, i32> = precomputed.notes_ending_at_start.iter()
        .map(|nn| (nn.channel, nn.pitch))
        .collect();
    let check_dir = !ending_by_channel.is_empty() && config.same_direction != 0.0;

    let (root_pc, root_quality, agg) = root_and_agg(group[0].start, config, state, &row);

    let scorer = ChordScorer {
        voices: &voices,
        n,
        table: &table,
        sustaining_notes: &precomputed.sustaining_notes,
        ending_by_channel: &ending_by_channel,
        check_dir,
        w_harmony,
        agg,
        root_quality,
        root_pc,
        chord_masks,
        leading_tone_pc: (config.root + 11).rem_euclid(12),
        config,
    };

    let product = voices.iter()
        .map(|v| v.cands.len())
        .try_fold(1usize, |acc, l| acc.checked_mul(l))
        .unwrap_or(usize::MAX);

    let ranked = if product <= JOINT_ENUM_CAP {
        scorer.enumerate_topk(k)
    } else {
        scorer.beam_search_topk(k)
    };

    if ranked.is_empty() {
        // No candidates at all — pass the group through unchanged.
        let notes = group.iter().map(|nn| {
            let mut note = *nn;
            note.muted = 0;
            note
        }).collect();
        return vec![(notes, 0.0)];
    }

    ranked.into_iter().map(|(chosen, (hard, soft))| {
        let notes = voices.iter().zip(&chosen).map(|(v, &ci)| {
            let mut note = v.note;
            note.pitch = v.cands[ci].pitch;
            note.muted = 0;
            note
        }).collect();
        (notes, soft - hard as f64 * VIOLATION_PENALTY)
    }).collect()
}

/// Single-best wrapper around `score_group_options`: appends the winning
/// voicing's notes to `temp_group_notes` and returns its score.
fn score_group(
    group: &[Note],
    temp_group_notes: &mut Vec<Note>,
    config: &Config,
    state: &HarmonizerState,
    precomputed: &PrecomputedHarmonyData,
    chord_masks: Option<&[u16]>,
) -> f64 {
    // `score_group_options` always yields at least one entry.
    let (notes, score) = score_group_options(group, config, state, precomputed, 1, chord_masks)
        .swap_remove(0);
    temp_group_notes.extend(notes);
    score
}

/// Greedy look-ahead: score the remaining groups one at a time (each conditioned
/// on the running context), memoized on a hash of `(group index, depth,
/// context)`. Returns the summed soft score over the look-ahead horizon.
///
/// The key must describe the state COMPLETELY. It used to hash only the last 10
/// notes, but the score depends on everything `build_precomputed_data` reads —
/// the whole trimmed window, including sustains and the 5-note-per-channel
/// history — so two beam branches that merely shared a 10-note suffix aliased
/// onto one cache entry and the winner was decided by which rayon worker got
/// there first. Same seed, different render. Hashing the full context (which is
/// already bounded by the caller's 32-beat trim) makes the cache sound and the
/// render reproducible.
fn score_lookahead(
    groups: &[Vec<Note>],
    start_idx: usize,
    depth: i32,
    context: &[Note],
    config: &Config,
    state: &HarmonizerState,
    cache: &DashMap<u64, f64>,
    plan: Option<&ChordPlan>,
) -> f64 {
    if depth == 0 || start_idx >= groups.len() {
        return 0.0;
    }

    let mut hasher = FxHasher64::default();
    start_idx.hash(&mut hasher);
    depth.hash(&mut hasher);
    for n in context {
        n.pitch.hash(&mut hasher);
        n.channel.hash(&mut hasher);
        n.start.to_bits().hash(&mut hasher);
        n.duration.to_bits().hash(&mut hasher);
    }
    let key = hasher.finish();

    if let Some(val) = cache.get(&key) {
        return *val;
    }

    let group = &groups[start_idx];
    let start_time = group[0].start;
    let precomputed = build_precomputed_data(context, group, start_time);

    let mut temp_notes = Vec::new();
    let local_score = score_group(
        group, &mut temp_notes, config, state, &precomputed, plan.map(|p| p.masks_for(start_idx)),
    );

    let mut next_context = context.to_vec();
    next_context.extend(temp_notes);

    let best_score = local_score
        + score_lookahead(groups, start_idx + 1, depth - 1, &next_context, config, state, cache, plan);

    cache.insert(key, best_score);
    best_score
}

use std::sync::mpsc::Sender;

fn score_group_beam(income: Vec<Note>, config: &Config, state: &HarmonizerState, progress_sender: Option<&Sender<(usize, usize)>>) -> Vec<Note> {
    // Each group's notes come back in channel order from group_by_start_array.
    let groups = group_by_start_array(income);
    // Built once for the whole render: the weight apportionment has to be a
    // property of the group index, not of whichever beam branch asks. Drawing it
    // per call would hand different branches different structures and make the
    // render depend on rayon's completion order.
    let plan = ChordPlan::build(&config.chord_templates, groups.len());
    let plan_ref = plan.as_ref();
    let lookahead = config.lookahead_depth;
    // One knob for both directions of the search: how many progressions survive
    // each group, and how many alternative voicings each one branches into. At
    // width 1 the whole thing collapses to the greedy walk (and the look-ahead,
    // having nothing to choose between, is skipped).
    let beam_width = config.beam_width.max(1) as usize;

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

                // Branch: this parent's best `beam_width` voicings of the group,
                // each scored on the horizon that follows it.
                let group_masks = plan_ref.map(|p| p.masks_for(i));
                score_group_options(group, config, state, &precomputed, beam_width, group_masks)
                    .into_iter()
                    .map(|(added_notes, group_score)| {
                        let actual_score = beam_state.actual_score + group_score;
                        // With a single option there is nothing to rank, so the
                        // look-ahead can only burn CPU.
                        let lookahead_score = if beam_width > 1 {
                            let mut next_context = trimmed_notes.to_vec();
                            next_context.extend(added_notes.iter().cloned());
                            score_lookahead(
                                groups_ref, i + 1, lookahead, &next_context, config, state,
                                cache_ref, plan_ref,
                            )
                        } else {
                            0.0
                        };
                        IntermediateCandidate {
                            parent_idx,
                            added_notes,
                            actual_score,
                            rank_score: actual_score + lookahead_score,
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect();

        // Best rank first; exact ties fall back to the parent's own running score
        // and then to beam position, so the survivor set never depends on the
        // order rayon happened to finish the branches in.
        candidates.sort_by(|a, b| {
            b.rank_score.partial_cmp(&a.rank_score).unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.actual_score.partial_cmp(&a.actual_score).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.parent_idx.cmp(&b.parent_idx))
                .then_with(|| a.added_notes.iter().map(|n| n.pitch).cmp(b.added_notes.iter().map(|n| n.pitch)))
        });

        beam = candidates.into_iter().take(beam_width).map(|c| {
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

/// A harmonized render plus the scoring story of every chosen chord.
pub struct Harmonised {
    pub notes: Vec<Note>,
    pub breakdown: Vec<GroupBreakdown>,
}

/// `harmonise2` plus the named score breakdown: after the beam settles on the
/// winning progression, each chosen chord is re-scored once with a collecting
/// sink. The numbers are exactly the ones the search saw — same context
/// trimming, same scorer — so the breakdown explains the render rather than
/// approximating it.
pub fn harmonise_explained(
    income: Vec<Note>,
    config: &Config,
    state: &HarmonizerState,
    progress_sender: Option<&Sender<(usize, usize)>>,
) -> Harmonised {
    let notes = score_group_beam(income.clone(), config, state, progress_sender);
    let breakdown = explain_render(&income, &notes, config, state);
    Harmonised { notes, breakdown }
}

/// Re-walk the winning progression group by group, re-scoring each chosen
/// chord with a `TraceCollector`. `income` is needed alongside the final notes
/// because candidate generation anchors on the INPUT pitches (contour springs,
/// fall-backs), not the harmonized ones.
fn explain_render(
    income: &[Note],
    final_notes: &[Note],
    config: &Config,
    state: &HarmonizerState,
) -> Vec<GroupBreakdown> {
    let groups = group_by_start_array(income.to_vec());
    let plan = ChordPlan::build(&config.chord_templates, groups.len());
    let mut out = Vec::with_capacity(groups.len());
    let mut consumed = 0usize;

    for (gi, group) in groups.iter().enumerate() {
        let Some(chosen_notes) = final_notes.get(consumed..consumed + group.len()) else {
            break; // defensive: the render did not produce one note per input note
        };
        let start_time = group[0].start;
        // The exact context the beam scored this group against on the winning
        // path: everything chosen before it, trimmed to the same 32-beat window.
        let prior = &final_notes[..consumed];
        consumed += group.len();
        let begin = prior.partition_point(|n| n.start < start_time - 32.0);
        let trimmed = &prior[begin..];

        let precomputed = build_precomputed_data(trimmed, group, start_time);
        let (w_harmony, w_smooth, harmony_ctx) = group_weights(start_time, config, state);
        let row = get_harmony_row(harmony_ctx, state.harmony_matrix.as_ref());
        let voices = build_joint_voices(group, w_smooth, config, state, &precomputed);
        let table = PairTable::build(&voices, &precomputed, &row, config.roughness_weight);
        let ending_by_channel: HashMap<i32, i32> = precomputed.notes_ending_at_start.iter()
            .map(|nn| (nn.channel, nn.pitch))
            .collect();
        let check_dir = !ending_by_channel.is_empty() && config.same_direction != 0.0;
        let (root_pc, root_quality, agg) = root_and_agg(start_time, config, state, &row);
        let scorer = ChordScorer {
            voices: &voices,
            n: voices.len(),
            table: &table,
            sustaining_notes: &precomputed.sustaining_notes,
            ending_by_channel: &ending_by_channel,
            check_dir,
            w_harmony,
            agg,
            root_quality,
            root_pc,
            chord_masks: plan.as_ref().map(|p| p.masks_for(gi)),
            leading_tone_pc: (config.root + 11).rem_euclid(12),
            config,
        };

        // Recover which candidate each voice's final pitch was. `None` only on
        // the pass-through path (no candidates were generated), which is
        // reported as an unscored group rather than guessed at.
        let chosen: Option<Vec<usize>> = voices.iter().zip(chosen_notes).map(|(v, nn)| {
            if v.note.channel != nn.channel {
                return None;
            }
            v.cands.iter().position(|c| c.pitch == nn.pitch)
        }).collect();

        let bar = (start_time / 4.0).floor() as i32;
        if let Some(chosen) = chosen {
            let mut collector = TraceCollector::with_voices(voices.len());
            let (hard, soft) = scorer.eval_with(&chosen, &mut collector);
            let voice_breakdowns = voices.iter().enumerate().map(|(i, v)| VoiceBreakdown {
                channel: v.note.channel,
                pitch: v.cands[chosen[i]].pitch,
                previous_pitch: v.prev,
                is_leader: collector.leader == Some(i),
                terms: std::mem::take(&mut collector.voice_terms[i]),
            }).collect();
            out.push(GroupBreakdown {
                start: start_time,
                bar,
                root_pc,
                score: soft - hard as f64 * VIOLATION_PENALTY,
                soft_score: soft,
                hard_violation_count: hard,
                hard_violations: collector.hard,
                chord_terms: collector.chord_terms,
                voices: voice_breakdowns,
                scored: true,
            });
        } else {
            out.push(GroupBreakdown {
                start: start_time,
                bar,
                root_pc,
                score: 0.0,
                soft_score: 0.0,
                hard_violation_count: 0,
                hard_violations: Vec::new(),
                chord_terms: Vec::new(),
                voices: chosen_notes.iter().map(|nn| VoiceBreakdown {
                    channel: nn.channel,
                    pitch: nn.pitch,
                    previous_pitch: None,
                    is_leader: false,
                    terms: Vec::new(),
                }).collect(),
                scored: false,
            });
        }
    }
    out
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
            contours: Contours::default(),
            harmony_matrix: None,
        }
    }

    /// A one-value contour at the tests' 4-beat resolution.
    fn flat(v: f64) -> Option<crate::contour::Contour> {
        crate::contour::Contour::new(vec![v], 4.0)
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

    // ----- melodic_distance_score (bass leap tolerance) -----

    #[test]
    fn bass_idiomatic_leaps_are_floored() {
        // P4, P5, octave in the bass: floored at BASS_LEAP_SCORE...
        for d in [5, 7, 12] {
            assert!(approx(melodic_distance_score(48, 48 - d, 4), BASS_LEAP_SCORE), "down {d}");
            assert!(approx(melodic_distance_score(48, 48 + d, 4), BASS_LEAP_SCORE), "up {d}");
        }
        // ...but still below a hold or a step, so steps stay preferred.
        assert!(BASS_LEAP_SCORE < melodic_distance_score(48, 50, 4));
    }

    #[test]
    fn bass_non_idiomatic_leaps_keep_the_ramp() {
        // A tritone or a 7th in the bass is not an idiomatic root move.
        assert!(approx(melodic_distance_score(48, 54, 4), get_distance_score(48, 54)));
        assert!(approx(melodic_distance_score(48, 58, 4), get_distance_score(48, 58)));
    }

    #[test]
    fn upper_voices_get_no_leap_tolerance() {
        for ch in [0, 1, 2, 3] {
            assert!(approx(melodic_distance_score(60, 67, ch), get_distance_score(60, 67)));
        }
    }

    // ----- parallel_motion_violation -----

    #[test]
    fn parallel_fifths_and_octaves_in_similar_motion_are_violations() {
        // C+G → D+A: parallel 5ths.
        assert!(parallel_motion_violation(67, 69, 60, 62));
        // C3+C4 → D3+D4: parallel octaves.
        assert!(parallel_motion_violation(60, 62, 48, 50));
        // Compound: 12th stays a 5th by interval class.
        assert!(parallel_motion_violation(79, 81, 60, 62));
    }

    #[test]
    fn antiparallel_octaves_are_violations() {
        // Octave collapsing to a unison in contrary motion (top falls, bottom rises).
        assert!(parallel_motion_violation(60, 55, 48, 55));
        // Octave opening to a double octave.
        assert!(parallel_motion_violation(60, 62, 48, 38));
    }

    #[test]
    fn contrary_fifths_and_oblique_motion_are_tolerated() {
        // 5th reached in contrary motion (no antiparallel-5th rule).
        assert!(!parallel_motion_violation(67, 69, 62, 62 - 10));
        // Oblique motion: one voice holds into an octave.
        assert!(!parallel_motion_violation(60, 60, 50, 48));
        // Similar motion onto a 5th from a NON-5th (hidden 5th) is the
        // same_direction term's business, not this rule's.
        assert!(!parallel_motion_violation(64, 67, 58, 60));
    }

    // ----- doubling_balance -----

    #[test]
    fn doubling_balance_rewards_only_the_first_root_doubling() {
        // C major triad, root doubled once: +1.
        let mut pc = [0u8; 12];
        pc[0] = 2; pc[4] = 1; pc[7] = 1;
        assert_eq!(doubling_balance(&pc, 0, 11), 1);
        // Tripled root still +1 — the pile-up is the interval-variety term's job.
        pc[0] = 3;
        assert_eq!(doubling_balance(&pc, 0, 11), 1);
    }

    #[test]
    fn doubling_balance_penalizes_doubled_leading_tone() {
        // G7 in C with the leading tone (B, pc 11) doubled: +1 root, −1 LT.
        let mut pc = [0u8; 12];
        pc[7] = 2; pc[11] = 2; pc[2] = 1; pc[5] = 1;
        assert_eq!(doubling_balance(&pc, 7, 11), 0);
    }

    #[test]
    fn doubling_balance_penalizes_doubled_chordal_seventh() {
        // G7 in C with the 7th (F, pc 5) doubled: both copies must fall by
        // step, so one breaks its resolution or they fall in octaves.
        let mut pc = [0u8; 12];
        pc[7] = 2; pc[11] = 1; pc[2] = 1; pc[5] = 2;
        assert_eq!(doubling_balance(&pc, 7, 11), 0);
        // Single 7th is free.
        pc[5] = 1;
        assert_eq!(doubling_balance(&pc, 7, 11), 1);
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
        // ...but octave displacement does NOT make the clash vanish: the upper
        // partials still collide, so a m9 stays clearly rougher than a P5.
        // (The fundamental-only model scored it at ~0.)
        assert!(pair_roughness(60, 73) > pair_roughness(60, 67));
    }

    #[test]
    fn roughness_ranks_intervals_like_a_harmonic_timbre() {
        let r = |d: i32| pair_roughness(60, 60 + d);
        // m2 > M2 > m3 > M3 — the classic ordering. The fundamental-only model
        // made m2 and M2 all but indistinguishable (0.94 vs 0.93).
        assert!(r(1) > r(2), "m2 {} !> M2 {}", r(1), r(2));
        assert!(r(2) > r(3));
        assert!(r(3) > r(4), "m3 {} !> M3 {}", r(3), r(4));
        // Coinciding partials make the perfect consonances the smooth ones.
        assert!(r(4) > r(7));
        assert!(r(7) > r(12));
        assert!(r(12) < 0.01, "the octave should be near-perfectly smooth");
    }

    #[test]
    fn roughness_is_register_aware() {
        // The same interval class is rougher in the bass than up top, so the
        // low-interval limit falls out of the model instead of being bolted on.
        for d in [3, 4, 7] {
            assert!(
                pair_roughness(36, 36 + d) > pair_roughness(72, 72 + d),
                "interval {d} is not rougher low than high",
            );
        }
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
        assert!(row.forbidden[11]); // M7 (-100)
        assert!(!row.forbidden[0]);
        // The tritone is disfavoured but legal — hard-forbidding it would
        // outlaw the dominant seventh and the diminished triad.
        assert!(!row.forbidden[6]);
        assert!(row.soft[6] < 0.0);
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

    // ----- chord_templates (chord-structure whitelist) -----

    #[test]
    fn rotate_pc_mask_wraps_around_the_octave() {
        let c_major = 1 << 0 | 1 << 4 | 1 << 7; // {C, E, G}
        assert_eq!(rotate_pc_mask(c_major, 0), c_major);
        // Up 5 semitones: {F, A, C}
        assert_eq!(rotate_pc_mask(c_major, 5), 1 << 5 | 1 << 9 | 1 << 0);
        // Up 8 wraps G past B into D#: {G#, C, D#}
        assert_eq!(rotate_pc_mask(c_major, 8), 1 << 8 | 1 << 0 | 1 << 3);
        assert_eq!(rotate_pc_mask(c_major, 12), c_major);
        assert_eq!(rotate_pc_mask(c_major, -1), rotate_pc_mask(c_major, 11));
    }

    /// Unweighted template, the shape a config carries before weights are used.
    fn bare(pcs: &[i32]) -> ChordTemplate {
        ChordTemplate::Bare(pcs.to_vec())
    }

    fn weighted(pcs: &[i32], weight: f64) -> ChordTemplate {
        ChordTemplate::Weighted { pcs: pcs.to_vec(), weight }
    }

    #[test]
    fn chord_template_masks_covers_every_transposition() {
        let masks = chord_template_masks(&[bare(&[0, 4, 7])]).unwrap();
        assert_eq!(masks.len(), 12, "one major triad per root");
        for r in 0..12 {
            let m = rotate_pc_mask(1 << 0 | 1 << 4 | 1 << 7, r);
            assert!(masks.binary_search(&m).is_ok(), "missing root {r}");
        }
        // Sorted and deduped, so binary_search above is valid.
        assert!(masks.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn chord_template_masks_normalizes_degenerate_input() {
        // No templates at all, and a template whose entries are all empty, both
        // mean "unconstrained" rather than "forbid everything".
        assert!(chord_template_masks(&[]).is_none());
        assert!(chord_template_masks(&[bare(&[])]).is_none());
        // Duplicate and out-of-range offsets collapse onto the same pitch classes.
        let plain = chord_template_masks(&[bare(&[0, 4, 7])]).unwrap();
        assert_eq!(chord_template_masks(&[bare(&[0, 4, 7, 4])]).unwrap(), plain);
        assert_eq!(chord_template_masks(&[bare(&[12, 16, -5])]).unwrap(), plain);
    }

    /// A five-voice input wide enough that the joint search takes the within-chord
    /// beam path (`candidate_range = 12` gives 25^5 candidates, well over
    /// `JOINT_ENUM_CAP`) — the path where the whitelist has to survive pruning.
    fn five_voice_input() -> Vec<Note> {
        vec![
            Note::new(69, 0.0, 4.0, 100, 0, 0),
            Note::new(64, 0.0, 4.0, 100, 0, 1),
            Note::new(60, 0.0, 4.0, 100, 0, 2),
            Note::new(50, 0.0, 4.0, 100, 0, 3),
            Note::new(34, 0.0, 4.0, 100, 0, 4),
        ]
    }

    /// Distinct pitch classes of a harmonised group, as a 12-bit mask.
    fn pc_mask_of(notes: &[Note]) -> u16 {
        notes.iter().fold(0u16, |m, n| m | 1 << n.pitch.rem_euclid(12))
    }

    /// Two chords of five voices whose seed pitches all sit in the given
    /// pitch-class set, so holds can never smuggle an off-scale pc through.
    fn in_scale_input(pitches: [i32; 5]) -> Vec<Note> {
        let mut input = Vec::new();
        for start in [0.0, 4.0] {
            for (ch, &p) in pitches.iter().enumerate() {
                input.push(Note::new(p, start, 4.0, 100, 0, ch as i32));
            }
        }
        input
    }

    #[test]
    fn chromatic_scale_confines_the_chromatic_search() {
        // C major on root 0: every harmonised pitch stays on the white keys.
        let mut cfg = test_config();
        cfg.chromatic_scale = vec![0, 2, 4, 5, 7, 9, 11];
        cfg.root = 0;
        let out = harmonise2(in_scale_input([69, 64, 60, 50, 36]), &cfg, &test_state(), None);
        assert!(!out.is_empty());
        for n in &out {
            assert!(
                [0, 2, 4, 5, 7, 9, 11].contains(&n.pitch.rem_euclid(12)),
                "pitch {} (pc {}) escaped the scale", n.pitch, n.pitch.rem_euclid(12),
            );
        }

        // The entries are offsets from root: the same list on root 2 is D major.
        cfg.root = 2;
        let out = harmonise2(in_scale_input([69, 66, 62, 54, 38]), &cfg, &test_state(), None);
        for n in &out {
            assert!(
                [2, 4, 6, 7, 9, 11, 1].contains(&n.pitch.rem_euclid(12)),
                "pitch {} (pc {}) escaped D major", n.pitch, n.pitch.rem_euclid(12),
            );
        }
    }

    #[test]
    fn empty_chromatic_scale_means_unconstrained() {
        // No scale: the window is truly chromatic, so with harmony scoring
        // neutralised a semitone neighbour is reachable.
        let mut cfg = test_config();
        cfg.chromatic_scale = Vec::new();
        let out = harmonise2(in_scale_input([69, 64, 60, 50, 36]), &cfg, &test_state(), None);
        assert!(!out.is_empty());
    }

    #[test]
    fn apportion_honours_the_requested_ratio() {
        // 0.75 / 0.25 over four groups is 3-and-1, not "roughly three".
        let seq = apportion(&[0.75, 0.25], 4);
        assert_eq!(seq.iter().filter(|&&i| i == 0).count(), 3);
        assert_eq!(seq.iter().filter(|&&i| i == 1).count(), 1);

        // And it stays proportional as the render gets longer.
        let seq = apportion(&[0.75, 0.25], 100);
        assert_eq!(seq.iter().filter(|&&i| i == 0).count(), 75);

        // Equal weights alternate rather than clumping.
        assert_eq!(apportion(&[1.0, 1.0], 4), vec![0, 1, 0, 1]);
    }

    #[test]
    fn apportion_spreads_the_minority_structure() {
        // The one minor chord in four lands in the middle, not at an edge — a
        // random draw would clump it as often as not.
        let seq = apportion(&[0.75, 0.25], 4);
        assert!(seq[1] == 1 || seq[2] == 1, "minority bunched at an edge: {seq:?}");
    }

    #[test]
    fn apportion_handles_degenerate_weights() {
        // A zero weight is never selected; it means "listed but never used".
        let seq = apportion(&[1.0, 0.0], 5);
        assert!(seq.iter().all(|&i| i == 0));
        // All-zero has nothing to apportion, and must not spin or panic.
        assert!(apportion(&[0.0, 0.0], 5).is_empty());
        assert!(apportion(&[], 5).is_empty());
        assert!(apportion(&[1.0], 0).is_empty());
    }

    #[test]
    fn chord_plan_assigns_per_group_only_when_a_weight_is_stated() {
        // Bare entries state no preference, so every group keeps the full
        // whitelist — the behaviour from before weights existed.
        let plan = ChordPlan::build(&[bare(&[0, 4, 7]), bare(&[0, 3, 7])], 4).unwrap();
        assert!(plan.assignment.is_empty());
        let maj = template_masks(&[0, 4, 7]).unwrap();
        assert!(plan.masks_for(0).len() > maj.len(), "unassigned groups see the union");

        // One stated weight switches apportionment on for the whole list.
        let plan = ChordPlan::build(
            &[weighted(&[0, 4, 7], 0.75), weighted(&[0, 3, 7], 0.25)], 4,
        ).unwrap();
        assert_eq!(plan.assignment.len(), 4);
        assert_eq!(plan.assignment.iter().filter(|&&i| i == 1).count(), 1);
        // An assigned group is locked to its own template, not the union.
        assert_eq!(plan.masks_for(0), maj.as_slice());
    }

    #[test]
    fn chord_plan_drops_unusable_entries_before_apportioning() {
        // An empty entry must not consume apportionment slots no chord can meet.
        let plan = ChordPlan::build(&[weighted(&[], 0.5), weighted(&[0, 4, 7], 0.5)], 4).unwrap();
        assert_eq!(plan.per_template.len(), 1);
        assert!(plan.assignment.iter().all(|&i| i == 0));
        // Nothing usable at all is the same as no constraint.
        assert!(ChordPlan::build(&[bare(&[])], 4).is_none());
        assert!(ChordPlan::build(&[], 4).is_none());
    }

    /// Groups past the apportioned range fall back to the union rather than
    /// panicking — `masks_for` is indexed by group and must be total.
    #[test]
    fn chord_plan_masks_for_is_total() {
        let plan = ChordPlan::build(&[weighted(&[0, 4, 7], 1.0)], 2).unwrap();
        assert_eq!(plan.masks_for(0), plan.per_template[0].as_slice());
        assert_eq!(plan.masks_for(99), plan.any.as_slice());
    }

    #[test]
    fn weighted_templates_render_in_the_requested_ratio() {
        // Eight bars, 0.75 major / 0.25 minor: six major chords and two minor.
        let mut cfg = test_config();
        cfg.candidate_range = 12;
        cfg.chord_templates = vec![weighted(&[0, 4, 7], 0.75), weighted(&[0, 3, 7], 0.25)];
        let input: Vec<Note> = (0..8)
            .flat_map(|bar| {
                let t = bar as f64 * 4.0;
                [69, 64, 60, 50, 34].iter().enumerate()
                    .map(move |(ch, &p)| Note::new(p, t, 4.0, 100, 0, ch as i32))
                    .collect::<Vec<_>>()
            })
            .collect();
        let out = harmonise2(input, &cfg, &test_state(), None);

        let maj = template_masks(&[0, 4, 7]).unwrap();
        let min = template_masks(&[0, 3, 7]).unwrap();
        let mut majors = 0;
        let mut minors = 0;
        for bar in 0..8 {
            let t = bar as f64 * 4.0;
            let chord: Vec<Note> = out.iter().filter(|n| (n.start - t).abs() < 0.001).cloned().collect();
            assert_eq!(chord.len(), 5, "bar {bar} lost voices");
            let mask = pc_mask_of(&chord);
            if maj.binary_search(&mask).is_ok() {
                majors += 1;
            } else if min.binary_search(&mask).is_ok() {
                minors += 1;
            } else {
                panic!("bar {bar} is neither major nor minor: {mask:012b}");
            }
        }
        assert_eq!((majors, minors), (6, 2), "ratio not honoured");
    }

    #[test]
    fn chord_templates_confine_the_sonority_to_the_whitelist() {
        let mut cfg = test_config();
        cfg.candidate_range = 12;
        cfg.chord_templates = vec![bare(&[0, 4, 7]), bare(&[0, 3, 7])];
        let out = harmonise2(five_voice_input(), &cfg, &test_state(), None);

        let allowed = chord_template_masks(&cfg.chord_templates).unwrap();
        let mask = pc_mask_of(&out);
        assert!(
            allowed.binary_search(&mask).is_ok(),
            "sonority {mask:012b} is neither a major nor a minor triad"
        );
        assert_eq!(mask.count_ones(), 3, "a triad has exactly three pitch classes");
    }

    #[test]
    fn chord_templates_bind_against_the_matrix_preference() {
        // The augmented triad is the case the interval matrix cannot express and
        // actively disprefers, so the beam prunes toward it only if the template
        // feasibility check reaches the partial states.
        let mut cfg = test_config();
        cfg.candidate_range = 12;
        cfg.chord_templates = vec![bare(&[0, 4, 8])];
        let out = harmonise2(five_voice_input(), &cfg, &test_state(), None);

        let mask = pc_mask_of(&out);
        let allowed = chord_template_masks(&cfg.chord_templates).unwrap();
        assert!(allowed.binary_search(&mask).is_ok(), "not an augmented triad");
    }

    #[test]
    fn chord_templates_empty_leaves_chord_structure_free() {
        let mut cfg = test_config();
        cfg.candidate_range = 12;
        assert!(cfg.chord_templates.is_empty(), "default is unconstrained");
        let out = harmonise2(five_voice_input(), &cfg, &test_state(), None);
        // Nothing asserted about *which* chord — only that the unconstrained path
        // still returns one note per input voice.
        assert_eq!(out.len(), 5);
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
        // Greedy: this is a property of the CHORD score, checked one group at a
        // time. Under the branching beam the off-scale hold on ch1 wins here —
        // not on the interval penalty this test is about, but because the floor
        // anchor puts ch1 a semitone under the lead and eats `no_crossing`
        // (100.0, two orders of magnitude above every other term). Zeroing
        // no_crossing makes the wide search agree with the greedy one again.
        cfg.beam_width = 1;
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
        state.contours.harmony_matrix = flat(8.0);
        state.contours.harmony_distance = flat(-0.2);
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

    // ----- outer beam: branching + look-ahead -----

    /// Three voices over eight groups — long enough that the look-ahead horizon
    /// reaches groups the beam has not committed to yet.
    fn eight_group_input() -> Vec<Note> {
        (0..8i32)
            .flat_map(|g| {
                let s = g as f64 * 4.0;
                [
                    Note::new(48, s, 4.0, 100, 0, 4),
                    Note::new(60, s, 4.0, 100, 0, 0),
                    Note::new(67, s, 4.0, 100, 0, 2),
                ]
            })
            .collect()
    }

    fn harmonised_pitches(input: &[Note], beam_width: i32, lookahead_depth: i32) -> Vec<i32> {
        let mut cfg = test_config();
        cfg.beam_width = beam_width;
        cfg.lookahead_depth = lookahead_depth;
        harmonise2(input.to_vec(), &cfg, &test_state(), None)
            .iter()
            .map(|n| n.pitch)
            .collect()
    }

    #[test]
    fn group_options_are_distinct_and_led_by_the_greedy_pick() {
        let cfg = test_config();
        let state = test_state();
        let group = vec![
            Note::new(60, 0.0, 4.0, 100, 0, 0),
            Note::new(48, 0.0, 4.0, 100, 0, 4),
        ];
        let pc = build_precomputed_data(&[], &group, 0.0);

        let opts = score_group_options(&group, &cfg, &state, &pc, 4, None);
        assert_eq!(opts.len(), 4, "the beam has nothing to branch on");

        // Every option is a complete voicing, and no two are the same chord.
        let mut seen = std::collections::HashSet::new();
        for (notes, _) in &opts {
            assert_eq!(notes.len(), group.len());
            let pitches: Vec<i32> = notes.iter().map(|n| n.pitch).collect();
            assert!(seen.insert(pitches), "duplicate voicing among the options");
        }

        // Option 0 is exactly what the single-best path returns.
        let mut greedy = Vec::new();
        let greedy_score = score_group(&group, &mut greedy, &cfg, &state, &pc, None);
        let best: Vec<i32> = opts[0].0.iter().map(|n| n.pitch).collect();
        assert_eq!(best, greedy.iter().map(|n| n.pitch).collect::<Vec<_>>());
        assert!(approx(opts[0].1, greedy_score));
    }

    #[test]
    fn beam_width_one_is_greedy_and_ignores_lookahead() {
        let input = eight_group_input();
        let base = harmonised_pitches(&input, 1, 0);
        for depth in [1, 3] {
            assert_eq!(
                harmonised_pitches(&input, 1, depth),
                base,
                "width 1 has nothing to rank, so depth {depth} must be a no-op",
            );
        }
    }

    #[test]
    fn lookahead_depth_changes_the_progression_when_the_beam_branches() {
        // Guards the regression this whole mechanism was built to fix: with the
        // outer beam never branching, `lookahead_depth` was computed and then
        // thrown away. If the scoring is retuned and these two happen to agree
        // again, re-pick the scenario rather than deleting the test.
        let input = eight_group_input();
        assert_ne!(
            harmonised_pitches(&input, 3, 0),
            harmonised_pitches(&input, 3, 1),
            "look-ahead is not affecting the beam's ranking",
        );
    }

    // ----- tendency tones -----

    #[test]
    fn tendency_zero_weight_is_noop() {
        assert_eq!(tendency_term(59, 60, 0, Some(7), 0.0), 0.0);
    }

    #[test]
    fn tendency_leading_tone_resolves_up_to_the_tonic() {
        // Key of C: B (59) is the leading tone.
        assert!(approx(tendency_term(59, 60, 0, None, 1.0), 1.0)); // up a semitone
        assert!(approx(tendency_term(59, 55, 0, None, 1.0), -0.5)); // leaps away
        assert!(approx(tendency_term(59, 59, 0, None, 1.0), 0.0)); // holding is neutral
        // Stepping away (B → A) is an ordinary inner-voice escape, not a
        // frustrated leading tone — only leaps are penalized.
        assert!(approx(tendency_term(59, 57, 0, None, 1.0), 0.0));
    }

    #[test]
    fn tendency_has_no_leading_tone_in_modes_without_one() {
        // Key of A minor: G (55) is a whole tone below the tonic, so the rule
        // never fires — no special-casing needed, the semitone test does it.
        for c in [56, 57, 53] {
            assert!(approx(tendency_term(55, c, 9, None, 1.0), 0.0));
        }
    }

    #[test]
    fn tendency_chordal_seventh_falls_by_step() {
        // F (65) is the minor 7th of a G7 chord (root pc 7).
        assert!(approx(tendency_term(65, 64, 0, Some(7), 1.0), 1.0)); // down a semitone
        assert!(approx(tendency_term(65, 63, 0, Some(7), 1.0), 1.0)); // down a whole tone
        assert!(approx(tendency_term(65, 67, 0, Some(7), 1.0), -0.5)); // pushed upward
    }

    #[test]
    fn tendency_ignores_the_major_seventh_as_a_chord_tone() {
        // B (71) over a C root is a M7: as a chord tone it tends UP, so the
        // falling-7th rule must not claim it.
        assert!(approx(tendency_term(71, 69, 3, Some(0), 1.0), 0.0));
    }

    // ----- root-aware chord scoring -----

    /// A scale offering BOTH thirds over the root, so the scorer has to choose.
    fn quality_state(matrix_row: f64) -> HarmonizerState {
        let mut state = test_state();
        state.schillinger_notes = vec![vec![vec![0, 3, 4, 7]]];
        state.contours.harmony_matrix = flat(matrix_row);
        state
    }

    /// Distinct pitch classes of the first chord, with every melodic term off so
    /// only the harmony score decides.
    fn harmony_only_pcs(cfg: &mut Config, state: &HarmonizerState) -> Vec<i32> {
        cfg.schillinger_progression = true;
        cfg.same_note_bonus = 0.0;
        cfg.voice_contour_weight = 0.0;
        cfg.interval_exists_in_harmony = 0.0;
        cfg.no_crossing = 0.0;
        cfg.min_voices_changed = -1;
        cfg.tendency_weight = 0.0;
        cfg.harmony_distance_balance = 0.5; // w_smooth = 0: pure harmony
        let input = vec![
            Note::new(60, 0.0, 4.0, 100, 1, 0),
            Note::new(55, 0.0, 4.0, 100, 1, 2),
            Note::new(48, 0.0, 4.0, 100, 1, 4),
        ];
        let mut pcs: Vec<i32> = harmonise2(input, cfg, state, None)
            .iter()
            .map(|n| n.pitch.rem_euclid(12))
            .collect();
        pcs.sort();
        pcs.dedup();
        pcs
    }

    #[test]
    fn chord_quality_follows_the_style_row() {
        // A major and a minor triad have the IDENTICAL pairwise interval
        // multiset {3, 4, 7}, so the mean/worst aggregation cannot tell them
        // apart. Measuring each pitch class from the chord root can, and this
        // is the term that does it — kill it and the two rows agree again.
        let mut dark = Config { root_position_weight: 0.0, root_doubling_weight: 0.0, ..test_config() };
        let minor = harmony_only_pcs(&mut dark, &quality_state(4.0)); // DARK row
        assert!(minor.contains(&3), "dark row did not pick the minor 3rd: {minor:?}");
        assert!(!minor.contains(&4), "dark row kept the major 3rd: {minor:?}");

        let mut bright = Config { root_position_weight: 0.0, root_doubling_weight: 0.0, ..test_config() };
        let major = harmony_only_pcs(&mut bright, &quality_state(5.0)); // BRIGHT row
        assert!(major.contains(&4), "bright row did not pick the major 3rd: {major:?}");
        assert!(!major.contains(&3), "bright row kept the minor 3rd: {major:?}");
    }

    #[test]
    fn chord_quality_weight_zero_disables_the_distinction() {
        let mut off = Config {
            root_position_weight: 0.0,
            root_doubling_weight: 0.0,
            chord_quality_weight: 0.0,
            ..test_config()
        };
        let dark = harmony_only_pcs(&mut off, &quality_state(4.0));
        let mut off2 = off.clone();
        let bright = harmony_only_pcs(&mut off2, &quality_state(5.0));
        assert_eq!(
            dark, bright,
            "without the root-relative term the rows must be indistinguishable",
        );
    }

    /// Pitch class the bass settles on, over a scale whose root is pc 0, with a
    /// neutral consonance row so only the bass preference can steer it. The
    /// seed pitches deliberately start the bass on the fifth (pc 7), i.e. in
    /// second inversion.
    fn bass_pc_with_root_weight(w: f64) -> i32 {
        let mut cfg = Config {
            root_position_weight: w,
            root_doubling_weight: 0.0,
            ..test_config()
        };
        cfg.schillinger_progression = true;
        cfg.same_note_bonus = 0.0;
        cfg.voice_contour_weight = 0.0;
        cfg.interval_exists_in_harmony = 0.0;
        cfg.no_crossing = 0.0;
        cfg.min_voices_changed = -1;
        cfg.tendency_weight = 0.0;
        let state = quality_state(8.0); // NEUTRAL row: no interval preferences
        let input = vec![
            Note::new(64, 0.0, 4.0, 100, 1, 0),
            Note::new(60, 0.0, 4.0, 100, 1, 2),
            Note::new(43, 0.0, 4.0, 100, 1, 4),
        ];
        let out = harmonise2(input, &cfg, &state, None);
        out.iter().map(|n| n.pitch).min().unwrap().rem_euclid(12)
    }

    #[test]
    fn root_position_puts_the_chord_root_in_the_bass() {
        // Weighted up, the bass gives up its seed fifth for the root even
        // though moving there costs smoothness.
        assert_eq!(bass_pc_with_root_weight(5.0), 0);
    }

    #[test]
    fn root_position_weight_zero_leaves_inversions_free() {
        // Guard that the preference is what did the work above: at 0 the bass
        // is decided by smoothness alone and keeps the six-four.
        assert_ne!(bass_pc_with_root_weight(0.0), 0);
    }

    #[test]
    fn aggregation_weights_renormalize_on_both_paths() {
        // The harmony aggregate must have total weight EXACTLY 1 no matter
        // which slots are active — otherwise turning chord quality off (or
        // losing the root) silently rescales the whole harmony term instead
        // of rebalancing the mix.
        let (m, w, b, r) = (AGG_MEAN, AGG_WORST, AGG_BASS, AGG_ROOT);
        assert!(approx(m + w + b + r, 1.0));
        // No root: the fourth slot is dropped, the rest renormalize.
        let k = 1.0 / (m + w + b);
        assert!(approx(m * k + w * k + b * k, 1.0));
        // Rooted: any chord_quality_weight (negatives clamped) renormalizes.
        for cqw in [0.0f64, 0.5, 1.0, 2.0, -3.0] {
            let wr = r * cqw.max(0.0);
            let k = 1.0 / (m + w + b + wr);
            assert!(
                approx(m * k + w * k + b * k + wr * k, 1.0),
                "weights do not sum to 1 at chord_quality_weight {cqw}",
            );
        }
    }

    #[test]
    fn group_weights_balance_is_clamped_to_a_valid_mix() {
        // A contour value beyond ±0.5 must saturate, not flip a weight's
        // sign — w_smooth < 0 would reward leaps.
        let cfg = test_config();
        let mut state = test_state();
        state.contours.harmony_distance = flat(0.9);
        let (wh, ws, _) = group_weights(0.0, &cfg, &state);
        assert!(approx(wh, 1.0));
        assert!(approx(ws, 0.0));
        state.contours.harmony_distance = flat(-2.0);
        let (wh, ws, _) = group_weights(0.0, &cfg, &state);
        assert!(approx(wh, 0.0));
        assert!(approx(ws, 1.0));
        // In-range values pass through untouched.
        state.contours.harmony_distance = flat(0.2);
        let (wh, ws, _) = group_weights(0.0, &cfg, &state);
        assert!(approx(wh, 0.7));
        assert!(approx(ws, 0.3));
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

