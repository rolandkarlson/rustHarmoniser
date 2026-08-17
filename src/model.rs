use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct Note {
    pub pitch: i32,
    pub start: f64,
    pub duration: f64,
    pub velocity: i32,
    pub muted: i32, // 0 or 1
    pub channel: i32,
    pub probability: i32,
}

impl Note {
    pub fn new(pitch: i32, start: f64, duration: f64, velocity: i32, muted: i32, channel: i32) -> Self {
        Self {
            pitch,
            start,
            duration,
            velocity,
            muted,
            channel,
            probability: 0,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Config {
    pub schillinger_progression: bool,
    pub last_note_exist_in_voice: f64,
    pub same_direction: f64,
    pub consecutive_octav_fift: f64,
    pub no_crossing: f64,
    pub last_note_same: f64,
    pub mode: i32,
    pub interval_exists_in_harmony: f64,
    // New fields
    pub voice_rhythm: Vec<f64>,
    pub schillinger_sequence: Vec<i32>,
    pub pl: i32,
    pub harmony_distance_balance: f64,
    pub lookahead_depth: i32,
    /// Outer beam search over chord progressions: how many partial progressions
    /// survive each group, AND how many alternative voicings each one branches
    /// into for the next group (same knob for both). 1 = greedy — every group
    /// takes its locally best chord and `lookahead_depth` has no effect, since
    /// there is nothing to rank. Cost per group grows as beam_width² ×
    /// (1 + lookahead_depth) group scorings, so raise it deliberately.
    #[serde(default = "default_beam_width")]
    pub beam_width: i32,
    pub render_length: i32,
    pub rng_seed: f64,
    pub voice_contour: Option<Vec<Vec<i32>>>,
    pub voice_rhythm_contour: Option<Vec<Vec<f64>>>,
    pub voice_contour_resolution: f64,
    pub chord_structure: Vec<i32>,
    pub harmony_distance_contour: Option<Vec<f64>>,
    pub mode_contour: Option<Vec<f64>>,
    pub chord_structure_contour: Option<Vec<Vec<f64>>>,
    pub schillinger_ex_contour: Option<Vec<Vec<f64>>>,
    pub harmony_matrix_contour: Option<Vec<f64>>,
    /// Per-voice, per-step melody-force weight over time (outer = 16 voices,
    /// inner = steps, like `voice_rhythm_contour`). When present and the
    /// selected voice's row is non-empty it overrides the scalar `melody_force`
    /// (sampled like the other contours); otherwise the scalar is used.
    /// See `melody_force` / melody_force_term.
    pub melody_force_contour: Option<Vec<Vec<f64>>>,
    /// 9×12 consonance scoring matrix (style rows × interval columns).
    /// None falls back to the built-in default in harmonizer::HARMONY_MATRIX.
    #[serde(default)]
    pub harmony_matrix: Option<Vec<Vec<f64>>>,
    pub main_pitch: i32,
    #[serde(default)]
    pub use_floor: bool,
    #[serde(default)]
    pub use_ceiling: bool,
    #[serde(default)]
    pub use_leading_voice: bool,
    #[serde(default)]
    pub leading_voice_track: i32,
    #[serde(default = "default_leading_clip")]
    pub leading_voice_clip: i32,
    pub use_resolve:bool,
    #[serde(default)]
    pub root: i32,
    /// Group-level common-tone control (relationship to the IMMEDIATELY preceding
    /// chord), distinct from the per-voice last_note_* penalties (a voice's own
    /// melodic history). Penalty subtracted per voice that holds its previous pitch.
    #[serde(default)]
    pub common_tone_penalty: f64,
    /// (Superseded by min/max_voices_changed — kept for snapshot compatibility.)
    /// Soft cap on common tones per chord. -1 disables.
    #[serde(default = "default_neg_one")]
    pub max_common_tones: i32,
    /// Voice-change budget: how many voices may change pitch between consecutive
    /// chords. Enforced by holding the lowest-benefit voices on their previous
    /// pitch (parsimonious voice leading). -1 disables each bound.
    #[serde(default = "default_neg_one")]
    pub min_voices_changed: i32,
    #[serde(default = "default_neg_one")]
    pub max_voices_changed: i32,
    /// "Stickiness": score bonus a NON-leader voice gets for holding its previous
    /// pitch (a common tone). Higher = voices keep common tones unless moving is
    /// clearly more consonant. The leader voice is excluded so it stays free to
    /// move. (Principled replacement for the old +30 unison bonus.)
    #[serde(default = "default_same_note_bonus")]
    pub same_note_bonus: f64,
    /// How strongly the per-voice pitch contour pulls candidates toward its target
    /// pitch (quadratic spring, one octave = full weight). 0 = contour ignored.
    #[serde(default = "default_one_f64")]
    pub voice_contour_weight: f64,
    /// Pitch search window: each voice considers previous pitch ± this many
    /// semitones (non-Schillinger candidate generation). Was a hardcoded 3.
    #[serde(default = "default_candidate_range")]
    pub candidate_range: i32,
    /// Melodic pressure applied to EVERY voice (unlike the leader-only repeat
    /// penalties): candidates are penalized by how recently/often they appeared
    /// in the voice's last 5 notes (recency-decayed, so A-B-A-B circling is
    /// caught, not just immediate repeats), and stepwise motion (1-2 semitones)
    /// gets a small reward. 0 = off; 1.0 ≈ the magnitude of the other ±1 terms.
    #[serde(default)]
    pub melody_force: f64,
    /// Starting/seed pitches for the 5 generated voices (high → low). Voice 0 is
    /// the leading voice. Editable in the GUI "Start Notes" modal, or fetched from
    /// the last chord of an Ableton clip. Missing/short → falls back per index to
    /// the historical defaults [70, 65, 60, 50, 34].
    #[serde(default = "default_start_notes")]
    pub start_notes: Vec<i32>,
    /// Blend between the H-Matrix style preference and register-aware sensory
    /// roughness in the pairwise consonance term. 0 = pure style/pitch-class,
    /// 1 = pure psychoacoustics. See harmonizer::pair_roughness.
    #[serde(default = "default_roughness_weight")]
    pub roughness_weight: f64,
    /// Scales the root-relative chord-quality term (major vs minor vs
    /// suspended), which is the only part of the harmony score that can tell
    /// chord qualities apart — the pairwise interval multiset cannot. 0 = off.
    #[serde(default = "default_one_f64")]
    pub chord_quality_weight: f64,
    /// Preference for the bass on the chord root: full bonus in root position,
    /// 0.4 with the third in the bass, 0 with the fifth (six-four), -0.3 for a
    /// non-chord tone. 0 = inversions are free; at 1.0 and above the bass stops
    /// taking inversions altogether.
    #[serde(default = "default_root_position")]
    pub root_position_weight: f64,
    /// Reward for doubling the chord root (the first doubling only) and equal
    /// penalty per extra voice on the key's leading tone. 0 = off.
    #[serde(default = "default_root_doubling")]
    pub root_doubling_weight: f64,
    /// Tendency-tone pressure: leading tone up to the tonic, chordal minor 7th
    /// down by step. Applied per voice on (previous pitch → candidate).
    /// 0 = off. See harmonizer::tendency_term.
    #[serde(default = "default_tendency_weight")]
    pub tendency_weight: f64,
    /// Generate `schillinger_sequence` from the mode's chord-transition table
    /// instead of using the literal sequence: one phrase of `pl` bars per
    /// `render_length`, each closing V → I. See schillinger::gen_cadenced_progression.
    #[serde(default)]
    pub use_generated_progression: bool,
}

pub const DEFAULT_START_NOTES: [i32; 5] = [70, 65, 60, 50, 34];
fn default_start_notes() -> Vec<i32> { DEFAULT_START_NOTES.to_vec() }
fn default_leading_clip() -> i32 { 1 }
fn default_neg_one() -> i32 { -1 }
fn default_same_note_bonus() -> f64 { 2.0 }
fn default_one_f64() -> f64 { 1.0 }
fn default_candidate_range() -> i32 { 3 }
fn default_beam_width() -> i32 { 3 }
fn default_roughness_weight() -> f64 { 0.5 }
fn default_root_position() -> f64 { 0.5 }
fn default_root_doubling() -> f64 { 0.5 }
fn default_tendency_weight() -> f64 { 0.5 }

impl Default for Config {
    fn default() -> Self {
        Self {
            schillinger_progression: true,
            // Rescaled for the normalized ([-1,1]) harmony/distance scoring.
            last_note_exist_in_voice: 1.0,
            same_direction: 1.0,
            consecutive_octav_fift: 0.0,
            no_crossing: 100.0,
            last_note_same: 0.5,
            mode: 0,
            interval_exists_in_harmony: 1.0,
            voice_rhythm: vec![4.0],
            schillinger_sequence: vec![0, 3, 4, 0],
            pl: 4,
            harmony_distance_balance: 0.2,
            lookahead_depth: 2,
            beam_width: default_beam_width(),
            render_length: 2,
            rng_seed: 1.0,
            voice_contour: None,
            voice_rhythm_contour: None,
            voice_contour_resolution: 4.0,
            chord_structure: vec![0, 1, 2, 4, 5],
            harmony_distance_contour: None,
            mode_contour: None,
            chord_structure_contour: None,
            schillinger_ex_contour: None,
            harmony_matrix_contour: None,
            melody_force_contour: None,
            harmony_matrix: None,
            main_pitch: 0,
            use_floor: false,
            use_ceiling: false,
            use_leading_voice: false,
            leading_voice_track: 0,
            leading_voice_clip: 1,
            use_resolve: false,
            root: 0,
            common_tone_penalty: 0.0,
            max_common_tones: -1,
            // Non-leader voices stick to their common tones (same_note_bonus); the
            // leader moves. min_voices_changed = 1 is a safety floor so a chord is
            // never fully static. max budget off — stickiness handles parsimony.
            min_voices_changed: 1,
            max_voices_changed: -1,
            same_note_bonus: 2.0,
            voice_contour_weight: 1.0,
            candidate_range: 3,
            melody_force: 0.0,
            start_notes: default_start_notes(),
            roughness_weight: default_roughness_weight(),
            chord_quality_weight: 1.0,
            // 0.5 leaves the bass in root position roughly 60% of the time on a
            // default render; at 1.0 it never takes an inversion at all.
            root_position_weight: default_root_position(),
            root_doubling_weight: default_root_doubling(),
            tendency_weight: default_tendency_weight(),
            use_generated_progression: false,
        }
    }
}

impl Config {
    pub fn randomize_contours(&mut self) {
        use crate::utils::SeededRng;

        let mut contours = Vec::new();
        // 16 voices
        for _ in 0..16 {
            // Generate 3 to 5 points
            let num_points = 3 + SeededRng::random_int(3); // 0..3 -> 0..2.99 -> 0..2. 3 + 0..2 = 3..5
            // utils.rs: (self._seeded_random(max as f64, 0.0)).floor() as i32
            // if max=8, returns 0..7.99 -> floor -> 0..7.
            // 3 + 0..7 = 3..10. Correct.

            let mut points: Vec<(f64, f64)> = Vec::new();
            let total_len = (self.render_length * 32) as f64;

            for _ in 0..num_points {
                let x = SeededRng::seeded_random(total_len, 0.0);
                let y = SeededRng::seeded_random(12.0, -12.0);
                points.push((x, y));
            }

            // Sort by x
            points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Interpolate
            let mut contour_vec = Vec::new();
            // Calculate number of steps.
            // +1 to cover the end?
            let steps = (total_len / self.voice_contour_resolution).ceil() as usize;

            for i in 0..steps {
                let pos = (i as f64) * self.voice_contour_resolution;

                let mut y = 0.0;
                if points.is_empty() {
                    y = 0.0;
                } else if pos <= points[0].0 {
                    y = points[0].1;
                } else if pos >= points.last().unwrap().0 {
                    y = points.last().unwrap().1;
                } else {
                    // Linear interpolation
                    for j in 0..points.len() - 1 {
                        if pos >= points[j].0 && pos <= points[j + 1].0 {
                            let p1 = points[j];
                            let p2 = points[j + 1];
                            // Avoid div by zero if x matches (unlikely with random, but possible if sorted puts them close)
                            if (p2.0 - p1.0).abs() < 0.0001 {
                                y = p1.1;
                            } else {
                                let t = (pos - p1.0) / (p2.0 - p1.0);
                                y = p1.1 + t * (p2.1 - p1.1);
                            }
                            break;
                        }
                    }
                }
                contour_vec.push(y.round() as i32);
            }
            contours.push(contour_vec);
        }
        self.voice_contour = Some(contours);
    }

    pub fn init_contours(&mut self) {
        let steps = ((self.pl as f64 * 4.0 * self.render_length as f64) / self.voice_contour_resolution).ceil() as usize;
        
        let mut harmony = vec![0.2; steps];
        let mut mode = vec![0.0; steps];
        let mut chord = vec![1.0; steps];
        let mut rhythm = vec![vec![4.0; steps]; 16];

        let snaps = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0];

        for i in 0..steps {
            let phase = (i as f64) / ((steps - 1).max(1) as f64);
            let tension = (phase * std::f64::consts::PI).sin();

            // Harmony Distance: 0.2 -> 0.45 -> 0.2
            harmony[i] = 0.2 + (0.25 * tension);

            // Mode: 0.0 -> 1.0 (mid) -> 0.0
            mode[i] = if phase >= 0.4 && phase <= 0.8 { 1.0 } else { 0.0 };

            // Chord Structure: 0.0 -> 5.0 -> 0.0
            chord[i] = (tension * 5.0).round().clamp(0.0, 5.0);

            // Rhythm Fractal Predictability (Macro + Mid + Micro Phrase Alignments)
            // self.pl roughly maps towards fundamental block layouts natively framing structures predictably
            let local_phase = ((i % self.pl as usize) as f64) / (self.pl as f64).max(1.0);
            let local_tension = (local_phase * std::f64::consts::PI).sin(); // 0 -> 1 -> 0 over 1 bar
            
            let mid_phase = ((i % (self.pl as usize * 4)) as f64) / (self.pl as f64 * 4.0).max(1.0);
            let mid_tension = (mid_phase * std::f64::consts::PI).sin(); // Over 4 bars

            // Fractal composite: 40% global structure + 30% section + 30% strict local phrase
            let rhythm_tension = tension * 0.4 + mid_tension * 0.3 + local_tension * 0.3;

            // Voice Rhythm
            for v in 0..16 {
                let base_speed = if v >= 3 { 4.0 } else { 1.0 };
                // Slower at structural starts/ends locally and globally, peaks towards faster speeds logically
                let target = base_speed - rhythm_tension * (if v >= 3 { 2.0 } else { 0.75 });
                
                // Quantize to snaps
                let mut closest = snaps[0];
                let mut min_diff = (target - closest).abs();
                for &s in snaps.iter().skip(1) {
                    let diff = (target - s).abs();
                    if diff < min_diff { min_diff = diff; closest = s; }
                }
                rhythm[v][i] = closest;
            }
        }

        // Harmony Matrix: start Strict Classical(0), move to Jazz(1) at peak, back to Strict(0)
        let harmony_matrix = (0..steps).map(|i| {
            let phase = (i as f64) / ((steps - 1).max(1) as f64);
            let tension = (phase * std::f64::consts::PI).sin();
            (tension * 1.0).round().clamp(0.0, 7.0)
        }).collect();

        // Melody force: per-voice rows, flat at the scalar value so default
        // behaviour is unchanged, but the contour editor has data to shape over
        // time (one row per voice, like voice_rhythm_contour).
        let melody_force = vec![vec![self.melody_force; steps]; 16];

        self.harmony_distance_contour = Some(harmony);
        self.mode_contour = Some(mode);
        self.chord_structure_contour = Some(vec![chord; 16]);
        self.voice_rhythm_contour = Some(rhythm);
        self.harmony_matrix_contour = Some(harmony_matrix);
        self.melody_force_contour = Some(melody_force);

        // Seed the editable scoring matrix with the built-in defaults so the UI
        // starts from the current values ("default should be as it is").
        if self.harmony_matrix.is_none() {
            self.harmony_matrix = Some(
                crate::harmonizer::HARMONY_MATRIX
                    .iter()
                    .map(|row| row.to_vec())
                    .collect(),
            );
        }
    }
}
