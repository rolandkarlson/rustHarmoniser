//! The whole render pipeline as a pure function: seed the RNG, generate the
//! five voices, build the Schillinger progression, harmonize, apply the main
//! pitch offset. No I/O — the server crate decides what to do with the result.

use crate::contour::Contours;
use crate::harmonizer::{gen_voice, gen_voice_from_notes, harmonise_explained, HarmonizerState};
use crate::model::{Config, Note, DEFAULT_START_NOTES};
use crate::trace::GroupBreakdown;
use crate::utils;
use serde::{Deserialize, Serialize};
use std::sync::mpsc::Sender;

/// A leading-voice clip supplied from outside (e.g. fetched from Ableton).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Leading {
    pub notes: Vec<Note>,
    pub clip_length: f64,
}

/// Everything a render produces. `notes` already carry `config.main_pitch`,
/// and so do the pitches inside `breakdown`, so both live in output space.
pub struct RenderResult {
    pub notes: Vec<Note>,
    /// The named scoring story of every chosen chord — why the search picked
    /// what it picked, term by term.
    pub breakdown: Vec<GroupBreakdown>,
    /// The per-voice, per-bar Schillinger pitch-class stacks the render was
    /// scored against — kept for analysis (e.g. bass-on-root statistics).
    pub schillinger_notes: Vec<Vec<Vec<i32>>>,
}

/// Run a full render. Deterministic: the same `config` (including `rng_seed`)
/// and `leading` always produce the same result.
pub fn render(
    config: &Config,
    leading: Option<&Leading>,
    progress: Option<&Sender<(usize, usize)>>,
) -> RenderResult {
    utils::SeededRng::set_seed(config.rng_seed);

    // Starting/seed pitch per voice (high → low). Configurable via the GUI
    // "Start Notes" modal; falls back per index to the historical defaults.
    let start = |voice: usize| -> i32 {
        config
            .start_notes
            .get(voice)
            .copied()
            .unwrap_or(DEFAULT_START_NOTES[voice])
    };

    let contours = Contours::from_config(config);

    let mut income = Vec::new();
    let lead_from_clip = config.use_leading_voice
        && leading.map_or(false, |l| !l.notes.is_empty());
    if lead_from_clip {
        let l = leading.unwrap();
        income.extend(gen_voice_from_notes(&l.notes, l.clip_length, config));
    } else {
        income.extend(gen_voice(start(0), &config.voice_rhythm, &[0], 0, 1, config, &contours));
    }
    for v in 1..5 {
        income.extend(gen_voice(start(v), &config.voice_rhythm, &[0], v as i32, 1, config, &contours));
    }

    // Sort by start time, then pitch descending within a group.
    income.sort_by(|a, b| {
        if (a.start - b.start).abs() > 0.001 {
            a.start.partial_cmp(&b.start).unwrap()
        } else {
            b.pitch.cmp(&a.pitch)
        }
    });

    // Runs the Schillinger progression internally — must stay AFTER the voice
    // generation above so the seeded RNG stream keeps its historical order.
    let state = HarmonizerState::new(config, contours);

    let harmonised = harmonise_explained(income, config, &state, progress);
    let mut notes = harmonised.notes;
    let mut breakdown = harmonised.breakdown;

    shape_dynamics(&mut notes, config);

    for note in &mut notes {
        note.pitch += config.main_pitch;
    }
    // Keep the breakdown in the same pitch space as the notes it explains.
    for g in &mut breakdown {
        if let Some(r) = g.root_pc.as_mut() {
            *r = (*r + config.main_pitch).rem_euclid(12);
        }
        for v in &mut g.voices {
            v.pitch += config.main_pitch;
            if let Some(p) = v.previous_pitch.as_mut() {
                *p += config.main_pitch;
            }
        }
    }

    RenderResult { notes, breakdown, schillinger_notes: state.schillinger_notes }
}

/// Phrase-shaped dynamics (`config.dynamics_weight`): deterministic velocities
/// computed from what the music is doing instead of the legacy random jitter.
///
/// Three ingredients, all already implied by the render's structure:
/// * **Tension arch** — a sine over the whole render plus a sine over each
///   phrase (`pl` bars), so every phrase crescendos toward its middle and
///   relaxes into its cadence, inside one long swell over the render.
/// * **Metric accent** — the inverse of the hold-shaping profile: downbeats
///   full accent, beat 3 next, backbeats lighter, offbeat 16ths none. Note
///   onsets that survive the downstream same-pitch merge are exactly the ones
///   this accents.
/// * **Melodic direction** — rising lines grow, falling lines ease off,
///   per voice.
///
/// A seeded ±3 humanization jitter rides on top so two performances of the
/// same phrase never accent identically — and so the seed keeps mattering on
/// a default render (with the literal progression, the legacy random velocity
/// was the ONLY seed-dependent output; a fully deterministic replacement made
/// every seed render byte-identical).
///
/// `weight` scales the swing around the base level (64): at 1.0 velocities
/// span roughly 55–110. At 0 the legacy random velocities are left untouched.
fn shape_dynamics(notes: &mut [Note], config: &Config) {
    let w = config.dynamics_weight;
    if w == 0.0 {
        return;
    }
    let total = ((config.pl * 4 * config.render_length).max(1)) as f64;
    let phrase = (config.pl.max(1) * 4) as f64;
    let mut prev_by_channel: [Option<i32>; 16] = [None; 16];
    for n in notes.iter_mut() {
        let global = (std::f64::consts::PI * (n.start / total).clamp(0.0, 1.0)).sin();
        let local = (std::f64::consts::PI * (n.start.rem_euclid(phrase) / phrase)).sin();
        let tension = 0.6 * global + 0.4 * local;
        let metric = 1.0 - crate::harmonizer::metric_hold_weight(n.start.rem_euclid(4.0));
        let ch = (n.channel.max(0) as usize).min(15);
        let dir = prev_by_channel[ch]
            .map(|p| ((n.pitch - p).clamp(-4, 4) as f64) / 4.0)
            .unwrap_or(0.0);
        prev_by_channel[ch] = Some(n.pitch);
        let jitter = utils::SeededRng::seeded_random(6.0, -3.0);
        let v = 64.0 + w * (28.0 * tension + 12.0 * metric + 6.0 * dir + jitter);
        n.velocity = (v.round() as i32).clamp(1, 127);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The render pipeline minus the pitch offset noise: notes + progression.
    fn pipeline(config: &Config) -> (Vec<Note>, Vec<Vec<Vec<i32>>>) {
        let r = render(config, None, None);
        (r.notes, r.schillinger_notes)
    }

    fn default_config() -> Config {
        let mut c = Config::default();
        c.init_contours();
        c
    }

    /// Share of bass notes (channel 4) sitting on the bar's chord root.
    fn bass_on_root(notes: &[Note], sch: &[Vec<Vec<i32>>]) -> f64 {
        let bars = &sch[0];
        let mut hits = 0;
        let mut total = 0;
        for n in notes.iter().filter(|n| n.channel == 4) {
            let bar = (n.start / 4.0).floor() as usize % bars.len();
            let root = bars[bar][0].rem_euclid(12);
            total += 1;
            if n.pitch.rem_euclid(12) == root {
                hits += 1;
            }
        }
        hits as f64 / total.max(1) as f64
    }

    /// Golden fingerprint of a default render per seed — guards the pure-core
    /// refactor: any change here means the engine's output changed. Print with:
    ///   cargo test --release golden_fingerprint -- --nocapture
    #[test]
    fn golden_fingerprint() {
        let expected: [(f64, u64); 3] = [
            (1.0, 0x30dc37bb66bdae6b),
            (7.0, 0x23d114d753660393),
            (42.0, 0xfad5ea6604dbaf05),
        ];
        for (seed, want) in expected {
            let mut cfg = default_config();
            cfg.rng_seed = seed;
            let (notes, _) = pipeline(&cfg);
            let mut h = std::collections::hash_map::DefaultHasher::new();
            use std::hash::{Hash, Hasher};
            for n in &notes {
                n.pitch.hash(&mut h);
                n.channel.hash(&mut h);
                ((((n.start) * 10000.0).round()) as i64).hash(&mut h);
                ((((n.duration) * 10000.0).round()) as i64).hash(&mut h);
                n.velocity.hash(&mut h);
            }
            let got = h.finish();
            println!("seed {seed}: {} notes, fingerprint {got:x}", notes.len());
            assert_eq!(got, want, "seed {seed}: render output changed");
        }
    }

    /// Musical read-out of a default render, for tuning by numbers rather than
    /// by ear alone:
    ///   cargo test --release musical_stats -- --ignored --nocapture
    #[test]
    #[ignore]
    fn musical_stats() {
        for w in [0.0, 0.25, 0.5, 1.0, 2.0] {
            let mut c = default_config();
            c.root_position_weight = w;
            let (notes, sch) = pipeline(&c);
            println!("root_position_weight={w:<4} bass on root {:.2}", bass_on_root(&notes, &sch));
        }
        for w in [0.0, 0.5, 1.0, 1.5] {
            println!(
                "tendency_weight={w:<4}     leading tones resolving up {:.2}",
                leading_tone_resolution_rate(w),
            );
        }
    }

    #[test]
    fn breakdown_explains_every_group_and_sums_to_the_score() {
        let cfg = default_config();
        let r = render(&cfg, None, None);
        assert!(!r.breakdown.is_empty());

        // One breakdown entry per distinct chord onset.
        let mut starts: Vec<i64> = r.notes.iter().map(|n| (n.start * 10000.0).round() as i64).collect();
        starts.sort_unstable();
        starts.dedup();
        assert_eq!(r.breakdown.len(), starts.len());

        for g in &r.breakdown {
            assert!(g.scored, "group at {} passed through unscored", g.start);
            // The named terms are the actual score, not an approximation:
            // chord terms + per-voice terms must sum to the soft score.
            let term_sum: f64 = g.chord_terms.iter().map(|t| t.value).sum::<f64>()
                + g.voices.iter().flat_map(|v| v.terms.iter()).map(|t| t.value).sum::<f64>();
            assert!(
                (term_sum - g.soft_score).abs() < 1e-6,
                "bar {}: terms sum {term_sum} vs soft score {}",
                g.bar, g.soft_score,
            );
            assert!(
                (g.score - (g.soft_score - g.hard_violation_count as f64 * 1000.0)).abs() < 1e-9,
                "bar {}: score/soft/hard inconsistent", g.bar,
            );
            // Exactly one voice carries the leader role per chord.
            assert_eq!(g.voices.iter().filter(|v| v.is_leader).count(), 1, "bar {}", g.bar);
            // The breakdown describes the notes that were actually rendered.
            for v in &g.voices {
                assert!(
                    r.notes.iter().any(|n| (n.start - g.start).abs() < 1e-6
                        && n.channel == v.channel
                        && n.pitch == v.pitch),
                    "bar {}: breakdown voice ch{} pitch {} not in the render",
                    g.bar, v.channel, v.pitch,
                );
            }
        }
    }

    #[test]
    fn full_pipeline_runs_and_is_deterministic() {
        let cfg = default_config();
        let (a, _) = pipeline(&cfg);
        assert!(!a.is_empty(), "the pipeline produced no notes");
        for n in &a {
            assert!((0..128).contains(&n.pitch), "pitch {} out of MIDI range", n.pitch);
            assert_eq!(n.muted, 0);
        }
        let (b, _) = pipeline(&cfg);
        let pa: Vec<i32> = a.iter().map(|n| n.pitch).collect();
        let pb: Vec<i32> = b.iter().map(|n| n.pitch).collect();
        assert_eq!(pa, pb, "the same seed must give the same render");
    }

    #[test]
    fn root_position_weight_moves_the_bass_onto_chord_roots() {
        // End-to-end version of the unit test: over a whole default render, the
        // bass sits on the chord root far more often with the preference on.
        let mut on = default_config();
        on.root_position_weight = 2.0;
        let (notes_on, sch) = pipeline(&on);
        let with = bass_on_root(&notes_on, &sch);

        let mut off = default_config();
        off.root_position_weight = 0.0;
        let (notes_off, sch_off) = pipeline(&off);
        let without = bass_on_root(&notes_off, &sch_off);
        let _ = sch_off;

        assert!(
            with > without,
            "bass-on-root {with:.2} with the preference vs {without:.2} without",
        );
    }

    /// Share of leading tones (pc 11 in the default key of C) that resolve up
    /// by semitone to the tonic, over a whole default render.
    fn leading_tone_resolution_rate(tendency_weight: f64) -> f64 {
        let mut c = default_config();
        c.tendency_weight = tendency_weight;
        let (notes, _) = pipeline(&c);
        let mut by_channel: std::collections::HashMap<i32, Vec<&Note>> = std::collections::HashMap::new();
        for n in &notes {
            by_channel.entry(n.channel).or_default().push(n);
        }
        let (mut resolved, mut total) = (0, 0);
        for line in by_channel.values_mut() {
            line.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
            for pair in line.windows(2) {
                if pair[0].pitch.rem_euclid(12) == 11 {
                    total += 1;
                    if pair[1].pitch - pair[0].pitch == 1 {
                        resolved += 1;
                    }
                }
            }
        }
        resolved as f64 / total.max(1) as f64
    }

    #[test]
    fn tendency_weight_makes_leading_tones_resolve() {
        // Without the term the leading tone is just another scale degree and
        // resolves by accident (~6%); with it, a third of them step up to the
        // tonic — the rest are bars where the next chord has no tonic to land
        // on, or where crossing/budget constraints outrank it.
        let off = leading_tone_resolution_rate(0.0);
        let on = leading_tone_resolution_rate(1.5);
        assert!(on > off + 0.15, "resolution rate {off:.2} → {on:.2} is not a real effect");
    }

    #[test]
    fn generated_progression_renders_end_to_end() {
        let mut cfg = default_config();
        cfg.use_generated_progression = true;
        let (notes, sch) = pipeline(&cfg);
        assert!(!notes.is_empty());
        assert_eq!(sch[0].len(), (cfg.pl * cfg.render_length) as usize);
    }

    #[test]
    fn leading_clip_replaces_voice_zero() {
        let mut cfg = default_config();
        cfg.use_leading_voice = true;
        let leading = Leading {
            notes: vec![Note::new(72, 0.0, 2.0, 80, 0, 0), Note::new(74, 2.0, 2.0, 80, 0, 0)],
            clip_length: 4.0,
        };
        let r = render(&cfg, Some(&leading), None);
        assert!(!r.notes.is_empty());
        // With an empty clip the generated voice 0 is used instead.
        let r2 = render(&cfg, Some(&Leading { notes: vec![], clip_length: 4.0 }), None);
        assert!(!r2.notes.is_empty());
    }

    /// The pitch a channel is sounding at `t`, if any (notes never cross bar
    /// lines, so this is well-defined within a render).
    fn pitch_at(notes: &[Note], channel: i32, t: f64) -> Option<i32> {
        notes.iter()
            .find(|n| n.channel == channel && n.start <= t + 1e-6 && n.start + n.duration > t + 1e-6)
            .map(|n| n.pitch)
    }

    #[test]
    fn dynamics_shape_velocities_by_phrase_and_meter() {
        // Shaped velocities live in a musical band and accent strong beats.
        let cfg = default_config();
        let (notes, _) = pipeline(&cfg);
        assert!(notes.iter().all(|n| (1..=127).contains(&n.velocity)));
        let mean = |pred: &dyn Fn(&Note) -> bool| -> f64 {
            let v: Vec<f64> = notes.iter().filter(|n| pred(n)).map(|n| n.velocity as f64).collect();
            v.iter().sum::<f64>() / v.len().max(1) as f64
        };
        let downbeats = mean(&|n| n.start.rem_euclid(4.0) < 0.001);
        let offbeats = mean(&|n| {
            let q = n.start.rem_euclid(1.0);
            q > 0.2 && q < 0.8
        });
        assert!(
            downbeats > offbeats + 4.0,
            "downbeat mean {downbeats:.1} should clearly exceed offbeat mean {offbeats:.1}",
        );

        // Weight 0 restores the legacy random velocities (small values).
        let mut legacy = default_config();
        legacy.dynamics_weight = 0.0;
        let (l, _) = pipeline(&legacy);
        assert!(
            l.iter().all(|n| n.velocity < 45),
            "legacy velocities should be untouched at weight 0",
        );
    }

    #[test]
    fn cadence_weight_lands_the_phrase_final_tonic() {
        // Cadence quality of each phrase-final downbeat: bass on the tonic,
        // third present, soprano on tonic/third. Higher weight, better closes.
        let quality = |w: f64| -> i32 {
            let mut cfg = default_config();
            cfg.use_generated_progression = true;
            cfg.render_length = 3;
            cfg.cadence_weight = w;
            let (notes, sch) = pipeline(&cfg);
            let bars = sch[0].len();
            let mut q = 0;
            for bar in (0..bars).filter(|b| (b + 1) % cfg.pl as usize == 0) {
                let t = bar as f64 * 4.0;
                let root = sch[0][bar][0].rem_euclid(12);
                let chord: Vec<&Note> = notes
                    .iter()
                    .filter(|n| (n.start - t).abs() < 1e-6)
                    .collect();
                let pcs: Vec<i32> = chord.iter().map(|n| n.pitch.rem_euclid(12)).collect();
                let bass = chord.iter().map(|n| n.pitch).min().unwrap_or(0);
                let top = chord.iter().map(|n| n.pitch).max().unwrap_or(0);
                if bass.rem_euclid(12) == root {
                    q += 1;
                }
                if pcs.contains(&(root + 3).rem_euclid(12)) || pcs.contains(&(root + 4).rem_euclid(12)) {
                    q += 1;
                }
                let sop = (top.rem_euclid(12) - root).rem_euclid(12);
                if sop == 0 || sop == 3 || sop == 4 {
                    q += 1;
                }
            }
            q
        };
        let with = quality(2.0);
        let without = quality(0.0);
        println!("cadence quality: {without} -> {with}");
        assert!(
            with > without,
            "cadence quality {without} → {with} should improve with the weight",
        );
    }

    #[test]
    fn phrase_echo_makes_phrases_rhyme() {
        // Similarity between consecutive phrases: sample every 8th-note position
        // of phrase 0 and phrase 1 and count positions where a voice repeats the
        // same melodic move (delta from the previous sample). With the echo
        // term on, the second phrase should rhyme with the first more often.
        let similarity = |w: f64| -> f64 {
            let mut cfg = default_config();
            cfg.phrase_echo_weight = w;
            let (notes, _) = pipeline(&cfg);
            let phrase = (cfg.pl * 4) as f64;
            let (mut matches, mut total) = (0, 0);
            for ch in 0..5 {
                let mut t = 0.5;
                while t < phrase {
                    let d0 = match (pitch_at(&notes, ch, t), pitch_at(&notes, ch, t - 0.5)) {
                        (Some(a), Some(b)) => Some(a - b),
                        _ => None,
                    };
                    let d1 = match (pitch_at(&notes, ch, t + phrase), pitch_at(&notes, ch, t + phrase - 0.5)) {
                        (Some(a), Some(b)) => Some(a - b),
                        _ => None,
                    };
                    if let (Some(a), Some(b)) = (d0, d1) {
                        total += 1;
                        if a == b {
                            matches += 1;
                        }
                    }
                    t += 0.5;
                }
            }
            matches as f64 / total.max(1) as f64
        };
        let with = similarity(2.0);
        let without = similarity(0.0);
        println!("phrase similarity: {without:.2} -> {with:.2}");
        assert!(
            with > without,
            "phrase similarity {without:.2} → {with:.2} should rise with the echo",
        );
    }

    #[test]
    fn loop_wrap_weight_smooths_the_seam() {
        // Seam cost: per voice, semitone distance from the last sounding pitch
        // back to the first — what the ear crosses when the clip loops.
        let seam = |w: f64| -> i32 {
            let mut cfg = default_config();
            cfg.loop_wrap_weight = w;
            let (notes, _) = pipeline(&cfg);
            (0..5)
                .map(|ch| {
                    let line: Vec<&Note> = {
                        let mut l: Vec<&Note> =
                            notes.iter().filter(|n| n.channel == ch).collect();
                        l.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
                        l
                    };
                    (line.last().unwrap().pitch - line.first().unwrap().pitch).abs()
                })
                .sum()
        };
        let with = seam(2.0);
        let without = seam(0.0);
        println!("seam distance: {without} -> {with}");
        assert!(
            with < without,
            "total seam distance {without} → {with} should shrink with the wrap term",
        );
    }

    #[test]
    fn rhythm_generators_drive_the_attack_pattern() {
        // r(4÷3) at 8th-note units: durations 1.5,0.5,1.0,1.0,0.5,1.5 beats,
        // span 6 beats — attacks at 0, 1.5, 2, 3, 4, 4.5, 6, ... per voice.
        let mut cfg = default_config();
        cfg.rhythm_generators = vec![4, 3];
        cfg.rhythm_unit = 0.5;
        cfg.rhythm_voice_rotation = 0;
        let (notes, _) = pipeline(&cfg);

        let pattern = [1.5, 0.5, 1.0, 1.0, 0.5, 1.5];
        let span = 6.0;
        for ch in 0..5 {
            let line: Vec<&Note> = {
                let mut l: Vec<&Note> = notes.iter().filter(|n| n.channel == ch).collect();
                l.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
                l
            };
            // Continuous coverage: each note ends where the next begins (ties
            // across bar lines are separate notes, so no gaps and no overlaps).
            for pair in line.windows(2) {
                assert!(
                    (pair[0].start + pair[0].duration - pair[1].start).abs() < 1e-6,
                    "ch{ch}: gap/overlap at {}",
                    pair[1].start,
                );
            }
            // Every pattern attack (cumulative r(4÷3) position) has a note
            // starting exactly there.
            let mut t = 0.0;
            let mut k = 0;
            while t < 24.0 {
                assert!(
                    line.iter().any(|n| (n.start - t).abs() < 1e-6),
                    "ch{ch}: no onset at pattern attack {t}",
                );
                t += pattern[k % pattern.len()];
                k += 1;
            }
            let _ = span;
        }

        // Rotation phase-shifts the cycle per voice: with rotation 1, channel 1
        // starts on the pattern's second element (0.5 beats), channel 0 on its
        // first (1.5) — their first-note durations must differ accordingly.
        let mut rot = default_config();
        rot.rhythm_generators = vec![4, 3];
        rot.rhythm_unit = 0.5;
        rot.rhythm_voice_rotation = 1;
        let (rn, _) = pipeline(&rot);
        let first_dur = |ch: i32| {
            rn.iter()
                .filter(|n| n.channel == ch)
                .min_by(|a, b| a.start.partial_cmp(&b.start).unwrap())
                .map(|n| n.duration)
                .unwrap()
        };
        assert!((first_dur(0) - 1.5).abs() < 1e-6, "ch0 should open with 1.5, got {}", first_dur(0));
        assert!((first_dur(1) - 0.5).abs() < 1e-6, "ch1 should open rotated with 0.5, got {}", first_dur(1));
    }

    /// TEMP repro: run the full generation pipeline with a config supplied via
    /// REPRO_CONFIG and dump the first chords vs the Schillinger scale per bar:
    ///   REPRO_CONFIG=path cargo test repro_first_chord_scale -- --ignored --nocapture
    #[test]
    #[ignore]
    fn repro_first_chord_scale() {
        let path = std::env::var("REPRO_CONFIG").expect("set REPRO_CONFIG");
        let config: Config =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let r = render(&config, None, None);
        let notes = r.notes;
        let schillinger_notes = r.schillinger_notes;

        for (v, bars) in schillinger_notes.iter().enumerate() {
            println!("voice {v}: bar0={:?} bar1={:?}", bars.get(0), bars.get(1));
        }

        let mut starts: Vec<f64> = notes.iter().map(|n| n.start).collect();
        starts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        starts.dedup_by(|a, b| (*a - *b).abs() < 0.001);
        for &s in starts.iter().take(6) {
            let bar = (s / 4.0).floor() as usize;
            let mut group: Vec<&Note> = notes.iter()
                .filter(|n| (n.start - s).abs() < 0.001)
                .collect();
            group.sort_by_key(|n| n.channel);
            println!("--- start {s} (bar {bar}) ---");
            for n in group {
                let voice_bars = &schillinger_notes[(n.channel as usize) % schillinger_notes.len()];
                let scale = &voice_bars[bar % voice_bars.len()];
                let in_scale = scale.iter().any(|&p| (p % 12 + 12) % 12 == (n.pitch % 12 + 12) % 12);
                println!(
                    "ch {} pitch {} pc {} scale {:?} in_scale {}",
                    n.channel, n.pitch, (n.pitch % 12 + 12) % 12, scale, in_scale
                );
            }
        }
    }
}
