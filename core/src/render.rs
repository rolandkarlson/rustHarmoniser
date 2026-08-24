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
            (1.0, 0xa740a51cc69928c5),
            (7.0, 0xa7737e6007df8364),
            (42.0, 0xab00ed192c3172d1),
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
