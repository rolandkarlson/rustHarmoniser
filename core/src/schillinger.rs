use crate::contour::Contours;
use crate::utils::{mod_shim, SeededRng};
use crate::music_theory::{generate_mode_from_steps};
use crate::model::Config;

// PL was 8 in JS, now passed via Config
const EXP: i32 = 2;

/// Penultimate degree of each mode's characteristic cadence (the caller
/// appends the tonic). Indexed by mode 0=Ionian … 6=Locrian.
///
/// One hardcoded V → I used to be forced on every mode, which was
/// Ionian-centric twice over: in Phrygian and Locrian degree 4 is never a
/// TARGET in the transition table, so the random walk could not reach it —
/// every phrase burned the full attempt budget and then had a diminished v°
/// forced on it as a "dominant". Each mode now closes with its own idiom:
/// * Ionian / Lydian: V → I (authentic; both have a major V).
/// * Dorian: IV → i (the modal plagal close its table is built around).
/// * Phrygian / Locrian: ♭II → i (the table's own "characteristic chord").
/// * Mixolydian: ♭VII → I.
/// * Aeolian: V → i — gen_schillinger_progression applies the harmonic-minor
///   inflection on dominant-rooted bars (the chord's subtonic is raised to a
///   leading tone), so the close is a true authentic cadence rather than the
///   modal v → i of natural minor.
const CADENCE_DEGREE: [i32; 7] = [4, 3, 1, 4, 6, 4, 1];

/// The scale-degree transition table for `mode` (0 = Ionian … 6 = Locrian):
/// row d lists the degrees a chord on degree d may move to.
fn mode_transitions(mode: i32) -> [&'static [i32]; 7] {
    let ionian_transitions: [&[i32]; 7] = [&[3, 4, 5, 1, 2], &[4, 6], &[5, 3], &[0, 1, 4], &[0, 5], &[3, 1, 4], &[0]];

    // 2. DORIAN - Focus on i (0) and IV (3)
    let dorian_transitions: [&[i32]; 7] = [&[3, 6, 1], &[0, 3], &[3, 4], &[0, 6], &[0, 3], &[6, 3], &[0, 3]];

    // 3. PHRYGIAN (0=i, 1=II, 2=III, 3=iv, 4=v°, 5=VI, 6=vii)
    // The flat-2nd creates a major II chord. The i -> II -> i movement is the core Phrygian sound.
    let phrygian_transitions: [&[i32]; 7] = [
        &[1, 3, 5],       // i   -> Heavily favors sliding up to II, or going to iv or VI
        &[0],             // II  -> THE characteristic chord. Slams straight back to i
        &[1, 3],          // III -> Moves to II or iv
        &[0, 1],          // iv  -> Resolves to i, or steps to II
        &[1, 5],          // v°  -> Diminished. Pushes to II or VI
        &[1, 0],          // VI  -> Moves to II or i
        &[0, 2],          // vii -> Steps to i or III
    ];

    // 4. LYDIAN (0=I, 1=II, 2=iii, 3=iv°, 4=V, 5=vi, 6=vii)
    // The sharp-4th creates a major II chord. Bouncing between I and II is the classic Lydian float.
    let lydian_transitions: [&[i32]; 7] = [
        &[1, 4, 2],       // I   -> Favors moving to II, V, or iii
        &[0, 4],          // II  -> THE characteristic chord. Resolves back to I, or moves to V
        &[0, 1],          // iii -> Moves to I or II
        &[4, 2],          // iv° -> Diminished. Pushes to V or iii
        &[0, 1],          // V   -> Resolves to I or steps to II
        &[1, 4],          // vi  -> Pushes to II or V
        &[0, 2],          // vii -> Resolves to I or iii
    ];

    // 5. MIXOLYDIAN - Focus on I (0), IV (3), and VII (6)
    let mixolydian_transitions: [&[i32]; 7] = [&[3, 6, 4], &[0, 3], &[3, 5], &[0, 6], &[0, 3], &[3, 1], &[0, 3]];

    // 6. AEOLIAN (Minor) - Uses borrowed V for strong resolution
    let aeolian_transitions: [&[i32]; 7] = [&[2, 3, 4, 5, 6], &[4, 6], &[5, 3], &[0, 1, 4, 6], &[0, 5], &[1, 3, 4], &[2, 0]];

    // 7. LOCRIAN (0=i°, 1=II, 2=iii, 3=iv, 4=V, 5=VI, 6=vii)
    // The root chord is diminished, making this mode inherently unstable and dark. It rarely resolves cleanly.
    let locrian_transitions: [&[i32]; 7] = [
        &[1, 3, 5],       // i°  -> Diminished root! Pushes away to II, iv, or VI
        &[0, 3],          // II  -> Steps back to i°, or moves to iv
        &[1, 5],          // iii -> Pushes to II or VI
        &[0, 1],          // iv  -> Moves back to i° or II
        &[1, 5],          // V   -> Pushes to II or VI
        &[0, 1],          // VI  -> Moves to i° or II
        &[0, 5],          // vii -> Pushes to i° or VI
    ];

    match mode {
        0 => ionian_transitions,
        1 => dorian_transitions,
        2 => phrygian_transitions,
        3 => lydian_transitions,
        4 => mixolydian_transitions,
        5 => aeolian_transitions,
        6 => locrian_transitions,
        _ => panic!("Invalid mode"),
    }
}

/// A random walk of `length` scale-degree chord roots through the transition
/// table of `mode`, starting on the tonic and — where the walk allows —
/// finishing on that mode's cadence degree (see CADENCE_DEGREE) so the caller
/// can append the tonic to complete the cadence. If no walk of that length
/// lands there within the attempt budget, the last chord is forced instead.
pub fn generate_progression(length: usize, mode: i32) -> Vec<i32> {
    if length == 0 {
        return vec![];
    }
    let transitions = mode_transitions(mode);
    let cadence = CADENCE_DEGREE[mod_shim(mode, 7) as usize];
    let max_attempts = 1000;

    for _ in 0..max_attempts {
        let mut progression = Vec::with_capacity(length);
        let mut current_chord = 0;
        progression.push(current_chord);

        for _ in 1..length {
            let possible_next_chords = transitions[current_chord as usize];
            let random_index = SeededRng::random_int(possible_next_chords.len() as i32) as usize;
            current_chord = possible_next_chords[random_index];
            progression.push(current_chord);
        }

        if *progression.last().unwrap() == cadence {
            return progression;
        }
    }

    // Fallback if the walk never lands on the cadence degree by itself (e.g.
    // very short phrases): force the final chord rather than returning an
    // uncadenced walk.
    let mut progression = Vec::with_capacity(length);
    let mut current_chord = 0;
    progression.push(current_chord);

    for i in 1..length {
        if i == length - 1 {
            progression.push(cadence);
            continue;
        }
        let possible_next_chords = transitions[current_chord as usize];
        let random_index = SeededRng::random_int(possible_next_chords.len() as i32) as usize;
        current_chord = possible_next_chords[random_index];
        progression.push(current_chord);
    }

    progression
}

fn find_sequence_with_condition(possible_steps: &[i32], sequence_length: i32) -> Option<Vec<i32>> {
    let max_attempts = 1000000;
    let mut attempts = 0;

    while attempts < max_attempts {
        let mut sequence = vec![0];
        let mut current_sum = 0;
        let mut last_value = -999;

        for _ in 1..sequence_length {
            let random_index = (SeededRng::seeded_random(1.0, 0.0) * possible_steps.len() as f64).floor() as usize;
            let mut step = possible_steps[random_index];

            if last_value + step == 0 {
                step *= -1;
            }
            last_value = step;
            current_sum += step;
            
            if last_value + step == 0 {
                step *= -1;
            }
            // Logic in JS for `step *= -1` appears TWICE.
            // JS:
            // if (lastValue + step === 0) { step *= -1; }
            // lastValue = step;
            // currentSum += step;
            // if (lastValue + step === 0) { step *= -1; } -> This checks step against itself? lastValue is step.
            // wait, lastValue is updated to step. so `step + step == 0`? No.
            // JS: `lastValue = step; ... if (lastValue + step === 0)` -> checks `step + step === 0` -> `2*step === 0`.
            // only if step is 0 (which it isn't, -2 etc * EXP).
            // So the second check is redundant or logic error in original JS, but I must replicate strictly if I want exact same behavior?
            // Actually, `currentSum` is updated.
            // JS:
            /*
            if (lastValue + step === 0) {
                step *= -1;
            }
            lastValue = step;
            // 2. Calculate the new sum
            currentSum += step;

            if (lastValue + step === 0) {
                step *= -1;
            }
            // 3. Add the new sum to the sequence
            sequence.push(currentSum);
            */
            // Since `lastValue = step`, `lastValue + step` is `2*step`.
            // The second check is effectively never true unless step=0.
            // I will copy it verbatim to be safe.
            
            sequence.push(mod_shim(current_sum, 7));
        }

        let last_element = sequence.last().unwrap();
        let modulo_result = mod_shim(*last_element, 7);
        // JS: `var moduloResult = ((lastElement % 7) + 7) % 7;`
        
        let target = mod_shim(4, 7);

        if modulo_result == target {
            return Some(sequence);
        }
        attempts += 1;
    }
    None
}

pub const NUM_VOICES: usize = 16;

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Schillinger rhythm resultant r(a÷b[÷c…]) — Book I's interference pattern of
/// uniform periodicities. Each generator lays attacks at its own multiples;
/// the resultant is the union of all attack points over one common span (the
/// generators' LCM), read back as the durations between consecutive attacks:
///
///   r(3÷2) = 2+1+1+2      r(4÷3) = 3+1+2+2+1+3      r(5÷4) = 4+1+3+2+2+3+1+4
///
/// Every two-generator resultant is a palindrome and sums to the span, which
/// is why the pattern loops seamlessly. Non-positive generators are ignored;
/// no usable generator yields an empty pattern (caller treats that as "off").
pub fn resultant(gens: &[i32]) -> Vec<i32> {
    let gens: Vec<i64> = gens.iter().filter(|&&g| g > 0).map(|&g| g as i64).collect();
    if gens.is_empty() {
        return Vec::new();
    }
    let span = gens.iter().fold(1i64, |acc, &g| acc / gcd(acc, g) * g);
    let mut points: Vec<i64> = vec![0];
    for &g in &gens {
        let mut t = g;
        while t <= span {
            points.push(t);
            t += g;
        }
    }
    points.sort_unstable();
    points.dedup();
    points.windows(2).map(|w| (w[1] - w[0]) as i32).collect()
}

/// Phrase-structured chord-root sequence: `render_length` phrases of `pl` bars,
/// each a mode-aware random walk that closes with the mode's own cadence
/// (CADENCE_DEGREE → tonic), so every phrase ends on a cadence instead of
/// wherever the loop happened to stop.
///
/// Uses the scalar `config.mode` — the per-bar `mode_contour` still colours the
/// scale each bar is realised in, but the transition table is chosen once.
pub fn gen_cadenced_progression(config: &Config) -> Vec<i32> {
    let phrase = config.pl.max(2) as usize;
    let phrases = config.render_length.max(1) as usize;
    let mode = mod_shim(config.mode, 7);
    let mut out = Vec::with_capacity(phrase * phrases);
    for _ in 0..phrases {
        // The body ends on the dominant; the appended tonic completes the cadence.
        let mut p = generate_progression(phrase - 1, mode);
        p.push(0);
        out.extend(p);
    }
    out
}

pub fn gen_schillinger_progression(config: &Config, contours: &Contours) -> Vec<Vec<Vec<i32>>> {

    let generated;
    let seq: &Vec<i32> = if config.use_generated_progression {
        generated = gen_cadenced_progression(config);
        &generated
    } else if config.schillinger_sequence.is_empty() {
        generated = vec![0];
        &generated
    } else {
        &config.schillinger_sequence
    };
    let bars = seq.len();

    let chord_list = vec![
        vec![0,1,2],
        vec![0,1,2,4,5],
        vec![0,1,2,4,5],
        vec![0,1,2,3,4],
        vec![0,1,2,3,4,5],
        vec![0,1,2,3,4,5,6]
    ];

    let mut per_voice: Vec<Vec<Vec<i32>>> = Vec::with_capacity(NUM_VOICES);

    for voice in 0..NUM_VOICES {
        let mut chord_notes = Vec::with_capacity(bars);

        for i in 0..bars {
            let start_time = i as f64 * 4.0;

            let current_mode = contours.mode.as_ref()
                .map(|c| mod_shim(c.at(start_time).round() as i32, 7))
                .unwrap_or(config.mode);

            let current_root = contours.root.as_ref()
                .map(|c| mod_shim(c.at(start_time).round() as i32, 12))
                .unwrap_or(config.root);

            let chord_idx = contours.chord_structure.as_ref()
                .and_then(|vc| vc.at_strict(voice, start_time))
                .map(|v| v.round() as usize)
                .unwrap_or(0);

            let n_struct = if chord_idx < chord_list.len() {
                &chord_list[chord_idx]
            } else {
                &config.chord_structure
            };

            let scale = generate_mode_from_steps(current_root, &current_mode);
            let ex = contours.schillinger_ex.as_ref()
                .and_then(|vc| vc.at_strict(voice, start_time))
                .map(|v| v.round() as i32)
                .unwrap_or(2);
            // Harmonic-minor inflection: in Aeolian, a chord rooted on the
            // dominant raises any subtonic (degree 7) it contains by a
            // semitone, turning minor v into major V with a true leading
            // tone. This is what makes the phrase-final v → i an actual
            // authentic cadence — and it lets the harmonizer's tendency-tone
            // reward and leading-tone doubling penalty fire in minor, since
            // both key on "a semitone below the tonic", a pitch class natural
            // minor otherwise never produces. Non-dominant bars keep the
            // natural subtonic (♭VII stays modal mid-phrase).
            let degree = seq[i as usize % seq.len()];
            let raise_leading_tone = current_mode == 5 && mod_shim(degree, 7) == 4;
            let notes: Vec<i32> = n_struct.iter().map(|&itm| {
                 let idx = mod_shim((itm * ex) + degree, scale.len() as i32) as usize;
                 let pc = scale[idx];
                 if raise_leading_tone && idx == 6 {
                     (pc + 1).rem_euclid(12)
                 } else {
                     pc
                 }
            }).collect();

            chord_notes.push(notes);
        }

        per_voice.push(chord_notes);
    }

    per_voice
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(pl: i32, render_length: i32, mode: i32) -> Config {
        Config { pl, render_length, mode, use_generated_progression: true, ..Config::default() }
    }

    fn prog(c: &Config) -> Vec<Vec<Vec<i32>>> {
        gen_schillinger_progression(c, &Contours::from_config(c))
    }

    #[test]
    fn resultants_match_the_classic_patterns() {
        assert_eq!(resultant(&[3, 2]), vec![2, 1, 1, 2]);
        assert_eq!(resultant(&[4, 3]), vec![3, 1, 2, 2, 1, 3]);
        assert_eq!(resultant(&[5, 4]), vec![4, 1, 3, 2, 2, 3, 1, 4]);
        assert_eq!(resultant(&[5, 3]), vec![3, 2, 1, 3, 1, 2, 3]);
        // Order and duplicates don't matter; a lone generator is a flat pulse.
        assert_eq!(resultant(&[2, 3]), resultant(&[3, 2]));
        assert_eq!(resultant(&[3, 3]), vec![3]);
        assert_eq!(resultant(&[4]), vec![4]);
    }

    #[test]
    fn resultants_are_palindromic_and_span_the_lcm() {
        for (a, b) in [(3, 2), (4, 3), (5, 2), (5, 3), (5, 4), (7, 4), (9, 5)] {
            let r = resultant(&[a, b]);
            let rev: Vec<i32> = r.iter().rev().copied().collect();
            assert_eq!(r, rev, "r({a}÷{b}) is not a palindrome: {r:?}");
            let lcm = (a * b) / {
                let (mut x, mut y) = (a, b);
                while y != 0 { let t = y; y = x % y; x = t; }
                x
            };
            assert_eq!(r.iter().sum::<i32>(), lcm, "r({a}÷{b}) does not span the lcm");
            // All pairs above are coprime: they interfere a+b-1 times per span.
            assert_eq!(r.len() as i32, a + b - 1, "r({a}÷{b}) has the wrong attack count");
        }
    }

    #[test]
    fn degenerate_generators_are_ignored() {
        assert!(resultant(&[]).is_empty());
        assert!(resultant(&[0, -3]).is_empty());
        assert_eq!(resultant(&[0, 3, 2]), vec![2, 1, 1, 2]);
    }

    #[test]
    fn three_generator_resultant_interferes_all_three() {
        // r(5÷4÷3): span 60, attacks at every multiple of 5, 4 and 3.
        let r = resultant(&[5, 4, 3]);
        assert_eq!(r.iter().sum::<i32>(), 60);
        let mut t = 0;
        let mut points = vec![0];
        for d in &r {
            t += d;
            points.push(t);
        }
        for g in [5, 4, 3] {
            for k in (g..=60).step_by(g as usize) {
                assert!(points.contains(&k), "r(5÷4÷3) misses attack {k} of generator {g}");
            }
        }
    }

    #[test]
    fn cadenced_progression_has_one_phrase_per_render_length() {
        SeededRng::set_seed(1.0);
        for (pl, rl) in [(4, 2), (8, 1), (2, 3), (4, 8)] {
            let p = gen_cadenced_progression(&cfg(pl, rl, 0));
            assert_eq!(p.len(), (pl * rl) as usize, "pl {pl} render_length {rl}");
        }
    }

    #[test]
    fn every_phrase_starts_on_the_tonic_and_ends_with_its_modes_cadence() {
        SeededRng::set_seed(7.0);
        for mode in 0..7 {
            let pl = 4;
            let cadence = CADENCE_DEGREE[mode as usize];
            let p = gen_cadenced_progression(&cfg(pl, 3, mode));
            for (i, phrase) in p.chunks(pl as usize).enumerate() {
                assert_eq!(phrase[0], 0, "mode {mode} phrase {i} does not open on I");
                assert_eq!(
                    &phrase[phrase.len() - 2..],
                    &[cadence, 0],
                    "mode {mode} phrase {i} does not close {cadence} → 0: {phrase:?}",
                );
            }
        }
    }

    #[test]
    fn cadence_degree_is_reachable_in_every_modes_transition_table() {
        // Guards the defect this mechanism replaced: the old hardcoded V → I
        // target was never a transition TARGET in Phrygian or Locrian, so the
        // walk burned its whole attempt budget on every phrase and then
        // force-appended a chord the mode's own grammar could not reach. The
        // cadence degree must appear as a target somewhere in the table, so
        // the walk can land on it organically and the fallback stays what it
        // is meant to be — a rarity for very short phrases, not the routine
        // outcome. (The end-to-end cadence shape itself is asserted above.)
        for mode in 0..7 {
            let cadence = CADENCE_DEGREE[mode as usize];
            let reachable = mode_transitions(mode)
                .iter()
                .any(|targets| targets.contains(&cadence));
            assert!(
                reachable,
                "mode {mode}: cadence degree {cadence} is not a target anywhere in its transition table",
            );
        }
    }

    #[test]
    fn cadenced_progression_is_deterministic_under_a_seed() {
        SeededRng::set_seed(42.0);
        let a = gen_cadenced_progression(&cfg(4, 4, 5));
        SeededRng::set_seed(42.0);
        let b = gen_cadenced_progression(&cfg(4, 4, 5));
        assert_eq!(a, b);
    }

    #[test]
    fn generated_progression_overrides_the_literal_sequence() {
        SeededRng::set_seed(3.0);
        let mut c = cfg(4, 2, 0);
        c.schillinger_sequence = vec![0, 0]; // would otherwise give 2 bars
        assert_eq!(prog(&c)[0].len(), 8);

        c.use_generated_progression = false;
        assert_eq!(prog(&c)[0].len(), 2);
    }

    #[test]
    fn empty_sequence_does_not_panic() {
        let mut c = cfg(4, 1, 0);
        c.use_generated_progression = false;
        c.schillinger_sequence = Vec::new();
        assert_eq!(prog(&c)[0].len(), 1);
    }

    #[test]
    fn aeolian_dominant_bars_raise_the_leading_tone() {
        // C Aeolian (root 0, mode 5): scale pcs [0,2,3,5,7,8,10]. The default
        // triad on degree 4 stacks degrees {4,6,1} → pcs {7,10,2}; the
        // harmonic-minor inflection must lift the subtonic 10 to the leading
        // tone 11 on that bar ONLY, leaving mid-phrase bars in natural minor.
        let mut c = cfg(4, 1, 5);
        c.use_generated_progression = false;
        c.schillinger_sequence = vec![0, 3, 4, 0];
        c.root = 0;
        let bars = &prog(&c)[0];

        assert!(bars[2].contains(&11), "dominant bar lacks the leading tone: {:?}", bars[2]);
        assert!(!bars[2].contains(&10), "dominant bar kept the subtonic: {:?}", bars[2]);
        // The chord root is untouched — bar_root_pc's invariant that notes[0]
        // IS the root must survive the inflection.
        assert_eq!(bars[2][0], 7, "dominant bar root moved: {:?}", bars[2]);
        // Every other bar stays natural minor: no leading tone anywhere else.
        for (i, bar) in bars.iter().enumerate() {
            if i != 2 {
                assert!(!bar.contains(&11), "bar {i} has a raised 7th: {bar:?}");
            }
        }
    }

    #[test]
    fn other_modes_never_raise_the_seventh() {
        // The inflection is an Aeolian-only convention: a degree-4 bar in any
        // other mode keeps its diatonic pitch classes exactly.
        for mode in [0, 1, 2, 3, 4, 6] {
            let mut c = cfg(4, 1, mode);
            c.use_generated_progression = false;
            c.schillinger_sequence = vec![4];
            c.root = 0;
            let scale = generate_mode_from_steps(0, &mode);
            let bar = &prog(&c)[0][0];
            for pc in bar {
                assert!(scale.contains(pc), "mode {mode}: pc {pc} is off-scale in {bar:?}");
            }
        }
    }

    #[test]
    fn root_contour_transposes_the_scale_per_bar() {
        // Two bars of the tonic triad, root contour C → G at the default 4-beat
        // resolution: bar 0 realises in C Ionian, bar 1 in G Ionian.
        let mut c = cfg(4, 1, 0);
        c.use_generated_progression = false;
        c.schillinger_sequence = vec![0, 0];
        c.root = 0;
        c.root_contour = Some(vec![0.0, 7.0]);
        let bars = &prog(&c)[0];
        assert_eq!(bars[0], vec![0, 4, 7], "bar 0 should stay in C");
        assert_eq!(bars[1], vec![7, 11, 2], "bar 1 should modulate to G");
    }

    #[test]
    fn chord_roots_track_the_progression_degrees() {
        // The first note of each bar's chord is the degree the progression
        // named — this is the invariant harmonizer::bar_root_pc relies on.
        SeededRng::set_seed(11.0);
        let mut c = cfg(4, 1, 0);
        c.use_generated_progression = false;
        c.schillinger_sequence = vec![0, 3, 4, 0];
        c.root = 2; // D ionian
        let scale = generate_mode_from_steps(2, &0);
        let bars = &prog(&c)[0];
        for (i, deg) in c.schillinger_sequence.iter().enumerate() {
            assert_eq!(bars[i][0], scale[*deg as usize], "bar {i}");
        }
    }
}
