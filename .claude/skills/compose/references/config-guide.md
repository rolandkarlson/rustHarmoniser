# rustHarmoniser config guide (for the compose skill)

Everything is one JSON `Config` (see `core/src/model.rs` for authoritative
docs). Fetch defaults from `GET /api/config` and patch. Scoring terms live on a
±1 scale — a weight of 1.0 is "as important as consonance", 2–3 is dominant.

## Structure & key

| Field | Meaning |
|---|---|
| `pl` | bars per phrase — cadences land every `pl` bars |
| `render_length` | number of phrases; total bars = `pl × render_length`, total beats = ×4 |
| `mode` | 0 Ionian, 1 Dorian, 2 Phrygian, 3 Lydian, 4 Mixolydian, 5 Aeolian, 6 Locrian |
| `root` | tonic pitch class 0–11 |
| `main_pitch` | semitone transpose applied to the finished render |
| `use_generated_progression` | true = mode-aware random walk closing each phrase with that mode's cadence (Ionian/Lydian V→I, Dorian IV→i, Phrygian/Locrian ♭II→i, Mixolydian ♭VII→I, Aeolian harmonic-minor V→i) |
| `schillinger_sequence` | literal per-bar scale-degree roots (0-based) when not generating; loops via mod |
| `start_notes` | 5 seed pitches high→low (default 70,65,60,50,34). Keep them in-key; a voice seeded off-scale can get stuck holding it |
| `rng_seed` | any float; same config+seed = identical render |

Loop clips: `loop_wrap_weight` 1.0–1.5 voices the final bar to lead back into
the opening (seam smoothness + tendency across the wrap). Ending the sequence
on the dominant makes the wrap into the downbeat tonic feel inevitable.

## Harmony character

- `harmony_matrix_contour` — style row over time: 0 Strict Classical, 1 Jazz,
  2 Tension (tritones/leading tones prized, triads dull), 3 Ethereal/quartal,
  4 Dark/minor, 5 Bright/Lydian, 6 Aggressive, 7 Ancient organum, 8 Neutral.
  Fractional values LERP through *adjacent* rows (0→2 passes through Jazz);
  step the contour if you don't want the midpoint color. Put row 2 on
  dominant bars and return to 0 on tonic arrivals — row 2 actively dislikes
  the resolution triad.
- `harmony_distance_contour` — harmony-vs-smoothness balance r (weights are
  0.5±r): raise toward 0.4–0.5 where consonance must win (cadences), lower
  toward 0–0.15 where smooth lines matter more.
- `roughness_weight` (0–1) — blend of style matrix vs register-aware
  psychoacoustic roughness. Higher = cleaner low-register spacing.
- `chord_quality_weight` — lets the style row steer major/minor chord color.
- `root_position_weight`, `root_doubling_weight` — classical bass/doubling
  discipline; 0 frees inversions.
- `tendency_weight` — leading tone rises, chordal 7th falls. The engine of
  cadential pull; 1.0–1.5 for functional styles, 0 for stasis/ambient.
- `cadence_weight` — extra pressure ONLY on phrase-final tonic downbeats:
  root in bass, complete triad, soprano on 1 or 3, homorhythmic arrival.
- `interval_exists_in_harmony` — penalty for repeated interval colors within
  a sonority (octave doublings always count).
- `chord_templates` — hard whitelist of chord structures, e.g.
  `[[0,4,7],[0,3,7]]` or weighted `{"pcs":[0,4,7],"weight":0.75}`.

## Contour mechanics (applies to every `*_contour`)

`steps = pl × 4 × render_length / voice_contour_resolution` entries;
entry *i* covers beats `[i×res, (i+1)×res)`. With res 4, one entry per bar;
with res 2, two per bar. **Compute the bar↔index mapping before drawing a
shape** — the classic bug is a tension peak landing one bar late, on the
arrival instead of the dominant. Per-voice contours (`voice_*`,
`chord_structure_contour`, `schillinger_ex_contour`, `melody_force_contour`)
are 16 rows (voices) × steps.

`root_contour` modulates the key per bar (moves scales AND every root-keyed
scoring term). `mode_contour` recolors the scale per bar; change modes at
phrase boundaries, not mid-phrase.

## Voice leading

| Field | Effect |
|---|---|
| `same_note_bonus` | hold bias for non-leader voices: 0.3–0.5 mobile, 1 chorale, 2–3 pads |
| `last_note_same`, `last_note_exist_in_voice` | leader repeat penalties (the per-chord "mover" pays these) |
| `melody_force` | anti-circling + stepwise reward on EVERY voice; 1–2 forces moving lines |
| `min/max_voices_changed` | voice-change budget per chord (-1 off; min 1 keeps chords alive) |
| `candidate_range` | ± semitone search window per voice (both modes; nearest chord tone each side always stays reachable) — 3–5 tight, 12 free |
| `no_crossing` | keep large (100); registers are also protected by per-voice buffers |
| `consecutive_octav_fift`, `same_direction` | parallels / contrary-motion pressure (0–1.5) |
| `phrase_echo_weight` | motif memory: reward repeating the interval made one phrase earlier at this position; 0.5–1.0 makes phrases rhyme |
| `suspension_tolerance`, `suspension_resolve_bonus` | prepared dissonant holds + their step-down resolutions |

## Rhythm

Priority: `rhythm_generators` > `lattice_beats` > `voice_rhythm_contour` >
flat `voice_rhythm`.

- **Schillinger resultants**: `rhythm_generators: [a,b]` (or `[a,b,c]`) —
  durations = interference pattern r(a÷b) in `rhythm_unit` beats.
  r(3÷2)=2+1+1+2, r(4÷3)=3+1+2+2+1+3, r(5÷4)=4+1+3+2+2+3+1+4.
  `rhythm_voice_rotation: 1` phase-shifts each voice one element — interlock.
  Set `lattice_beats: 0` alongside (the density term keys off it).
- **Lattice mode**: `lattice_beats: 0.5` = uniform 8th grid; rhythm then
  *emerges* from hold-vs-change, shaped by `metric_hold_strength` (changes
  gravitate to strong beats; negative = syncopation), `momentum_weight`
  (runs vs sustains), `density_weight` (contour becomes a density target).
- `dynamics_weight` — phrase-shaped velocities (tension arch + metric accent
  + line direction + humanization). 1.0 default; 0 = legacy random.

## Style recipes (starting points, then adjust)

- **Chorale/classical**: mode 0 or 5, generated progression, matrix 0,
  tendency 1.5, cadence 1.0, hold bias 1.0, cand_range 4, suspensions on.
- **Jazz comping**: matrix 1, hold bias 0.5, roughness 0.3, cand_range 6,
  metric_hold 0.8, momentum 0.7.
- **Ambient/pad**: matrix 3, tendency 0, hold bias 2.5–3, 2-chord literal
  sequence, slow rhythm, echo 0, cadence 0.
- **Dark tense loop**: mode 5, matrix contour 4 with spikes to 2 on dominant
  bars, loop_wrap 1.5, seq ending on degree 4, tendency 1.2.
- **Motoric/Bach**: rhythm_generators [4,3] rotation 1 (or lattice 0.5 +
  melody_force 1.5), circle-of-fifths-ish literal sequence, suspensions high.

## Reading the breakdown

The API response and archive carry a per-chord breakdown: named `chord_terms`
(harmony_mean/worst/bass/quality, root_position, root_doubling, cadence_*,
parallel_motion…) and per-voice terms (smoothness, tendency, phrase_echo,
loop_wrap, hold_stickiness, suspension…), plus `hard_violations`. When a
render sounds wrong, find the group and read which term won — tune that term,
not a guess. A single `voice_budget` violation at beat 0 is a startup artifact.
