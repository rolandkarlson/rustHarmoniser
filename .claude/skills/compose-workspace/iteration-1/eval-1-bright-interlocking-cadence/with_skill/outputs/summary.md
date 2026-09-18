# Bright 16-bar piece, interlocking rhythms, strong final cadence

Final archive: `render/render_1787738063884.json` (copied here as `final_render.json`).
Generate iterations used: **5**.

## The music

C Lydian, 16 bars in four 4-bar phrases, five voices. The rhythm is a
Schillinger resultant r(4÷3) in 8th-note units with `rhythm_voice_rotation: 1`,
so each voice strikes the same 3+1+2+2+1+3 pattern one element out of phase —
the voices genuinely interlock, handing the pulse around rather than moving in
block chords. Harmony sits on the Bright/Lydian style row the whole way, which
keeps the sonorities open and major-leaning: the analyzer shows essentially
zero dissonant-pair share in every bar. The dynamics arc rises through the
third phrase (peak mean velocity 98 at bar 10) and relaxes into the ending.

Each phrase closes on the tonic (generated Lydian progression, V→I cadences).
The final cadence is the strongest moment by design: the H-matrix contour
spikes to the Tension row on the last dominant bar (bar 15) and returns to
Strict Classical on the arrival, the harmony-vs-smoothness balance is pushed
to its consonance maximum (0.5) on bar 16, and the melody-force anti-hold
pressure is released in the final bar so the voices settle. Result: the bar-16
downbeat is a full root-position C major (C2–C3–C4–E4–G4, root tripled), and
the last bass strike of the piece is C2 — root position all the way out.

## Analyzer highlights (final render)

- 0 real hard violations (only the known beat-0 artifact)
- All 4 phrase-final downbeats: tonic pcs with bass on root; bars 4, 12, 16
  are complete triads (bar 8 omits the fifth — the one small blemish)
- Max melodic jump: 5 semitones in every voice (tight, singable lines)
- Voice mobility: 29–45 moves per voice across 64 beats — active interlock,
  not chorale blocks
- Dissonant-pair share ≈ 0 everywhere — reads as "bright"
- Velocity arc: 77 → 98 (bar 10 peak) → 79 (relaxed final bar)

## Config choices (patched from server defaults)

| Choice | Value | Why |
|---|---|---|
| `mode` / `root` | 3 (Lydian) / 0 (C) | "bright" |
| `pl` / `render_length` | 4 / 4 | 16 bars, cadence every 4 |
| `use_generated_progression` | true | mode-aware progression with V→I phrase cadences |
| `rhythm_generators` | [4,3], unit 0.5, rotation 1, lattice 0 | interlocking r(4÷3) with per-voice phase shift |
| `harmony_matrix_contour` | 5×14, 2 (bar 15), 0 (bar 16) | Bright row throughout; tension spike on the final dominant, clean classical arrival |
| `harmony_distance_contour` | ~0.2 rising to 0.4 at each cadence, 0.5 at the end | consonance wins hardest at arrivals |
| `cadence_weight` / `tendency_weight` | 1.2 / 1.2 | strong functional pull and cadence discipline |
| `same_note_bonus` / `melody_force` | 0.8 / 0.5 | mobile lines that suit the interlock |
| `melody_force_contour` | 0.5 everywhere, 0.0 in bar 16 | release anti-hold pressure so the ending settles on the tonic |
| `candidate_range` | 4 | tight voice leading (max jump = a fourth) |
| `phrase_echo_weight` | 0.6 | the four phrases rhyme motivically |
| `loop_wrap_weight` | 0 | one-shot piece, not a loop |
| `rng_seed` | 42 | reproducibility |

## Iteration log

1. First render — already clean; all cadences rooted, but the piece's very
   last bass strike drifted to E (first-inversion tail).
2. Raised `root_position_weight` 1.0 + `cadence_weight` 1.5 — overall
   bass-on-root improved to 69% but the *final downbeat* lost its root. Worse
   where it mattered.
3. Seed change — confirmed the beam search is deterministic (identical notes,
   only velocities moved). Seed is not the lever for cadence shape.
4. `beam_width` 5 — wider search degraded cadence purity (added-2nd colors on
   arrivals). Reverted.
5. Back to iteration-1 weights + `melody_force_contour` released in the final
   bar — root-position arrival AND a root-position final strike. Kept.
