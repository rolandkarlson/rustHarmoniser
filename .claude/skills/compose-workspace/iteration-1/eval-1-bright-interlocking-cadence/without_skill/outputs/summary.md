# Bright 16-bar piece with interlocking rhythms and a strong final cadence

Final artifact: `final_render.json` — a copy of `render/render_1787739143254.json`,
the self-contained archive of the winning generate (full input config including
`rng_seed`, plus all 341 output notes and the per-chord score breakdown). Re-POSTing
`input.config` from that file to `http://127.0.0.1:3000/api/generate` reproduces it.

## What it sounds like

A 16-bar (64-beat), five-voice piece in C major — four 4-bar phrases, each closing
with its own V->I cadence, so the form breathes in question/answer pairs before the
final arrival.

- **Bright**: C Ionian throughout (mode and root contours flattened to zero so the
  default mid-render Dorian bump never darkens it). The pitch-class census is
  dominated by C, G, A, D, E — a pentatonic-leaning major glow — with B and F
  appearing mostly as cadential tendency tones. The whole texture sits higher than
  the engine default (start notes 74/69/64/55/43 instead of a subterranean bass at 34).
- **Interlocking rhythms**: the attack grid is the Schillinger resultant r(4:3) =
  3+1+2+2+1+3 in eighth-note units, with `rhythm_voice_rotation: 1` starting each
  voice one element deeper into the cycle — five staggered strata of the same
  pattern. On top of that, `min_voices_changed: 0` plus the hold-bias shaping means
  motion is traded around: of the 103 onsets where any pitch changes, 92 involve
  exactly one voice. The music is in near-constant eighth/quarter motion, but the
  motion hops from voice to voice — hocket-like give and take rather than
  homorhythmic chords.
- **Strong final cadence**: bar 15 is a true dominant — G pedal in the bass, the
  leading tone B sounding in two voices, the soprano climbing A4->B4->C5 (scale
  degrees 6-7-8) so it lands on the tonic exactly at the bar-16 downbeat as the
  bass falls G2->C3. Bar 16 settles through an added-sixth color (A3 resolving to
  C4) onto a final sustained root-position C major triad — C3/G3/C4/G4/E5, bass on
  the root — with velocities tapering (92 -> ~70) under the dynamics arch.
  Authentic cadence, complete triad, no loose ends.

## How it was produced

Local engine only (no Ableton MCP calls, server untouched). Fetched the default
config from `GET /api/config`, patched it, and posted the whole config to
`POST /api/generate` (one request at a time). Key settings on top of the defaults:

| knob | value | why |
|---|---|---|
| `pl` / `render_length` | 4 / 4 | 4 phrases x 4 bars = 16 bars (64 beats) |
| `use_generated_progression` | true | mode-aware random-walk roots; every phrase forced to close V->I |
| `mode`, `root` | 0, 0 | C Ionian |
| `mode_contour`, `root_contour` | all 0 | kill the default Dorian mid-phrase bump; no modulation |
| `start_notes` | 74,69,64,55,43 | brighter register |
| `rhythm_generators` | [4,3] | Schillinger resultant 3+1+2+2+1+3 attack pattern |
| `rhythm_unit` | 0.5 | pattern in eighth notes |
| `rhythm_voice_rotation` | 1 | each voice offset one element into the cycle -> interlock |
| `cadence_weight` | 2.5 | heavy phrase-final tonic terms (bass root, complete triad, soprano on 1/3, homorhythmic landing) |
| `tendency_weight` | 1.0 | firm leading-tone resolution |
| `min_voices_changed` | 0 | let the ending settle on holds instead of forced motion |
| `loop_wrap_weight` | 0 | one-shot piece, not a loop |
| `rng_seed` | 7 | reproducibility |

## Iterations

1. **v1** (`render_1787739074227.json`, `min_voices_changed: 1`,
   `cadence_weight: 2.0`): right length, right key, good interlock — but the
   forced-motion rule pushed the bass off the root to G2 on the very last
   eighth, and the bar-16 downbeat carried an unresolved D. Cadence verdict:
   good approach, weak landing.
2. **v2 = final** (`render_1787739143254.json`): allowing full holds and raising
   the cadence weight fixed both — verified numerically from the archive
   (bar-by-bar downbeat sonorities, final-eighth sonority, per-onset
   voice-change histogram, pitch-class census) rather than by ear.
