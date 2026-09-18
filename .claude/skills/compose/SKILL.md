---
name: compose
description: Compose music with the rustHarmoniser engine in this repo — translate a musical brief (mood, mode, length, loop or one-shot, rhythm feel) into an engine config, generate via the local API, verify the result by the numbers, iterate, and deliver into Ableton. Use this whenever the user asks to compose, generate, render, or "make" music — a loop, a piece, a progression, a harmonization, a pad, "something dark/bright/tense" — even if they never mention the harmonizer, a config, or Ableton by name.
---

# Compose with rustHarmoniser

rustHarmoniser renders 5 voices (channels 0=soprano … 4=bass): a Schillinger
progression layer picks per-bar chords, and a beam search re-pitches every
voice against a scored rulebook (consonance matrix, voice leading, tendency
tones, cadences, rhythm shaping). You compose by choosing the *rules and
curves*, not the notes. The whole knob surface is one JSON config POSTed to a
local server; every render is archived with a per-chord score breakdown that
explains why each voicing won.

The craft of this skill: translate what the user *means* ("dark", "tense
before the drop", "hymn-like") into config values, then verify the intent
actually landed by measuring the output — never by assuming.

## Step 0 — Preflight

```bash
curl -s http://127.0.0.1:3000/api/config >/dev/null && echo up || echo down
```

If down, build and start it (from the repo root):

```bash
cargo build --release && nohup ./target/release/rustnote > server.log 2>&1 &
```

Two hard-won rules:
- **A rebuilt binary is not a restarted server.** If you change Rust code, the
  running process keeps the old build — kill and relaunch before generating.
- **Never issue concurrent generate requests.** The engine's seeded RNG is
  global; parallel renders corrupt each other. One at a time.

## Step 1 — Brief → config

Start from the server's defaults and patch only what the brief implies:

```bash
curl -s http://127.0.0.1:3000/api/config > /tmp/compose_config.json
```

The short map (full knob guide with ranges and recipes:
[references/config-guide.md](references/config-guide.md) — read it before your
first compose in a session):

| Brief says | Patch |
|---|---|
| dark / sad / minor | `mode: 5` (Aeolian), H-matrix contour toward row 4 (Dark) |
| bright / floaty | `mode: 3` (Lydian) or 0, matrix row 5 (Bright) or 3 (Ethereal) |
| jazzy / lush | matrix row 1, `roughness_weight` ~0.3, looser `candidate_range` 5–7 |
| tense, building | H-matrix contour ramps to row 2 **on the dominant bar, back to 0 on the arrival bar** — never leave row 2 on a tonic arrival |
| N bars long | `pl × render_length = N` (pl = bars per phrase; cadence lands every pl bars) |
| it's a loop | `loop_wrap_weight: 1.0–1.5`; ending the sequence on V makes the wrap pull home |
| strong endings | `cadence_weight: 0.8–1.5`, `tendency_weight: 1.0–1.5`, `use_generated_progression: true` |
| groove / interlocking rhythm | `rhythm_generators: [4,3]` or `[5,4]`, `rhythm_unit: 0.5`, `rhythm_voice_rotation: 1`, and set `lattice_beats: 0` |
| calm / chorale / pads | no generators, slow `voice_rhythm_contour` or default; `same_note_bonus` 1.5–3 |
| busy, moving lines | `same_note_bonus` 0.3–0.5, `melody_force` 1–2, `lattice_beats: 0.5` + `density_weight` |
| tight voice leading | `candidate_range: 3–5` (works in both modes; nearest chord tone each side always stays reachable) |
| motif / phrases that rhyme | `phrase_echo_weight` 0.5–1.0 |
| key of X | `root`: pitch class 0–11 (or transpose everything with `main_pitch`) |

Contour lengths matter: every contour has
`steps = pl × 4 × render_length / voice_contour_resolution` entries, and the
resolution (beats per step) decides which bar each entry covers — compute the
bar↔index mapping before drawing a tension curve, and double-check the peak
lands on the dominant bar, not one bar late.

## Step 2 — Generate

```bash
curl -s -X POST http://127.0.0.1:3000/api/generate \
  -H 'Content-Type: application/json' \
  -d @/tmp/compose_config.json > /tmp/compose_result.json
```

The response carries `status`, a message naming the **archive path**
(`render/render_<ts>.json` — config + notes + full score breakdown), and the
breakdown itself. The server also writes `output.json` (repo root) and pushes
the notes into the Ableton Max device.

## Step 3 — Verify by the numbers, then iterate

Run the bundled analyzer on the archive the render just named:

```bash
python3 .claude/skills/compose/scripts/analyze_render.py render/render_<ts>.json
```

It reports: per-bar dissonance share, bass-on-root rate, per-voice motion and
max jumps, velocity arc, loop seam distance, cadence-bar quality, and any hard
violations. Read it against the brief — "tension into the cadence" should show
dissonance peaking on pre-cadence bars; "tight voice leading" should show max
jumps ≤ a fifth; a loop should show a small seam.

(A single `voice_budget` violation at beat 0 is a known cosmetic artifact —
ignore it.)

When something is off, change the *responsible* knob, regenerate, re-measure:

| Symptom | Knob |
|---|---|
| too static, drones | lower `same_note_bonus`, raise `melody_force`, raise `min_voices_changed` |
| jumpy, instrumental lines | lower `candidate_range`, raise smoothness side (`harmony_distance_balance` down) |
| cadences don't land | raise `cadence_weight`, `tendency_weight`; check the phrase ends on the tonic |
| tension in the wrong place | re-index the H-matrix contour (bar mapping above) |
| muddy low end | raise `roughness_weight` (register-aware), check bass register via `start_notes` |
| loop seam bumps | raise `loop_wrap_weight` |
| all bars feel the same | vary `harmony_matrix_contour`, add `root_contour` modulation, `phrase_echo_weight` for form |

One or two iterations is normal; measure each one rather than stacking blind
changes.

## Step 4 — Deliver and report

The render is already in Ableton (Max device path, track 0 / clip 1). If
AbletonMCP tools are available, audition it: `start_playback` / `fire_clip`,
set an apt `set_tempo`. Only push notes into other clips with
`add_notes_to_clip` if the user asks for a specific destination.

Report to the user in **musical terms first**: what was composed (mode, form,
where the tension sits, how it ends), then the notable numbers from the
analyzer, then the archive path so the exact render is reproducible. Invite
adjustments in plain language — "warmer", "busier bass", "longer phrases" —
and map them yourself.
