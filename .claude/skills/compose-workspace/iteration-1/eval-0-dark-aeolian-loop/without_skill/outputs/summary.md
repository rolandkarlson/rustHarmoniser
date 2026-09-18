# Dark Aeolian 8-bar loop

Final artifact: `final_render.json` — a copy of `render/render_1787737677618.json`, the archive of the last `/api/generate` call. It contains the full input config (`input.config`) and the 105 rendered notes (`output.notes`), so it reproduces exactly (renders are deterministic for a given config; on this config the seed only jitters velocities +/-3).

## The music

**A Aeolian, 8 bars (32 beats), two 4-bar phrases, one chord per bar:**

| bar | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| chord | Am | Dm11 | **E7b9** | Am | F | **Bdim(11)** | **E7** | Am |
| degree | i | iv | V | i | VI | ii-dim | V | i |

- **Tension into each cadence, three ways at once:**
  - *Harmonic:* chord extensions widen through each phrase (triads -> 9th/11th colors -> full 7th/b9 dominant), then snap back to a bare Am triad at each arrival. The Aeolian engine raises the subtonic only on dominant bars, so G# appears exclusively in bars 3 and 7 and resolves to A (v2 walks G#2 -> A2 into the final cadence; the melody climbs the E7b9 arpeggio G#-B-D-F into bar 4).
  - *Rhythmic:* onset density accelerates into each cadence — 8 / 16 / 22 / 5 onsets in phrase 1, 8 / 15 / 26 / 5 in phrase 2 — half-notes at phrase start, eighth-note motion on the dominant bar, then a whole-bar chord on the arrival.
  - *Dynamic:* shaped velocities crescendo through the phrase (bar means 83 -> 96 -> 103 -> 109 into the first cadence).
- **Tight voice leading:** average 1.55 voices move per onset, mean step 2.56 semitones, max leap a 4th; upper voices are mostly stepwise while the bass walks roots (A | D | E | A || F | D | E | A, root position on every downbeat).
- **Loop-ready:** notes end exactly at beat 32; the final bar is the same root-position Am the loop opens on (only the bass re-enters an octave lower), and `loop_wrap_weight` scored the final bar against the opening pitches, so the wrap is seamless. Set the Ableton clip loop to 8 bars.
- **Dark color:** the E7b9 keeps the b9 (F natural) grinding against the bass E, phrase 2 runs VI -> half-diminished ii -> V, the bass sits down at A1/A0, and the harmony matrix contour plays the "Dark" style row with the "Tension" row on pre-cadence bars.

## How it was produced

Single POST of a hand-built config to the local engine at `127.0.0.1:3000/api/generate` (Schillinger mode, joint chord scoring, beam 4 / lookahead 2 — 105 notes in ~17 ms). Key config choices (all visible in `final_render.json` under `input.config`):

- `mode: 5` (Aeolian), `root: 9` (A), `mode_contour`/`root_contour` pinned for all 8 bars; `pl: 4`, `render_length: 2` -> two phrases, cadence terms fire on bars 4 and 8.
- Literal `schillinger_sequence: [0,3,4,0, 5,1,4,0]` (i-iv-V-i | VI-ii0-V-i) spanning the full render so nothing wraps early.
- `chord_structure_contour` per voice: triad -> add-extensions -> 7th/9th -> triad each phrase (bass locked to root/3rd/5th).
- `voice_rhythm_contour` per voice: durations shrink into the dominant bar (2 -> 1 -> 0.5 beats) and land on a 4-beat arrival chord.
- `harmony_matrix_contour: [4,4,2,4, 4,2,2,4]` — Dark row, Tension row pre-cadence; `harmony_distance_contour` arches toward consonance at arrivals.
- Voice-leading tightness: `candidate_range: 3` (stepwise candidates), `same_note_bonus: 2.0` (hold common tones), `min_voices_changed: 1`.
- Expression: `tendency_weight: 1.0` (leading-tone pull), `cadence_weight: 1.5` (root-position, complete, homorhythmic arrivals), `loop_wrap_weight: 1.5` (seam), `phrase_echo_weight: 0.6` (phrase 2 echoes phrase 1's motif rhythm), `dynamics_weight: 1.2`, `root_position_weight: 0.7`.
- `start_notes: [69,64,60,52,33]` — an A-minor spread with a deep A1 bass anchor.

Verified by parsing `output.json`: pitch-class content is pure A natural minor except the raised G# on dominant bars; both phrase-final downbeats are root-position A-minor triads; loop seam distance is minimal (Am -> Am).

To reproduce: `curl -X POST -H 'Content-Type: application/json' --data "$(python3 -c 'import json;print(json.dumps(json.load(open("final_render.json"))["input"]["config"]))')" http://127.0.0.1:3000/api/generate`
