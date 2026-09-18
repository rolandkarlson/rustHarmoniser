# Dark Aeolian Loop — 8 bars, A minor

## The music

Two 4-bar phrases in A Aeolian, five voices (soprano to bass), built to run as a
seamless loop:

- **Phrase 1 (bars 1–4):** i — iv — V — i. Tension climbs through the
  subdominant into the dominant bar (measured dissonance 0.00 → 0.28 → 0.38),
  then resolves onto a dark **Am7 arrival** (A–C–E–G, bass on the root,
  dissonance falling back to 0.25) at the phrase-final downbeat.
- **Phrase 2 (bars 5–8):** III — VI — ii° — V. Opens with a brief lift onto the
  relative major, then darkens step by step (0.30 → 0.33 → 0.40 through the
  diminished ii°) into a genuine **E7 dominant** (E–G#–B–D) on bar 8. That
  chord never resolves inside the clip — it resolves across the loop wrap into
  the clean tonic downbeat of bar 1, so the loop pulls itself back around.

Voice leading is tight throughout: maximum melodic jump is a perfect fourth in
any voice (soprano 5 semitones once, inner voices ≤4, baritone moves only by
step), and the loop seam is small (6 semitones total across all five voices).
Rhythm accelerates into each cadence — upper voices move from quarters to
eighths as tension builds, the bass moves from whole notes to halves on the
dominant bars — and the shaped dynamics arc peaks (velocity 98) on the
phrase-1 cadence and relaxes (78) into the loop restart.

Tempo suggestion in Ableton: ~70–90 BPM, minor-key pad or string rack.

## Config choices (why)

| Choice | Value | Reason |
|---|---|---|
| `mode` / `root` | 5 (Aeolian) / 9 (A) | dark, natural-minor brief |
| `mode_contour` | all 5 | contour overrides the scalar; pinned to Aeolian |
| `root_contour` | `[]` (disabled) | default all-zeros would have forced C |
| `pl` / `render_length` | 4 / 2 | 8 bars, cadence every 4 bars |
| `schillinger_sequence` | `[0,3,4,0,2,5,1,4]` | i–iv–V–i, III–VI–ii°–V; ends on V so the wrap resolves home |
| `harmony_matrix_contour` | `[4,4,2,0,4,4,2,2]` | Dark row as baseline, Tension row on dominant/ii° bars, Strict Classical on the phrase-1 arrival |
| `harmony_distance_contour` | `[0.45,.2,.15,.5,.3,.2,.1,.1]` | consonance wins on arrival bars, smoothness wins on tense bars |
| `loop_wrap_weight` | 1.5 | voice the final bar to lead back into bar 1 |
| `cadence_weight` / `tendency_weight` | 1.5 / 1.2 | strong functional arrivals, leading-tone pull |
| `candidate_range` | 4 | tight voice leading (±4 semitone search window) |
| `same_note_bonus` | 1.5 | held common tones without droning |
| `melody_force_contour` | rises 0.2 → 0.7 into each cadence | lines move more as tension builds |
| `voice_rhythm_contour` | 5 rows, durations shrink into cadences | rhythmic intensification into each arrival |
| `phrase_echo_weight` | 0.6 | the two phrases rhyme motivically |
| `start_notes` | 69,64,60,52,45 | A-minor spread, in-key seeds |
| `consecutive_octav_fift` / `root_position_weight` | 0.5 / 0.7 | cleaner counterpoint, grounded cadence basses |
| `rng_seed` | 42 | reproducible |

## Iterations

**3 generate calls.**

1. First render: shape already close (tight jumps, seam 3), but the phrase-1
   arrival carried non-chord tones (Am with F and G on the downbeat, 0.30
   dissonance) and bar 5 spiked to 0.45 right after the arrival.
2. Weight-only nudge (matrix row 0 on the arrival, cadence 1.5, reshaped
   distance curve) — the beam search returned byte-identical notes; scores
   moved, the argmax didn't.
3. Changed the harmony itself: phrase 2 now opens on III instead of VI, easing
   the post-cadence bar and giving a monotonic build into the E7. Final:
   clean Am7 arrival, rising phrase-2 tension, dominant-seventh loop wrap,
   zero real hard violations.

Archive: `render/render_1787737213736.json` (copied here as
`final_render.json`). Analyzer output in `analysis.txt`.
