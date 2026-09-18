# Ableton output

The server now creates and replaces Ableton Session clips through the
AbletonMCP Remote Script. It no longer rewrites `harmonize.js`. Every Generate
still writes the engine notes to `output.json` and a self-contained file under
`render/`. The archive contains the core configuration, imported Lead Clip,
output settings, derived clip plans, and score breakdown.

Output is a separate layer from composition. Its transformations work on a
copy of the completed engine notes and use a deterministic random stream derived
from the core `rng_seed`. They do not consume or change the core renderer's RNG.
The available route modes are:

- `notes`: selected voices, with optional transposition, pitch-class folding,
  timing humanization, time scaling, and deterministic adjacent-note merging.
- `trance_arp`: the former bass-and-lead trance texture, using configurable
  bass and lead voices, octave floor, and duration pattern.
- `contour_arp`: active harmony pitch classes follow the former two-wave melodic
  contour around `initial_pitch`.
- `voice_arp`: a duration cycle selects active notes using `voice_pattern`.

`time_scale` multiplies note starts, note durations, and the complete configured
render span (`pl × 4 × render_length`). This retains trailing rests. Humanize is
measured in final output beats. An engine probability of zero is its historical
“unspecified” sentinel and exports as probability `1.0`.

Fugue defaults create one raw route per fugue voice. Merge, humanization, fold,
and transposition are off, so protected entries retain their separate repeated
attacks and exact timing. Output itself defaults off; Generate can safely render,
cache, and archive music while routes are being edited.

## HTTP API

`POST /api/generate` accepts the new envelope:

```json
{
  "config": {},
  "output": {
    "enabled": false,
    "routes": []
  }
}
```

A legacy bare core Config is still accepted and receives disabled default
routes. When `output.enabled` is true, every enabled route is validated before
rendering and the generated plan is sent as one `write_harmonizer_clips` batch.
A render remains successful and cached if Live later reports a failed, partial,
or uncertain export.

- `GET /api/output/preview` returns the cached settings, source context, and
  per-clip summaries without contacting Live.
- `POST /api/output/preview` with `{ "output": ... }` replans the last render
  without rendering or writing clips.
- `POST /api/output/export` with `{ "output": ... }` validates and sends the
  last generated music. It never silently retries a timed-out write.
- `GET /api/output/session` returns the Remote Script handshake, Live session,
  track names, and clip slots for the Output page. `/api/ableton/session` is an
  alias.

Preview and explicit export always use the cached render's Config for its voice
count, length, and Lead Clip collision guard. New unsent composition settings
cannot reinterpret old notes.

## Remote Script extension

The repository-owned extension is in `integrations/ableton/`. It performs a
whole-batch preflight on Live's main thread, then updates destinations in order.
Existing clips keep their identity; all existing note IDs are removed even when
they lie outside the old loop markers. Empty slots receive a new MIDI clip.
Markers and loop length are set to the full planned length without starting
playback.

The installer refuses an unfamiliar or partially patched AbletonMCP file,
backs up files it changes, and supports dry-run and verification:

```sh
python3 integrations/ableton/install.py "/path/to/Remote Scripts/AbletonMCP" --dry-run
python3 integrations/ableton/install.py "/path/to/Remote Scripts/AbletonMCP"
python3 integrations/ableton/install.py "/path/to/Remote Scripts/AbletonMCP" --check
```

Live must be restarted after installation so `get_script_info` advertises
`write_harmonizer_clips`. A runtime error can leave earlier clips written; the
response identifies every written, failed, and unattempted destination. A
timeout is explicitly uncertain because the already scheduled Live main-thread
task may still finish. Review Live before choosing whether to send again.
