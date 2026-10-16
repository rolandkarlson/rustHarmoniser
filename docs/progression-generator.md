# Chord progression generator

Open **Generators → Chord progression**, choose a method and phrase ending,
then press **Fill Schillinger**. It fills `PL × Render Len` bars, opens the
Schillinger editor, enables Schillinger and disables the automatic Cadence
override. Other configuration fields stay unchanged. The resulting sequence is
ordinary editable configuration and is included in the existing snapshots.

**Functional (Markov)** uses weighted chord transitions with an optional preference
for descending fifths, descending thirds or ascending steps. Major and minor
routes are adapted from the [Musicnotes formula diagrams](https://www.musicnotes.com/blog/chord-progression-formulas/).
The weights are application choices; the article does not give probabilities.
Other modes use modal transition tables. A backward probability pass selects
paths that can reach the chosen ending, without retries or forced invalid edges.

The table and cadence for each phrase use the Mode contour at that phrase's first
bar (or the scalar mode if the contour is empty). Sampling wraps at the contour's
length and uses its beat resolution, matching the engine. The existing Mode and
Root contours still determine each bar's actual pitches. Mid-phrase mode changes
can therefore recolor the intended cadence. Aeolian uses the engine's major V;
its natural VII remains distinct from the raised vii° in the article, which the
current degree-only sequence cannot separately represent.

**Schillinger-inspired cycles** separate repeating diatonic root motion from chord
structure. See [Berklee's description of Schillinger's root cycles](https://www.berklee.edu/berklee-today/fall-2000/The-Schillinger).
From zero-based degree 0, fifths move by −4, thirds by −2 and ascending steps by +1,
wrapping modulo 7. Each phrase restarts on the tonic. Mixed cycles rotate a
repeating −4, −2, +1 pattern using the seed. These are a limited root-cycle tool,
not an implementation of the entire Schillinger System.

Endings apply to every phrase:

- **Resolve to tonic:** characteristic cadence then tonic inside the phrase.
- **Lead into next phrase:** end on the cadential degree, resolving into the next
  phrase's tonic (or the first bar when looped).
- **Free movement:** no forced ending; preserves the full root cycle.

Cadential degrees by mode are V, IV, II, V, VII, V, II for Ionian through Locrian.
One-bar phrases contain the tonic. Two-bar resolved phrases contain only the
cadence. For cycles, a selected ending replaces the final one or two roots.

The progression seed is independent of the arrangement seed. Identical settings
and seed reproduce the result; **New variation** picks a new seed and fills again.
Single root cycles are fixed and do not use random variation. Root-degree labels
are not chord-quality labels: Mode, Chord and Sch. EXP still control chord tones.
Dimensions must be positive integers, with a maximum of 8192 bars per fill.

Validation: `cd gui && npm test` covers legal transitions in all modes, cadence
and loop endpoints, short phrases, root cycles, reproducibility, contour sampling,
configuration preservation and invalid inputs. `npm run build` checks the UI.
