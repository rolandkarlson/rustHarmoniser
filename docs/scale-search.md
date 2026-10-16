# Schillinger scale search

Enable **Try scales per chord** in Algorithm settings to try Ionian, Dorian,
Phrygian, Lydian, Mixolydian, Aeolian and Locrian independently for each chord
block (one four-beat bar). Repeated occurrences of the same progression chord
can choose different modes. The root and chord-degree progression are retained,
including the seeded degree sequence when Cadence is enabled.

The search starts from your configured modes and makes one pass through the
rendered bars. For each bar it tries the six alternative modes, re-harmonizes the
whole progression, and keeps the change with the best total score. This includes
voice leading and later consequences. The engine maximizes score (rewards minus
penalties); a lower penalty improves that score. Ties keep the current setting.
Every trial starts from the same seed. This bounded coordinate search does not
exhaustively test every combination or guarantee the global optimum.

Fugue scale fitting and the score breakdown use the chosen harmony. An
alternative that cannot render valid fitted pitches is skipped. Search never
accepts a lower total score than the initial render.

The generation message reports each bar's chosen scale. The API and archive
store `selected_scale_modes` as a bar-ordered array of mode indices (0–6), empty
when search is off. Saved composition settings are retained.

Cost is up to six extra full renders per bar, so long progressions may be slow.
The option defaults to off, is ignored in Chromatic mode, and resets with style
presets.
