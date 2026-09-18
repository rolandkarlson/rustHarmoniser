# Fugue exposition

Fugue mode uses a monophonic Ableton MIDI clip as its subject and builds one finite exposition for three to eight voices. In the GUI, open **Lead Clip**, select the source track and clip, turn on **Fugue**, and choose **Voices**. Three voices is the default and preserves the original behavior. Set the key root and mode to match the subject before generating.

The entry order begins with voice 1 subject, voice 0 real answer, and voice 2 subject. Voices 3 through the selected last voice enter afterward, alternating real answers and subjects. The answer is a literal transposition, set by **Answer** (`7` means a perfect fifth above).

**Registers** contains one semitone register offset per channel. Answer entries add **Answer** to their register offset. The automatic spread is `0, 0, -12, -24, -24, -36, -36, -48`, so successive subject/answer pairs descend by octaves. **Bass Register** edits the last value in this list; if the final entry is an answer, its answer interval is added too. Direct configurations with four to eight voices use the register vector when supplied and the automatic value for every missing item. Three-voice configurations continue to use the legacy bass-shift field, so old snapshots render the same notes.

The imported timing, durations, pitches after the planned transposition, and separate repeated-note attacks are protected while the harmonizer creates accompanying lines in every active voice. Each voice is silent until its first entry. Added voices without an explicit start-note seed derive one from the subject and their planned register, clamped only for generated accompaniment. Protected pitches remain literal unless **Keep subject transformations in scale** is enabled.

**Keep subject transformations in scale** fits every subject and answer note—including the opening imported subject and all inverted or stretched returns—to the nearest MIDI pitch in that voice's Schillinger harmony at the note's onset. An equal-distance tie chooses the lower pitch. The note keeps its original onset, duration, velocity, and separate attack. A sustained note crossing into another bar is fitted once from its onset bar and is not split or re-fitted mid-note. The option uses the same generated Schillinger progression that scores the accompaniment, and it is ignored when Schillinger progression is off.

**Entry Gap** is measured in beats. Leave it at `0` to use the source clip length. The render must be long enough to contain all opening entries; increase **Render Len** if the GUI reports that the exposition does not fit. Subject notes must be monophonic and inside the source clip. Muted notes are ignored. No original or transformed note may last more than 32 beats.

Turn on **Repeat** to continue cycling protected subject entries through all selected voices in the same `[1, 0, 2, 3…]` order after the opening exposition. The default transformation cycle is:

1. chromatic inversion at normal time;
2. original subject at `2×` time;
3. chromatic inversion at `2×` time;
4. original subject at normal time.

Each cycle row can be changed to Normal or Inverted and `0.5×`, `1×`, `2×` or `4×` time, reordered with the arrow buttons, removed, or extended with **+ Step**. Chromatic inversion reflects every semitone interval around the first sounding subject pitch. Each entry is freshly derived from the imported subject, so transformations never accumulate.

Time scaling changes note starts, note durations, the full source clip span, and the gap leading to the next repeat. This preserves leading, internal and trailing rests. In a three-voice fugue with a four-beat source and automatic four-beat gap, repeat entries begin at beats 12, 16 and 24 for the first three default transformations. With an explicit eight-beat gap, the first repeat begins at beat 24. If a voice is still occupied by an earlier stretched entry, its next entry waits until that voice is free.

Only complete transformed entries are scheduled. When the next cycle item does not fit before the render ends, repetition stops and the generated accompanying lines fill the remaining available time. The planner does not skip an oversized item to try a shorter later transformation.

Two optional fugue-aware scoring controls let the generated accompaniment use the protected entry plan:

- **Subject Ant.** looks ahead to subject or answer notes in other voices that will begin while the current generated note is still sounding. It penalizes dissonant interval styles, unisons, and voice crossings in proportion to the real overlap. Notes already sounding and protected notes beginning on the current onset stay with the ordinary chord scorer, so the same event is not counted twice. This is conservative: it does not yet infer that a future passing tone or suspension will resolve.
- **Entry Approach** shapes only the final generated note before that voice's next protected return. It favors a semitone or whole-tone approach to the return's first sounding pitch. The planner uses the real first note rather than the clip-block boundary, so leading rests are respected; approaches more than four beats away receive no score.

Both controls default to `0`, which keeps the earlier fugue and ordinary harmony scoring unchanged. Their contributions appear in the render explanation as `fugue_future_overlap` and `fugue_entry_approach`. Imported subject and answer notes remain immutable even at high weights. Conflicts consisting only of immutable protected notes can be reported in the normal score context but are not "fixed" by changing those notes.

Open the **Output** page to route the fugue into Ableton. Its Track and Slot fields use the familiar one-based labels shown in Live; the HTTP API stores those indices from zero. The default fugue routes write one unmodified clip per selected voice through the Rust server and AbletonMCP bridge, preserving timing and repeated-note attacks. Every enabled destination must be distinct and must not reuse the Lead Clip's source track and slot. See [Ableton output](output.md) for routing, preview, transformation, and export details.

This mode produces an alternating subject/real-answer exposition, optional transformed subject returns, and free generated accompaniment. It does not yet create tonal answers, recurring countersubjects, episodes, stretto, or a complete fugue form.
