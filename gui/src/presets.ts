import { DEFAULT_HARMONY_MATRIX } from './harmony.ts';

export const PRESET_INFO = {
  jazz: { label: 'Tonal Jazz', description: 'Major-key I–vi/IV–ii–V seventh-chord turnarounds, a swing grid for the lead, syncopated inner voices and quarter-note bass. Favors resolution across the loop seam.' },
  modal_jazz: { label: 'Modal Jazz', description: 'Four-note quartal stacks on Dorian scale degrees 1 and 4, spacious rhythms and little functional resolution pressure.' },
  classical: { label: 'Classical', description: 'Major-key tonal phrases, triads and functional sevenths, common tones and restrained motion. Ends on a tonic cadence.' },
  bach: { label: 'Bach-inspired', description: 'Baroque-inspired chordal motion, descending-fifth sequences, prepared suspensions and cadential broadening. This is a chord-based texture, not a model of Bach’s counterpoint.' },
  ambient: { label: 'Ambient', description: 'Slow Lydian I–II movement, widely spaced registers and sustained color, favoring one voice changing at a time.' },
  trance: { label: 'Trance', description: 'Minor-key anthem loops, an eighth-note grid for lead and bass, root-focused triads and sustained inner parts. Harmonic MIDI material for a trance arrangement.' },
} as const;

export type PresetName = keyof typeof PRESET_INFO;

export const DEFAULT_FUGUE_TRANSFORM_CYCLE = [
  { invert: true, time_scale: 1 },
  { invert: false, time_scale: 2 },
  { invert: true, time_scale: 2 },
  { invert: false, time_scale: 1 },
] as const;

export const DEFAULT_FUGUE_REGISTER_SHIFTS = [0, 0, -12, -24, -24, -36, -36, -48] as const;

export function resizeFugueRegisterShifts(value: unknown, count: number): number[] {
  const saved = Array.isArray(value) ? value : [];
  return Array.from({ length: count }, (_, i) =>
    typeof saved[i] === 'number' && Number.isFinite(saved[i])
      ? Math.round(saved[i])
      : DEFAULT_FUGUE_REGISTER_SHIFTS[i]
  );
}

/** Keep valid custom routes in order, then append unused track indices. */
export function resizeFugueOutputTracks(value: unknown, count: number): number[] {
  const routes: number[] = [];
  if (Array.isArray(value)) {
    for (const raw of value) {
      const track = typeof raw === 'number' && Number.isSafeInteger(raw)
        && raw >= 0 && raw <= 2_147_483_647 ? raw : -1;
      if (track >= 0 && !routes.includes(track) && routes.length < count) routes.push(track);
    }
  }
  const highest = routes.length > 0 ? Math.max(...routes) : -1;
  let candidate = highest < 2_147_483_647 ? highest + 1 : 0;
  while (routes.length < count) {
    while (routes.includes(candidate)) candidate = candidate === 2_147_483_647 ? 0 : candidate + 1;
    routes.push(candidate);
    candidate = candidate === 2_147_483_647 ? 0 : candidate + 1;
  }
  return routes;
}

const finite = (value: unknown, fallback: number): number =>
  typeof value === 'number' && Number.isFinite(value) ? value : fallback;

function randomSource(seed: number) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let n = Math.imul(state ^ state >>> 15, state | 1);
    n ^= n + Math.imul(n ^ n >>> 7, n | 61);
    return ((n ^ n >>> 14) >>> 0) / 4294967296;
  };
}

/** A style owns every musical field. Preserve only the user's tonal center,
 * transposition, phrase count and lead-clip routing, plus unknown app metadata.
 * No previous rhythm generator, whitelist, search budget or contour may silently
 * override a new preset. All randomness comes from the saved seed.
 */
export function createPreset(kind: PresetName, previous: Record<string, unknown>, seed: number) {
  const random = randomSource(seed);
  const root = ((Math.round(finite(previous.root, 0)) % 12) + 12) % 12;
  const phrases = Math.max(1, Math.round(finite(previous.render_length, 2)));
  const bars = phrases * 4;
  const flat = (value: number) => Array<number>(bars).fill(value);
  const voices = (fn: (voice: number, bar: number) => number) =>
    Array.from({ length: 16 }, (_, voice) => Array.from({ length: bars }, (_, bar) => fn(voice, bar)));
  const arch = (bar: number) => Math.sin(Math.PI * bar / Math.max(bars - 1, 1));
  const choose = <T,>(pool: readonly T[]): T => pool[Math.floor(random() * pool.length)];

  const nc = {
    ...previous,
    root,
    main_pitch: Math.round(finite(previous.main_pitch, 0)),
    render_length: phrases,
    pl: 4,
    rng_seed: seed,
    use_leading_voice: previous.use_leading_voice === true,
    leading_voice_track: Math.round(finite(previous.leading_voice_track, 0)),
    leading_voice_clip: Math.round(finite(previous.leading_voice_clip, 1)),
    // Resets are explicit so a preset works after any manual experimentation.
    schillinger_progression: true,
    use_generated_progression: false,
    use_floor: false,
    use_ceiling: false,
    use_resolve: false,
    beam_width: 3,
    lookahead_depth: 3,
    candidate_range: 4,
    voice_contour_resolution: 4,
    start_notes: [72, 67, 60, 55, 43],
    mode: 0,
    root_contour: flat(root),
    mode_contour: flat(0),
    schillinger_sequence: [] as number[],
    chord_structure: [0, 1, 2],
    chord_structure_contour: voices(() => 0),
    // Third-stacking determines chord identity. Register/voice targets below
    // determine spacing; increasing this factor is not an octave spread.
    schillinger_ex_contour: voices(() => 2),
    chord_templates: [] as number[][],
    chromatic_scale: [] as number[],
    harmony_matrix: DEFAULT_HARMONY_MATRIX.map(row => [...row]),
    harmony_matrix_contour: flat(0),
    harmony_distance_balance: 0.2,
    harmony_distance_contour: flat(0.2),
    last_note_exist_in_voice: 0.5,
    last_note_same: 0.2,
    same_direction: 0,
    consecutive_octav_fift: 5,
    no_crossing: 10,
    interval_exists_in_harmony: 0.4,
    same_note_bonus: 1,
    common_tone_penalty: 0,
    max_common_tones: -1,
    min_voices_changed: -1,
    max_voices_changed: -1,
    voice_contour_weight: 1,
    voice_contour: voices((v, bar) => Math.round((v === 4 ? -2 : v === 0 ? 3 : 1) * arch(bar))),
    melody_force: 0.2,
    melody_force_contour: voices((v, bar) => (v === 0 ? 0.25 : 0.1) * (0.7 + 0.3 * arch(bar))),
    roughness_weight: 0.4,
    chord_quality_weight: 1,
    root_position_weight: 0.5,
    root_doubling_weight: 0.8,
    tendency_weight: 1.5,
    metric_hold_strength: 0.7,
    momentum_weight: 0.4,
    density_weight: 0, // inactive without a uniform lattice
    lattice_beats: 0,
    voice_rhythm: [1],
    voice_rhythm_contour: voices((v, bar) => bar % 4 === 3 ? 2 : v === 0 ? 1 : 2),
    voice_rhythm_patterns: [] as number[][],
    rhythm_generators: [] as number[],
    rhythm_unit: 0.5,
    rhythm_voice_rotation: 0,
    suspension_tolerance: 0.6,
    suspension_resolve_bonus: 1,
    dynamics_weight: 0.8,
    cadence_weight: 1,
    loop_wrap_weight: 0,
    phrase_echo_weight: 0.5,
    fugue_mode: false,
    fugue_voice_count: 3,
    fugue_voice_register_shifts: [] as number[],
    fugue_repeat_subject: false,
    fugue_transform_cycle: DEFAULT_FUGUE_TRANSFORM_CYCLE.map(step => ({ ...step })),
    fugue_entry_spacing: 0,
    fugue_answer_interval: 7,
    fugue_fit_to_scale: false,
    fugue_bass_subject_shift: -12,
    fugue_subject_anticipation: 0,
    fugue_entry_approach: 0,
    soprano_voice: 0,
    bass_voice: 4,
    fugue_output_tracks: [4, 5, 6],
    fugue_output_clip: 2,
  };

  if (kind === 'jazz') {
    Object.assign(nc, {
      candidate_range: 5, no_crossing: 3, consecutive_octav_fift: 0,
      root_position_weight: 0.4, root_doubling_weight: 0, tendency_weight: 1,
      same_note_bonus: 0.5, interval_exists_in_harmony: 0.2,
      harmony_matrix_contour: flat(1), roughness_weight: 0.3,
      chord_structure_contour: voices(() => 1),
      // These pools admit complete seventh chords, including their guide tones.
      chord_templates: [[0, 4, 7, 11], [0, 3, 7, 10], [0, 4, 7, 10], [0, 3, 6, 10]],
      metric_hold_strength: -0.3, suspension_tolerance: 0, suspension_resolve_bonus: 0,
      cadence_weight: 0, loop_wrap_weight: 1, phrase_echo_weight: 0.4,
      voice_rhythm_patterns: [[2 / 3, 1 / 3], [1.5, 0.5, 1, 1], [1.5, 0.5, 1, 1], [2, 1.5, 0.5], [1]],
      voice_rhythm_contour: voices(v => v === 0 ? 0.5 : v === 4 ? 1 : 1.5),
    });
    for (let phrase = 0; phrase < phrases; phrase++) {
      nc.schillinger_sequence.push(...choose([[0, 5, 1, 4], [0, 3, 1, 4]]));
    }
  } else if (kind === 'modal_jazz') {
    Object.assign(nc, {
      mode: 1, mode_contour: flat(1), harmony_matrix_contour: flat(1),
      schillinger_ex_contour: voices(() => 3), // deliberately quartal
      chord_structure_contour: voices(() => 1), // fourth member includes the Dorian sixth on IV
      consecutive_octav_fift: 0, no_crossing: 3,
      root_position_weight: 0.1, root_doubling_weight: 0, chord_quality_weight: 0.3,
      tendency_weight: 0, suspension_tolerance: 0, suspension_resolve_bonus: 0,
      cadence_weight: 0, loop_wrap_weight: 0.3, same_note_bonus: 1.5,
      interval_exists_in_harmony: 0, metric_hold_strength: 0.2,
      voice_rhythm_contour: voices(v => v === 0 ? 1 : v === 4 ? 2 : 4),
    });
    nc.schillinger_sequence = Array.from({ length: bars }, (_, bar) => [0, 0, 0, 0, 3, 3, 0, 0][bar % 8]);
  } else if (kind === 'classical' || kind === 'bach') {
    if (kind === 'bach') {
      nc.mode = random() < 0.6 ? 0 : 5;
      nc.mode_contour = flat(nc.mode);
      nc.root_position_weight = 0.4;
      nc.same_note_bonus = 0.8;
      // Inner voices retain common tones; activity is concentrated in the top
      // line. Chord pools do not claim to model passing-note counterpoint.
      nc.melody_force_contour = voices((v, bar) => v === 0 ? 0.4 + 0.1 * arch(bar) : 0.1);
    }
    for (let phrase = 0; phrase < phrases; phrase++) {
      // Complete eight-bar sequences fit only when a later tonic close remains.
      if (kind === 'bach' && phrase > 0 && phrase + 2 < phrases && random() < 0.5) {
        nc.schillinger_sequence.push(0, 3, 6, 2, 5, 1, 4, 0);
        phrase++;
      } else {
        nc.schillinger_sequence.push(...choose([[0, 3, 4, 0], [0, 1, 4, 0], [0, 5, 4, 0]]));
      }
    }
    // Functional sevenths belong on ii and V; the tonic remains a triad.
    nc.chord_structure_contour = voices((_v, bar) => [1, 4].includes(nc.schillinger_sequence[bar]) ? 1 : 0);
    if (kind === 'bach') {
      nc.voice_rhythm_contour = voices((v, bar) => {
        const tonicClose = bar % 4 === 3 && nc.schillinger_sequence[bar] === 0;
        return v === 0 ? (tonicClose ? 1 : 0.5) : v === 4 ? 1 : 2;
      });
    }
  } else if (kind === 'ambient') {
    Object.assign(nc, {
      mode: 3, mode_contour: flat(3), harmony_matrix_contour: flat(3),
      start_notes: [84, 76, 67, 55, 43],
      root_position_weight: 0.1, root_doubling_weight: 0, chord_quality_weight: 0.3,
      tendency_weight: 0, consecutive_octav_fift: 0, no_crossing: 3,
      last_note_exist_in_voice: 0, last_note_same: 0, same_note_bonus: 3,
      max_voices_changed: 1, interval_exists_in_harmony: 0,
      metric_hold_strength: 0, momentum_weight: 0,
      suspension_tolerance: 0, suspension_resolve_bonus: 0,
      cadence_weight: 0, loop_wrap_weight: 0.2, phrase_echo_weight: 0,
      melody_force: 0, melody_force_contour: voices(() => 0),
      voice_rhythm_contour: voices((v, bar) => (bar + v) % 3 === 0 ? 2 : 4),
      voice_contour: voices((v, bar) => Math.round(2 * Math.sin(bar / bars * Math.PI + v))),
    });
    nc.schillinger_sequence = Array.from({ length: bars }, (_, bar) => Math.floor(bar / 4) % 2);
  } else {
    Object.assign(nc, {
      mode: 5, mode_contour: flat(5), harmony_matrix_contour: flat(4),
      root_position_weight: 2, root_doubling_weight: 1,
      consecutive_octav_fift: 0, no_crossing: 3,
      interval_exists_in_harmony: 0, same_note_bonus: 1.5,
      tendency_weight: 0.3, suspension_tolerance: 0, suspension_resolve_bonus: 0,
      cadence_weight: 0, loop_wrap_weight: 0.5, metric_hold_strength: 0.8,
      voice_rhythm_patterns: [[0.5], [4], [4], [2], [0.5]],
      voice_rhythm_contour: voices(v => v === 0 || v === 4 ? 0.5 : v === 3 ? 2 : 4),
      melody_force_contour: voices(v => v === 0 ? 0.5 : 0.05),
    });
    const pattern = choose([[0, 5, 2, 6], [0, 6, 5, 6], [0, 5, 6, 4], [0, 3, 5, 6]]);
    for (let phrase = 0; phrase < phrases; phrase++) nc.schillinger_sequence.push(...pattern);
  }

  return nc;
}
