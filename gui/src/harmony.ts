// Keep these structures in sync with core/src/schillinger.rs.
// At expansion 2, indices name tertian chord members, not MIDI intervals.
export const CHORD_LIST = [
  [0,1,2], [0,1,2,3], [0,1,2,6],
  [0,1,2,3,4], [0,1,2,3,4,5], [0,1,2,3,4,5,6]
];

export const MODE_NAMES = ['Ion', 'Dor', 'Phr', 'Lyd', 'Mix', 'Aeo', 'Loc'];
export const MODE_LABELS = ['Ionian', 'Dorian', 'Phrygian', 'Lydian', 'Mixolydian', 'Aeolian', 'Locrian'];
// Seven scale-degree offsets from the tonic; mirrors music_theory.rs.
export const DEFAULT_MODE_SCALES = [
  [0, 2, 4, 5, 7, 9, 11],
  [0, 2, 3, 5, 7, 9, 10],
  [0, 1, 3, 5, 7, 8, 10],
  [0, 2, 4, 6, 7, 9, 11],
  [0, 2, 4, 5, 7, 9, 10],
  [0, 2, 3, 5, 7, 8, 10],
  [0, 1, 3, 5, 6, 8, 10],
];

function validOffsets(row: unknown, limit: number): row is number[] {
  return Array.isArray(row) && row.length > 0 && row[0] === 0
    && row.every((n, i) => Number.isInteger(n) && n >= 0 && n < limit && (i === 0 || n > row[i - 1]));
}

// Resolve each slot independently, matching the engine's legacy-config fallback.
export function getModeScales(custom: unknown): number[][] {
  return DEFAULT_MODE_SCALES.map((fallback, i) => {
    const row = Array.isArray(custom) && custom.length === 7 ? custom[i] : undefined;
    return validOffsets(row, 12) ? [...row] : [...fallback];
  });
}

/** A literal scale of 1–12 ascending pitch-class offsets, starting at the tonic. */
export function parseModeScale(text: string): number[] | null {
  const parts = text.split(',').map(part => part.trim());
  if (!parts.every(part => /^\d+$/.test(part))) return null;
  const row = parts.map(Number);
  return validOffsets(row, 12) ? row : null;
}

export function getChordStructures(custom: unknown): number[][] {
  return CHORD_LIST.map((fallback, i) => {
    const row = Array.isArray(custom) && custom.length === 6 ? custom[i] : undefined;
    return validOffsets(row, 7) ? [...row] : [...fallback];
  });
}

export function isCustomRow(row: number[], fallback: number[]): boolean {
  return row.length !== fallback.length || row.some((value, i) => value !== fallback[i]);
}

// Same degree expansion and default Aeolian dominant inflection as the engine.
// Offsets suffice for comparing chords: transposing both preserves equality.
export function chordPitchClasses(mode: number, degree: number, expansion: number, members: number[], scales: number[][]): number[] {
  const slot = ((mode % 7) + 7) % 7;
  const scale = scales[slot];
  const raiseLeadingTone = slot === 5 && !isCustomRow(scale, DEFAULT_MODE_SCALES[5]) && ((degree % 7) + 7) % 7 === 4;
  return members.map(member => {
    const index = (((member * expansion + degree) % scale.length) + scale.length) % scale.length;
    return (scale[index] + (raiseLeadingTone && index === 6 ? 1 : 0)) % 12;
  });
}

// Harmony scoring matrix — 9 style rows × 12 interval columns.
// Mirrors the built-in default in src/harmonizer.rs (HARMONY_MATRIX).
export const HARMONY_MATRIX_ROWS = [
  'Classical', 'Jazz', 'Tension', 'Ethereal', 'Dark', 'Bright', 'Aggressive', 'Ancient', 'Neutral'
];
export const HARMONY_MATRIX_COLS = ['P1', 'm2', 'M2', 'm3', 'M3', 'P4', 'TT', 'P5', 'm6', 'M6', 'm7', 'M7'];
export const DEFAULT_HARMONY_MATRIX: number[][] = [
  // Row 0 STRICT CLASSICAL — the tritone is strongly disfavoured but NOT
  // forbidden (-0.5): hard-forbidding it outlaws the dominant seventh and the
  // diminished triad. Mirrors harmonizer::HARMONY_MATRIX; keep the two in sync.
  [1.0, -100.0, -0.4, 0.8, 0.9, 0.5, -0.5, 1.0, 0.7, 0.8, -0.3, -100.0],
  [0.6, 0.0, 0.7, 0.8, 0.9, 0.5, 0.6, 0.9, 0.5, 0.8, 1.0, 0.8],
  [-0.2, 0.8, 0.2, -0.3, -0.3, -0.2, 1.0, 0.0, -0.3, -0.3, 0.5, 0.9],
  [1.0, -100.0, 0.8, -0.2, 0.2, 1.0, -0.5, 1.0, 0.0, 0.5, 0.7, -0.4],
  [1.0, -0.5, -0.1, 1.0, -0.2, 0.3, -0.2, 0.8, 0.6, 0.2, 0.5, -0.6],
  [1.0, -0.7, 0.5, -0.1, 1.0, -0.2, 0.8, 0.9, 0.2, 0.6, -0.3, 0.6],
  [-0.5, 1.0, 0.4, -0.6, -0.6, -0.4, 1.0, -0.5, -0.6, -0.6, 0.5, 1.0],
  [1.0, -100.0, -100.0, -100.0, -100.0, 1.0, -100.0, 1.0, -100.0, -100.0, -100.0, -100.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
];

export const CHORD_NAMES = [
  'Triad', '7th', 'Diatonic 6th', '9th', '11th', '13th',
];
