// Keep these structures in sync with core/src/schillinger.rs.
// At expansion 2, indices name tertian chord members, not MIDI intervals.
export const CHORD_LIST = [
  [0,1,2], [0,1,2,3], [0,1,2,6],
  [0,1,2,3,4], [0,1,2,3,4,5], [0,1,2,3,4,5,6]
];

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
