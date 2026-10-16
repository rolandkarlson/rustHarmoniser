/** Editable chord-root generators. Degrees are zero-based, as in the engine. */
export type ProgressionMethod = 'markov' | 'cycles';
export type ProgressionEnding = 'resolve' | 'loop' | 'free';
export type Motion = 'balanced' | 'fifths' | 'thirds' | 'steps';
export type RootCycle = 'fifths' | 'thirds' | 'steps' | 'mixed';

export interface ProgressionOptions {
  method: ProgressionMethod;
  ending: ProgressionEnding;
  motion: Motion;
  cycle: RootCycle;
  seed: number;
}

export interface ProgressionConfig {
  pl: number;
  render_length: number;
  mode?: number;
  mode_contour?: number[] | null;
  voice_contour_resolution?: number;
  schillinger_sequence?: number[];
  schillinger_progression?: boolean;
  use_generated_progression?: boolean;
}

export const DEFAULT_PROGRESSION_OPTIONS: ProgressionOptions = {
  method: 'markov', ending: 'resolve', motion: 'balanced', cycle: 'fifths', seed: 1,
};
export const DEGREE_NAMES = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'];
export const MODE_LABELS = ['Ionian / major', 'Dorian', 'Phrygian', 'Lydian', 'Mixolydian', 'Aeolian / minor', 'Locrian'];
// Same characteristic closes as core/src/schillinger.rs (not V–I in every mode).
export const CADENCE_DEGREES = [4, 3, 1, 4, 6, 4, 1];

type Transition = readonly [degree: number, weight: number];
// Major/minor root routes adapted from the Musicnotes diagrams:
// https://www.musicnotes.com/blog/chord-progression-formulas/
// Weights and modal extensions are our choices, not probabilities in the article.
// Aeolian degree VII stays the natural subtonic: the engine cannot encode both
// VII and raised vii° as distinct roots. Its V already uses harmonic minor.
export const FUNCTIONAL_TRANSITIONS: readonly (readonly (readonly Transition[])[])[] = [
  [ // Ionian
    [[4, 30], [3, 25], [5, 20], [1, 10], [2, 10], [6, 5]],
    [[4, 80], [6, 20]], [[5, 70], [3, 30]],
    [[1, 35], [4, 55], [6, 10]], [[0, 70], [5, 20], [6, 10]],
    [[3, 50], [1, 50]], [[0, 70], [4, 20], [5, 10]],
  ],
  [ // Dorian
    [[3, 50], [6, 30], [1, 20]], [[0, 70], [3, 30]], [[3, 60], [4, 40]],
    [[0, 60], [6, 40]], [[0, 50], [3, 50]], [[6, 50], [3, 50]], [[0, 70], [3, 30]],
  ],
  [ // Phrygian
    [[1, 50], [3, 30], [5, 20]], [[0, 100]], [[1, 60], [3, 40]],
    [[0, 60], [1, 40]], [[1, 50], [5, 50]], [[1, 60], [0, 40]], [[0, 60], [2, 40]],
  ],
  [ // Lydian
    [[1, 50], [4, 30], [2, 20]], [[0, 60], [4, 40]], [[0, 50], [1, 50]],
    [[4, 60], [2, 40]], [[0, 60], [1, 40]], [[1, 50], [4, 50]], [[0, 70], [2, 30]],
  ],
  [ // Mixolydian
    [[3, 40], [6, 40], [4, 20]], [[0, 50], [3, 50]], [[3, 60], [5, 40]],
    [[0, 60], [6, 40]], [[0, 60], [3, 40]], [[3, 50], [1, 50]], [[0, 60], [3, 40]],
  ],
  [ // Aeolian
    [[3, 25], [4, 30], [5, 20], [6, 10], [2, 10], [1, 5]],
    [[4, 100]], [[5, 70], [3, 30]], [[1, 40], [4, 60]],
    [[0, 75], [5, 25]], [[3, 50], [1, 50]], [[2, 100]],
  ],
  [ // Locrian
    [[1, 40], [3, 30], [5, 30]], [[0, 60], [3, 40]], [[1, 50], [5, 50]],
    [[0, 60], [1, 40]], [[1, 50], [5, 50]], [[0, 60], [1, 40]], [[0, 50], [5, 50]],
  ],
];

const mod7 = (n: number) => ((n % 7) + 7) % 7;
export function progressionModeAt(config: ProgressionConfig, bar: number): number {
  const contour = config.mode_contour;
  const resolution = config.voice_contour_resolution ?? 4;
  const value = contour?.length && resolution > 0
    ? contour[Math.floor(bar * 4 / resolution) % contour.length]
    : config.mode ?? 0;
  return Number.isFinite(value) ? mod7(Math.round(value)) : 0;
}

function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let t = Math.imul(state ^ state >>> 15, 1 | state);
    t ^= t + Math.imul(t ^ t >>> 7, 61 | t);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function choose(options: readonly Transition[], random: () => number): number {
  let remaining = random() * options.reduce((sum, [, weight]) => sum + weight, 0);
  for (const [degree, weight] of options) {
    remaining -= weight;
    if (remaining < 0) return degree;
  }
  return options[options.length - 1][0];
}

function markovPhrase(length: number, mode: number, ending: ProgressionEnding, motion: Motion, random: () => number): number[] {
  if (length === 1) return [0];
  const target = CADENCE_DEGREES[mode];
  if (length === 2 && ending === 'resolve') return [target, 0];
  const size = length - (ending === 'resolve' ? 1 : 0);
  const preferred = { balanced: -1, fifths: 3, thirds: 5, steps: 1 }[motion];
  const transitions = FUNCTIONAL_TRANSITIONS[mode].map((row, from) => {
    const weighted: Transition[] = row.map(([to, weight]) => [to, weight * (mod7(to - from) === preferred ? 4 : 1)]);
    const total = weighted.reduce((sum, [, weight]) => sum + weight, 0);
    return weighted.map(([to, weight]): Transition => [to, weight / total]);
  });
  // Backward probabilities condition the walk on a reachable ending. This
  // avoids random retries, exponential backtracking, and forced illegal edges.
  const suffix = Array.from({ length: size }, () => Array<number>(7).fill(0));
  suffix[size - 1] = Array.from({ length: 7 }, (_, d) => ending === 'free' || d === target ? 1 : 0);
  for (let i = size - 2; i >= 0; i--) {
    suffix[i] = transitions.map(row => row.reduce((sum, [to, p]) => sum + p * suffix[i + 1][to], 0));
    const max = Math.max(...suffix[i]);
    if (max > 0) suffix[i] = suffix[i].map(p => p / max);
  }
  const sequence = [0];
  for (let i = 1; i < size; i++) {
    const options = transitions[sequence[i - 1]]
      .map(([to, p]): Transition => [to, p * suffix[i][to]])
      .filter(([, p]) => p > 0);
    if (!options.length) throw new Error('No progression reaches this ending. Choose Free movement.');
    sequence.push(choose(options, random));
  }
  if (ending === 'resolve') sequence.push(0);
  return sequence;
}

// Diatonic root cycles, inspired by Schillinger's separate root-motion layer:
// descending fifth = -4 degree steps; descending third = -2; ascending second = +1.
// Mixed cycles and the optional cadential tail are compositional choices here,
// not a complete implementation of the Schillinger System.
const CYCLES: Record<RootCycle, readonly number[]> = {
  fifths: [-4], thirds: [-2], steps: [1], mixed: [-4, -2, 1],
};

export function generateChordProgression(config: ProgressionConfig, options: ProgressionOptions): number[] {
  const { pl, render_length: phrases } = config;
  if (!Number.isSafeInteger(pl) || pl < 1 || !Number.isSafeInteger(phrases) || phrases < 1 || pl * phrases > 8192) {
    throw new Error('Use positive whole numbers for PL and Render Len, with at most 8192 bars.');
  }
  if (!Number.isSafeInteger(options.seed) || options.seed < 0 || options.seed > 4294967295) {
    throw new Error('Use a whole-number seed between 0 and 4294967295.');
  }
  const random = seededRandom(options.seed);
  const sequence: number[] = [];
  for (let phrase = 0; phrase < phrases; phrase++) {
    const mode = progressionModeAt(config, phrase * pl);
    if (options.method === 'markov') {
      sequence.push(...markovPhrase(pl, mode, options.ending, options.motion, random));
      continue;
    }
    const cycle = CYCLES[options.cycle];
    // A mixed pattern varies its rotation per phrase; a pure cycle is fixed.
    const rotation = cycle.length > 1 ? Math.floor(random() * cycle.length) : 0;
    const block = [0];
    for (let i = 1; i < pl; i++) block.push(mod7(block[i - 1] + cycle[(i - 1 + rotation) % cycle.length]));
    if (pl > 1 && options.ending !== 'free') {
      block[pl - (options.ending === 'resolve' ? 2 : 1)] = CADENCE_DEGREES[mode];
      if (options.ending === 'resolve') block[pl - 1] = 0;
    }
    sequence.push(...block);
  }
  return sequence;
}

/** The literal editor sequence must win over the engine's Cadence override. */
export function fillProgression<T extends ProgressionConfig>(config: T, options: ProgressionOptions) {
  return {
    ...config,
    schillinger_sequence: generateChordProgression(config, options),
    schillinger_progression: true,
    use_generated_progression: false,
  };
}
