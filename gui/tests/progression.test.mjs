import test from 'node:test';
import assert from 'node:assert/strict';
import {
  CADENCE_DEGREES, DEFAULT_PROGRESSION_OPTIONS, FUNCTIONAL_TRANSITIONS,
  fillProgression, generateChordProgression, progressionModeAt,
} from '../src/progression.ts';

const config = { pl: 8, render_length: 4, mode: 0, voice_contour_resolution: 4 };
const generate = (c = config, o = {}) => generateChordProgression(c, { ...DEFAULT_PROGRESSION_OPTIONS, ...o });

test('Markov phrases use valid transitions and characteristic endings in every mode', () => {
  for (let mode = 0; mode < 7; mode++) {
    for (const pl of [1, 2, 3, 4, 8, 32, 128]) {
      for (const ending of ['resolve', 'loop', 'free']) {
        for (const motion of ['balanced', 'fifths', 'thirds', 'steps']) {
          const sequence = generate({ ...config, mode, pl }, { ending, motion, seed: pl * 71 });
          assert.equal(sequence.length, pl * config.render_length);
          assert.ok(sequence.every(d => Number.isInteger(d) && d >= 0 && d <= 6));
          for (let bar = 0; bar < sequence.length; bar += pl) {
            const phrase = sequence.slice(bar, bar + pl);
            if (pl === 1) { assert.deepEqual(phrase, [0]); continue; }
            if (!(pl === 2 && ending === 'resolve')) assert.equal(phrase[0], 0);
            if (ending === 'resolve') assert.deepEqual(phrase.slice(-2), [CADENCE_DEGREES[mode], 0]);
            if (ending === 'loop') assert.equal(phrase.at(-1), CADENCE_DEGREES[mode]);
            for (let i = 1; i < phrase.length; i++) {
              assert.ok(FUNCTIONAL_TRANSITIONS[mode][phrase[i - 1]].some(([to]) => to === phrase[i]),
                `illegal transition in mode ${mode}: ${phrase[i - 1]} → ${phrase[i]}`);
            }
          }
        }
      }
    }
  }
});

test('seeds reproduce progressions and supply different alternatives', () => {
  assert.deepEqual(generate(config, { seed: 42 }), generate(config, { seed: 42 }));
  assert.notDeepEqual(generate(config, { seed: 42 }), generate(config, { seed: 43 }));
  const options = { method: 'cycles', cycle: 'mixed', ending: 'free' };
  assert.deepEqual(generate(config, { ...options, seed: 42 }), generate(config, { ...options, seed: 42 }));
  assert.notDeepEqual(generate(config, { ...options, seed: 42 }), generate(config, { ...options, seed: 43 }));
});

test('motion preference changes the distribution of legal free-walk choices', () => {
  const c = { ...config, pl: 1024, render_length: 1 };
  const count = (seq, step) => seq.slice(1).filter((d, i) => (d - seq[i] + 7) % 7 === step).length;
  const balanced = generate(c, { ending: 'free' });
  for (const [motion, step] of [['fifths', 3], ['thirds', 5], ['steps', 1]]) {
    assert.ok(count(generate(c, { motion, ending: 'free' }), step) > count(balanced, step));
  }
});

test('root cycles wrap diatonic degrees, reset each phrase, and preserve their selected motion', () => {
  const expected = {
    fifths: [0, 3, 6, 2, 5, 1, 4, 0],
    thirds: [0, 5, 3, 1, 6, 4, 2, 0],
    steps: [0, 1, 2, 3, 4, 5, 6, 0],
  };
  for (const [cycle, phrase] of Object.entries(expected)) {
    assert.deepEqual(generate(config, { method: 'cycles', ending: 'free', cycle }), Array(4).fill(phrase).flat());
    assert.deepEqual(generate(config, { method: 'cycles', ending: 'free', cycle, seed: 123 }), Array(4).fill(phrase).flat());
  }
  for (let mode = 0; mode < 7; mode++) {
    for (const pl of [1, 2, 4, 8]) {
      const c = { ...config, mode, pl, render_length: 1 };
      const resolved = generate(c, { method: 'cycles', cycle: 'thirds' });
      assert.deepEqual(pl > 1 ? resolved.slice(-2) : resolved, pl > 1 ? [CADENCE_DEGREES[mode], 0] : [0]);
      const loop = generate(c, { method: 'cycles', ending: 'loop' });
      assert.equal(loop.at(-1), pl > 1 ? CADENCE_DEGREES[mode] : 0);
    }
  }
});

test('phrases sample the looping mode contour at engine beat resolution', () => {
  const c = { ...config, pl: 4, render_length: 3, mode: 5, mode_contour: [2, 2, 4, 4], voice_contour_resolution: 8 };
  assert.deepEqual([0, 4, 8].map(bar => progressionModeAt(c, bar)), [2, 4, 2]);
  const seq = generate(c);
  assert.deepEqual([seq.slice(2, 4), seq.slice(6, 8), seq.slice(10, 12)], [[1, 0], [6, 0], [1, 0]]);
  assert.equal(progressionModeAt({ ...c, mode_contour: [] }, 1), 5);
  assert.equal(progressionModeAt({ ...c, mode_contour: null }, 1), 5);
  assert.equal(progressionModeAt({ ...c, mode: -1, mode_contour: [] }, 0), 6);
});

test('filling changes only the editor sequence and the two switches needed to hear it', () => {
  const c = {
    ...config, root: 7, rng_seed: 77, mode_contour: [0, 5], voice_contour: [[1, 2]],
    rhythm_generators: [3, 2], chord_structure_contour: [[1]], schillinger_ex_contour: [[3]],
    schillinger_sequence: [6, 5], schillinger_progression: false, use_generated_progression: true,
  };
  const before = structuredClone(c);
  const next = fillProgression(c, DEFAULT_PROGRESSION_OPTIONS);
  assert.deepEqual(c, before);
  assert.equal(next.schillinger_progression, true);
  assert.equal(next.use_generated_progression, false);
  assert.equal(next.schillinger_sequence.length, 32);
  for (const field of Object.keys(c).filter(f => !['schillinger_sequence', 'schillinger_progression', 'use_generated_progression'].includes(f))) {
    assert.deepEqual(next[field], c[field], field);
  }
});

test('invalid dimensions and seeds fail clearly instead of hanging or silently truncating', () => {
  for (const pl of [0, -1, 2.5, NaN, Infinity, 8193]) assert.throws(() => generate({ ...config, pl }), /PL and Render Len/);
  for (const render_length of [0, -1, 2.5, Infinity]) assert.throws(() => generate({ ...config, render_length }), /PL and Render Len/);
  for (const seed of [NaN, Infinity, -1, 1.5, 4294967296]) assert.throws(() => generate(config, { seed }), /seed/);
  assert.equal(generate({ ...config, pl: 8192, render_length: 1 }).length, 8192);
  assert.equal(generate(config, { seed: 0 }).length, 32);
});
