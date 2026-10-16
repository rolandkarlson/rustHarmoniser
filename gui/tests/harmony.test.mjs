import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { CHORD_LIST, DEFAULT_MODE_SCALES, getModeScales, getChordStructures, chordPitchClasses, parseModeScale } from '../src/harmony.ts';

test('GUI and engine agree on all default mode degrees', () => {
  const source = readFileSync(new URL('../../core/src/music_theory.rs', import.meta.url), 'utf8');
  const body = source.split('pub const DEFAULT_MODE_SCALES:')[1].split('= [')[1].split('];')[0];
  const rust = [...body.matchAll(/\[([\d, ]+)\]/g)].map(m => m[1].split(',').map(Number));
  assert.deepEqual(DEFAULT_MODE_SCALES, rust);
});

test('old, malformed and partially valid tables resolve without mutating defaults', () => {
  for (const value of [undefined, null, [], 'bad', [[0]]]) {
    assert.deepEqual(getModeScales(value), DEFAULT_MODE_SCALES);
    assert.deepEqual(getChordStructures(value), CHORD_LIST);
  }
  const modes = getModeScales(null);
  modes[0] = [0, 2, 3, 5, 7, 8, 11];
  for (const invalid of [[], [0, 2, 3, 5, 7, 8, 12], [0, 2, 3, 5, 7, 7, 11], [1, 2, 3, 5, 7, 8, 11], [0, 2, 3.5, 5, 7, 8, 11], [0, 3, 2, 5, 7, 8, 11]]) {
    modes[1] = invalid;
    assert.deepEqual(getModeScales(modes)[0], modes[0]);
    assert.deepEqual(getModeScales(modes)[1], DEFAULT_MODE_SCALES[1]);
  }
  const chords = getChordStructures(null);
  chords[0] = [0, 3];
  for (const invalid of [[], [1, 2], [0, 7], [0, -1], [0, 2, 2], [0, 1.5], [0, 3, 1]]) {
    chords[1] = invalid;
    assert.deepEqual(getChordStructures(chords)[0], [0, 3]);
    assert.deepEqual(getChordStructures(chords)[1], CHORD_LIST[1]);
  }
  getModeScales(null)[0][1] = 1;
  getChordStructures(null)[0].push(6);
  assert.deepEqual(DEFAULT_MODE_SCALES[0], [0, 2, 4, 5, 7, 9, 11]);
  assert.deepEqual(CHORD_LIST[0], [0, 1, 2]);
});

test('comma-separated scales accept different lengths and reject incomplete or invalid input', () => {
  for (const row of [[0], [0, 7], [0, 2, 4, 7, 9], [0, 2, 4, 6, 8, 10], [0, 2, 3, 5, 6, 8, 9, 11], Array.from({ length: 12 }, (_, i) => i)]) {
    assert.deepEqual(parseModeScale(` ${row.join(', ')} `), row);
    const scales = getModeScales(null);
    scales[0] = row;
    assert.deepEqual(getModeScales(scales)[0], row);
  }
  for (const text of ['', ' ', '0,', ',0', '0,,2', '0,2,2', '0,4,2', '1,2,4', '0,12', '0,-1', '0,2.5', '0,NaN', '0,2x', '0 2 4']) {
    assert.equal(parseModeScale(text), null, text);
  }
});

test('chord highlights wrap using the entered scale length', () => {
  const scales = getModeScales(null);
  scales[0] = [0, 2, 4, 7, 9];
  assert.deepEqual(chordPitchClasses(0, 0, 2, CHORD_LIST[0], scales), [0, 4, 9]);
  assert.deepEqual(chordPitchClasses(0, 6, 2, CHORD_LIST[0], scales), [2, 7, 0]);
  assert.deepEqual(chordPitchClasses(0, -1, 2, CHORD_LIST[0], scales), [9, 2, 7]);
  scales[0] = [0, 2, 3, 5, 6, 8, 9, 11];
  assert.deepEqual(chordPitchClasses(0, 6, 2, CHORD_LIST[0], scales), [9, 0, 3]);
  scales[0] = [0];
  assert.deepEqual(chordPitchClasses(0, -1, 2, CHORD_LIST[0], scales), [0, 0, 0]);
});

test('mode highlights use custom scales, member sets, expansion and Aeolian inflection', () => {
  const scales = getModeScales(null);
  assert.deepEqual(chordPitchClasses(0, 0, 2, CHORD_LIST[0], scales), [0, 4, 7]);
  scales[0] = [0, 2, 3, 5, 7, 8, 11];
  assert.deepEqual(chordPitchClasses(0, 0, 2, [0, 1, 3], scales), [0, 3, 11]);
  assert.deepEqual(chordPitchClasses(0, -1, 1, [0, 1, 3], scales), [11, 0, 3]);
  assert.deepEqual(chordPitchClasses(5, 4, 2, CHORD_LIST[0], scales), [7, 11, 2]);
  scales[5] = [0, 2, 3, 6, 7, 8, 10];
  assert.deepEqual(chordPitchClasses(5, 4, 2, CHORD_LIST[0], scales), [7, 10, 2]);
});
