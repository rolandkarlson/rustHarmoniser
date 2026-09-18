import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createPreset, DEFAULT_FUGUE_TRANSFORM_CYCLE, PRESET_INFO, resizeFugueOutputTracks, resizeFugueRegisterShifts } from '../src/presets.ts';
import { CHORD_LIST } from '../src/harmony.ts';

const names = Object.keys(PRESET_INFO);
const preserved = {
  root: 7, main_pitch: -12, render_length: 5,
  use_leading_voice: true, leading_voice_track: 2, leading_voice_clip: 3,
};

test('every core musical field resets, independently of the previous preset', () => {
  const model = readFileSync(new URL('../../core/src/model.rs', import.meta.url), 'utf8');
  const configBody = model.split('pub struct Config {')[1].split('\n}')[0];
  const fields = [...configBody.matchAll(/pub (\w+):/g)].map(m => m[1]);
  const polluted = Object.fromEntries(fields.map(f => [f, 'stale setting']));
  Object.assign(polluted, preserved);
  const before = structuredClone(polluted);
  for (const name of names) {
    const clean = createPreset(name, preserved, 42);
    const actual = createPreset(name, polluted, 42);
    for (const field of fields) {
      assert.ok(Object.hasOwn(clean, field), `${name} forgot ${field}`);
      assert.deepEqual(actual[field], clean[field], `${name} inherited ${field}`);
    }
    for (const other of names) {
      assert.deepEqual(createPreset(name, createPreset(other, preserved, 7), 42), clean);
    }
    for (const [key, value] of Object.entries(preserved)) assert.equal(actual[key], value);
  }
  assert.deepEqual(polluted, before, 'must not mutate the live config');
});

test('GUI and Rust agree on chord-member indices', () => {
  const source = readFileSync(new URL('../../core/src/schillinger.rs', import.meta.url), 'utf8');
  const body = source.split('pub const CHORD_STRUCTURES:')[1].split('= [')[1].split('];')[0];
  const rust = [...body.matchAll(/&\[([\d, ]+)\]/g)].map(m => m[1].split(',').map(Number));
  assert.deepEqual(CHORD_LIST, rust);
});

test('tonal Jazz keeps one key and includes guide-tone seventh chords', () => {
  const c = createPreset('jazz', { render_length: 8 }, 42);
  assert.ok(c.mode_contour.every(m => m === 0));
  assert.ok(c.schillinger_ex_contour.flat().every(x => x === 2));
  assert.ok(c.chord_structure_contour.flat().every(x => x === 1));
  assert.deepEqual(c.voice_rhythm_patterns[0], [2 / 3, 1 / 3]);
  for (let i = 0; i < c.schillinger_sequence.length; i += 4) {
    const phrase = c.schillinger_sequence.slice(i, i + 4);
    assert.equal(phrase[0], 0);
    assert.ok([3, 5].includes(phrase[1]));
    assert.deepEqual(phrase.slice(2), [1, 4]);
  }
  assert.equal(c.cadence_weight, 0);
  assert.ok(c.loop_wrap_weight > 0);
});

test('Classical and Bach keep tertian harmony and complete tonic endings', () => {
  for (const name of ['classical', 'bach']) {
    for (let seed = 1; seed <= 30; seed++) {
      const c = createPreset(name, { render_length: 8 }, seed);
      assert.equal(c.schillinger_sequence.length, 32);
      assert.deepEqual(c.schillinger_sequence.slice(-2), [4, 0]);
      assert.ok(c.schillinger_ex_contour.flat().every(x => x === 2));
      assert.equal(c.chord_structure_contour[0].at(-1), 0);
      assert.ok(c.cadence_weight > 0);
      assert.equal(c.loop_wrap_weight, 0);
      if (name === 'bach') {
        for (let bar = 3; bar < 32; bar += 4) {
          assert.equal(c.voice_rhythm_contour[0][bar], c.schillinger_sequence[bar] === 0 ? 1 : 0.5);
        }
      }
    }
  }
});

test('modal, ambient and dance settings match their distinct textures', () => {
  const modal = createPreset('modal_jazz', {}, 42);
  assert.ok(modal.mode_contour.every(m => m === 1));
  assert.ok(modal.schillinger_ex_contour.flat().every(x => x === 3));
  assert.ok(modal.chord_structure_contour.flat().every(x => x === 1));
  assert.equal(modal.tendency_weight, 0);
  const ambient = createPreset('ambient', {}, 42);
  assert.ok(ambient.schillinger_ex_contour.flat().every(x => x === 2));
  assert.deepEqual(ambient.schillinger_sequence, [0, 0, 0, 0, 1, 1, 1, 1]);
  assert.equal(ambient.suspension_resolve_bonus, 0);
  assert.equal(ambient.cadence_weight, 0);
  const trance = createPreset('trance', { render_length: 5 }, 42);
  assert.ok(trance.chord_structure_contour.flat().every(x => x === 0));
  for (let bar = 4; bar < 20; bar++) {
    assert.equal(trance.schillinger_sequence[bar], trance.schillinger_sequence[bar % 4]);
  }
  assert.deepEqual(trance.voice_rhythm_patterns[4], [0.5]);
});

test('presets reset fugue repeats to an independent default transform cycle', () => {
  const first = createPreset('bach', {}, 42);
  const second = createPreset('bach', {}, 42);
  assert.equal(first.fugue_repeat_subject, false);
  assert.equal(first.fugue_fit_to_scale, false);
  assert.deepEqual(first.fugue_transform_cycle, DEFAULT_FUGUE_TRANSFORM_CYCLE);
  first.fugue_transform_cycle[0].invert = false;
  assert.equal(second.fugue_transform_cycle[0].invert, true, 'preset cycles must not share mutable rows');
});

test('fugue voice helpers preserve custom values and safely fill new slots', () => {
  assert.deepEqual(resizeFugueRegisterShifts([1, 0, -13], 5), [1, 0, -13, -24, -24]);
  assert.deepEqual(resizeFugueOutputTracks([7, 2, 7, -1], 5), [7, 2, 8, 9, 10]);
  assert.deepEqual(resizeFugueOutputTracks([2_147_483_647], 3), [2_147_483_647, 0, 1]);
  assert.deepEqual(resizeFugueOutputTracks([1e30, 4], 3), [4, 5, 6]);
  assert.equal(createPreset('bach', {}, 42).fugue_voice_count, 3);
  assert.deepEqual(createPreset('bach', {}, 42).fugue_voice_register_shifts, []);
});
