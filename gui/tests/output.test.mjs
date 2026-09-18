import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import ts from 'typescript';

const source = await readFile(new URL('../src/output.ts', import.meta.url), 'utf8');
const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
const output = await import(`data:text/javascript;base64,${Buffer.from(js).toString('base64')}`);

test('raw routes preserve articulations and use the legacy output slot', () => {
  const routes = output.rawRoutes(5);
  assert.deepEqual(routes.map(r => r.track_index), [4, 5, 6, 7, 8]);
  assert.ok(routes.every(r => r.clip_index === 2));
  assert.ok(routes.every(r => r.merge_adjacent === false));
});

test('initial output migrates fugue targets once and respects saved custom routing', () => {
  let stored = null;
  globalThis.localStorage = { getItem: () => stored, setItem: (_key, value) => { stored = value; } };
  const migrated = output.initialOutput({ fugue_mode: true, fugue_voice_count: 3, fugue_output_tracks: [9, 10, 11], fugue_output_clip: 4 });
  assert.deepEqual(migrated.routes.map(r => [r.track_index, r.clip_index]), [[9, 4], [10, 4], [11, 4]]);
  const custom = { enabled: true, routes: [{ ...migrated.routes[0], name: 'Kept custom' }] };
  output.saveOutput(custom);
  assert.deepEqual(output.initialOutput({ fugue_mode: true, fugue_voice_count: 8 }).routes, custom.routes);
});

test('partial voice routing is a warning rather than a validation error', () => {
  const settings = { enabled: true, routes: [output.route('Lead only', [0])] };
  assert.deepEqual(output.validateOutput(settings), []);
  assert.deepEqual(output.unmappedVoices(settings, 3), [1, 2]);
});

test('list parsing accepts fractional values after the draft is committed', () => {
  assert.deepEqual(output.parseNumberList('0.25, 0.5, 1'), [0.25, 0.5, 1]);
});
