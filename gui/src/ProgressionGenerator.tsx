import { useState } from 'react';
import {
  CADENCE_DEGREES, DEFAULT_PROGRESSION_OPTIONS, DEGREE_NAMES, MODE_LABELS,
  fillProgression, progressionModeAt,
  type ProgressionConfig, type ProgressionOptions,
} from './progression';

const controlClass = 'rounded border border-slate-700 bg-slate-900 px-2 py-1.5 text-sm text-slate-200 focus:border-cyan-500 outline-none';

export function ProgressionGenerator<T extends ProgressionConfig>({ config, onApply }: {
  config: T;
  onApply: (next: ReturnType<typeof fillProgression<T>>, message: string) => void;
}) {
  const [options, setOptions] = useState(DEFAULT_PROGRESSION_OPTIONS);
  const [seed, setSeed] = useState('1');
  const [error, setError] = useState('');
  const setOption = <K extends keyof ProgressionOptions>(key: K, value: ProgressionOptions[K]) =>
    setOptions(current => ({ ...current, [key]: value }));
  const mode = progressionModeAt(config, 0);
  const sequence = config.schillinger_sequence ?? [];
  const fixedCycle = options.method === 'cycles' && options.cycle !== 'mixed';

  const apply = (newVariation: boolean) => {
    const nextSeed = fixedCycle ? 1 : newVariation ? Math.floor(Math.random() * 4294967296) : seed.trim() === '' ? NaN : Number(seed);
    try {
      const next = fillProgression(config, { ...options, seed: nextSeed });
      setSeed(String(nextSeed));
      setError('');
      onApply(next, `${options.method === 'markov' ? 'Functional Markov' : 'Schillinger-inspired root cycle'} filled ${next.schillinger_sequence.length} bars — Schillinger enabled, automatic Cadence off`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <section aria-label="Chord progression generator" className="bg-slate-950/60 border border-slate-800 rounded-lg p-3 space-y-3">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h2 className="text-sm font-bold text-orange-300">Chord progression</h2>
        <span className="text-xs text-slate-400">{config.pl * config.render_length} bars · {config.render_length} phrases of {config.pl} · starts in {MODE_LABELS[mode]}</span>
      </div>
      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1 text-xs text-slate-400">
          Method
          <select aria-label="Progression method" className={controlClass} value={options.method} onChange={e => setOption('method', e.target.value as ProgressionOptions['method'])}>
            <option value="markov">Functional (Markov)</option>
            <option value="cycles">Schillinger-inspired cycles</option>
          </select>
        </label>
        {options.method === 'markov' ? (
          <label className="flex flex-col gap-1 text-xs text-slate-400">
            Favor root movement
            <select aria-label="Root movement preference" className={controlClass} value={options.motion} onChange={e => setOption('motion', e.target.value as ProgressionOptions['motion'])}>
              <option value="balanced">Balanced</option>
              <option value="fifths">Descending fifths</option>
              <option value="thirds">Descending thirds</option>
              <option value="steps">Ascending steps</option>
            </select>
          </label>
        ) : (
          <label className="flex flex-col gap-1 text-xs text-slate-400">
            Root cycle
            <select aria-label="Root cycle" className={controlClass} value={options.cycle} onChange={e => setOption('cycle', e.target.value as ProgressionOptions['cycle'])}>
              <option value="fifths">Descending fifths</option>
              <option value="thirds">Descending thirds</option>
              <option value="steps">Ascending steps</option>
              <option value="mixed">Mixed fifths / thirds / steps</option>
            </select>
          </label>
        )}
        <label className="flex flex-col gap-1 text-xs text-slate-400">
          Phrase ending
          <select aria-label="Progression phrase ending" className={controlClass} value={options.ending} onChange={e => setOption('ending', e.target.value as ProgressionOptions['ending'])}>
            <option value="resolve">Resolve to tonic</option>
            <option value="loop">Lead into next phrase</option>
            <option value="free">Free movement</option>
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-slate-400">
          Progression seed
          <input aria-label="Progression seed" type="number" min="0" max="4294967295" step="1" value={seed} disabled={fixedCycle}
            onChange={e => setSeed(e.target.value)} className={`${controlClass} w-32 disabled:opacity-40`} />
        </label>
        <button onClick={() => apply(false)} className="px-3 py-1.5 bg-orange-700 hover:bg-orange-600 text-white font-bold rounded-lg transition-colors">
          Fill Schillinger
        </button>
        <button onClick={() => apply(true)} disabled={fixedCycle}
          title={fixedCycle ? 'A single root cycle is fixed. Choose a mixed cycle for seeded variations.' : 'Choose a new progression seed and fill the sequence again'}
          className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-100 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed">
          New variation
        </button>
      </div>
      <p className="text-xs leading-relaxed text-slate-400">
        {options.method === 'markov'
          ? <>Weighted chord choices inspired by the <a href="https://www.musicnotes.com/blog/chord-progression-formulas/" target="_blank" rel="noreferrer" className="text-cyan-400 underline">Musicnotes formulas</a>, with modal adaptations. Each phrase uses its starting mode; the Mode contour still colors each bar.</>
          : <>Repeating diatonic root movements inspired by <a href="https://www.berklee.edu/berklee-today/fall-2000/The-Schillinger" target="_blank" rel="noreferrer" className="text-cyan-400 underline">Schillinger’s root cycles</a>. Free movement keeps the cycle intact; other endings replace its last one or two roots. A single cycle is fixed; Mixed varies with the seed.</>}
        {' '}Fills one chord root per bar and opens the Schillinger editor. Enables Schillinger and turns off automatic Cadence; keeps your key, rhythms, voicings and scoring settings.
      </p>
      {options.ending !== 'free' && (
        <p className="text-xs text-slate-500">
          First phrase close: {DEGREE_NAMES[CADENCE_DEGREES[mode]]} → I{options.ending === 'loop' ? ' across the phrase boundary' : ''} (root degrees).
          {' '}One-bar phrases stay on I; two-bar resolved phrases contain only the cadence.
        </p>
      )}
      {error && <p role="alert" className="text-sm text-red-400">{error}</p>}
      {sequence.length > 0 && (
        <div className="border-t border-slate-800 pt-2">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Current root degrees · {sequence.length} bars</div>
          <p aria-label="Current progression root degrees" className="font-mono text-sm text-orange-200 break-words">
            {sequence.slice(0, 64).map((degree, i) => `${i > 0 ? i % config.pl === 0 ? ' | ' : ' · ' : ''}${DEGREE_NAMES[degree] ?? degree + 1}`).join('')}
            {sequence.length > 64 && ' …'}
          </p>
          <p className="text-[11px] text-slate-500 mt-1">Roman numerals show root degrees, not chord quality. Mode, Chord and Sch. EXP determine the chord tones.</p>
        </div>
      )}
    </section>
  );
}
