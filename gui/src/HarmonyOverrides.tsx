import { useEffect, useState } from 'react';
import { CHORD_LIST, CHORD_NAMES, DEFAULT_MODE_SCALES, MODE_LABELS, getChordStructures, getModeScales, isCustomRow, parseModeScale } from './harmony';

export function HarmonyOverrides({ kind, value, onChange, onClose }: {
  kind: 'mode' | 'chord';
  value: unknown;
  onChange: (rows: number[][] | null) => void;
  onClose: () => void;
}) {
  const isMode = kind === 'mode';
  const title = isMode ? 'Mode Overrides' : 'Chord Overrides';
  const defaults = isMode ? DEFAULT_MODE_SCALES : CHORD_LIST;
  const rows = isMode ? getModeScales(value) : getChordStructures(value);
  const labels = isMode ? MODE_LABELS : CHORD_NAMES;
  const [drafts, setDrafts] = useState<Record<number, string>>({});
  const invalidDraft = isMode && Object.values(drafts).some(draft => parseModeScale(draft) === null);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape' && !invalidDraft) onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose, invalidDraft]);

  const updateRow = (index: number, row: number[]) => onChange(rows.map((current, i) => i === index ? row : current));
  const resetRow = (index: number) => {
    setDrafts(current => {
      const next = { ...current };
      delete next[index];
      return next;
    });
    updateRow(index, [...defaults[index]]);
  };

  return (
    <div className="fixed inset-0 z-[100] bg-black/70 flex items-center justify-center p-6"
      onMouseDown={event => { if (event.target === event.currentTarget && !invalidDraft) onClose(); }}>
      <div role="dialog" aria-modal="true" aria-labelledby="harmony-overrides-title"
        className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl max-w-[95vw] max-h-[90vh] overflow-auto">
        <div className="flex flex-wrap items-center justify-between gap-4 px-5 py-3 border-b border-slate-800 sticky top-0 bg-slate-900 z-10">
          <h2 id="harmony-overrides-title" className={`text-lg font-bold ${isMode ? 'text-yellow-300' : 'text-green-300'}`}>{title}</h2>
          <div className="flex gap-2">
            <button onClick={() => { setDrafts({}); onChange(null); }} className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-200 text-sm font-bold rounded-lg">Reset to default</button>
            <button autoFocus onClick={onClose} disabled={invalidDraft} className="px-3 py-1.5 bg-cyan-700 hover:bg-cyan-600 text-slate-100 text-sm font-bold rounded-lg disabled:opacity-40">Done</button>
          </div>
        </div>
        <div className="p-5 space-y-4">
          <p className="text-sm text-slate-400 max-w-3xl">
            {isMode
              ? 'Enter comma-separated semitone offsets above Root, for example 0,2,4,7,9 for a pentatonic scale. Each slot accepts 1–12 ascending notes from 0 to 11, starting at 0.'
              : 'Choose the members available in each Chord contour slot. Expansion multiplies these indices: at 2 (thirds), members 0–6 are root, 3rd, 5th, 7th, 9th, 11th and 13th. The root stays included.'}
          </p>
          <table className="border-collapse w-full">
            <thead><tr>
              <th className="px-2 py-2 text-left text-xs uppercase tracking-wider text-slate-500">Contour slot</th>
              {isMode ? <th className="px-2 py-2 text-left text-xs text-slate-400">Scale offsets</th>
                : Array.from({ length: 7 }, (_, i) => <th key={i} className="px-2 py-2 text-xs font-mono text-slate-400">Member {i}</th>)}
              <th className="px-2 py-2 text-xs text-slate-500">Reset</th>
            </tr></thead>
            <tbody>{rows.map((row, ri) => {
              const custom = isCustomRow(row, defaults[ri]);
              const draft = drafts[ri] ?? row.join(',');
              const invalid = isMode && parseModeScale(draft) === null;
              return <tr key={ri} className="hover:bg-slate-800/40">
                <th scope="row" className="px-2 py-2 text-left text-sm font-normal text-slate-300 whitespace-nowrap">
                  <span className="text-slate-600 font-mono mr-2">{ri}</span>{labels[ri]}{custom && <span className="ml-2 text-cyan-300 text-xs">Custom</span>}
                </th>
                {isMode ? <td className="px-2 py-1">
                  <div className="flex items-center gap-3">
                    <input aria-label={`${labels[ri]} scale offsets`} aria-invalid={invalid}
                      aria-describedby={invalid ? `mode-error-${ri}` : undefined} type="text" value={draft}
                      onChange={event => {
                        const text = event.target.value;
                        setDrafts(current => ({ ...current, [ri]: text }));
                        const scale = parseModeScale(text);
                        if (scale) updateRow(ri, scale);
                      }}
                      className={`w-full min-w-64 bg-slate-950 border rounded px-3 py-2 text-sm font-mono text-yellow-200 ${invalid ? 'border-rose-500' : 'border-slate-700'}`} />
                    <span className="text-xs text-slate-500 whitespace-nowrap">{row.length} notes</span>
                  </div>
                  {invalid && <p id={`mode-error-${ri}`} role="alert" className="mt-1 text-xs text-rose-300">Not applied. Use ascending whole numbers 0–11, starting at 0, separated by commas.</p>}
                </td> : Array.from({ length: 7 }, (_, ci) => <td key={ci} className="px-1 py-1 text-center">
                    <button aria-label={`${labels[ri]} member ${ci}`} aria-pressed={row.includes(ci)} disabled={ci === 0}
                      onClick={() => updateRow(ri, row.includes(ci) ? row.filter(n => n !== ci) : [...row, ci].sort((a, b) => a - b))}
                      className={`w-14 py-2 rounded border text-sm font-mono ${row.includes(ci) ? 'border-green-700 bg-green-950 text-green-300' : 'border-slate-800 bg-slate-950 text-slate-600'} disabled:opacity-50`}>
                      {row.includes(ci) ? '✓' : '—'}
                    </button>
                </td>)}
                <td className="px-2 py-1"><button aria-label={`Reset ${labels[ri]}`} disabled={!custom && !invalid} onClick={() => resetRow(ri)}
                  className="text-xs text-slate-400 hover:text-slate-200 disabled:opacity-30">Reset</button></td>
              </tr>;
            })}</tbody>
          </table>
          <p className="text-xs text-slate-500 max-w-3xl">
            Changes apply immediately to all voices using that slot and are saved with snapshots on Generate. Custom slots have * in the contour labels.
            {isMode && ' Chord expansion and progression degrees wrap around the number of notes you enter. Edited scales play as written; the default Aeolian scale still raises its seventh on dominant chords. Progression generators keep each slot’s usual degree transitions.'}
            {!isMode && ' These are the note pools for Schillinger generation; the Chords whitelist separately restricts the final voicing.'}
          </p>
        </div>
      </div>
    </div>
  );
}
