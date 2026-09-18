import { useEffect, useMemo, useState } from 'react';
import { outOfRangeVoices, parseNumberList, rawRoutes, route, unmappedVoices, validateOutput, type OutputClip, type OutputMode, type OutputRoute, type OutputSettings } from './output';

const API = 'http://127.0.0.1:3000';
const modes: { value: OutputMode; label: string }[] = [
  { value: 'notes', label: 'Notes' }, { value: 'trance_arp', label: 'Trance arp' },
  { value: 'contour_arp', label: 'Contour arp' }, { value: 'voice_arp', label: 'Voice arp' },
];
function DraftList({ value, onCommit }: { value: number[]; onCommit: (v: number[]) => void }) {
  const [draft, setDraft] = useState(value.join(', '));
  const commit = () => { const next = parseNumberList(draft); onCommit(next); setDraft(next.join(', ')); };
  return <input value={draft} onChange={e => setDraft(e.target.value)} onBlur={commit} onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur(); }} className="mt-1 w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 text-sm normal-case"/>;
}

function DraftNumber({ value, onCommit, integer = false }: { value: number; onCommit: (v: number) => void; integer?: boolean }) {
  const [draft, setDraft] = useState(String(value));
  const commit = () => { const parsed = Number(draft); const next = Number.isFinite(parsed) ? (integer ? Math.round(parsed) : parsed) : value; onCommit(next); setDraft(String(next)); };
  return <input inputMode="decimal" value={draft} onChange={e => setDraft(e.target.value)} onBlur={commit} onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur(); }} className="mt-1 w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 text-sm text-slate-200"/>;
}

function preset(kind: string, count: number): OutputRoute[] {
  const all = Array.from({ length: count }, (_, i) => i);
  if (kind === 'raw') return rawRoutes(count);
  if (kind === 'all') return [route('All voices', all)];
  if (kind === 'trance') return [{ ...route('Trance arp', all), mode: 'trance_arp', arp_durations: [0.25, 0.25, 0.5] }];
  if (kind === 'contour') return [{ ...route('Contour arp', all), mode: 'contour_arp', voice_pattern: all }];
  if (kind === 'folded') return [{ ...route('Folded voices', all), fold_pitch: 72 }];
  if (kind === 'single') return all.map((v, i) => route(`Part ${i + 1}`, [v], 4, 1 + i));
  if (kind === 'sections') { const split = Math.ceil(count / 2); return [route('Upper section', all.slice(0, split), 4), route('Lower section', all.slice(split), 5)]; }
  if (kind === 'legacy') return [
    { ...route('Mixed humanized', all, 0, 2), humanize_beats: 0.01 },
    { ...route('Trance arp', all, 1, 2), mode: 'trance_arp' },
    { ...route('Contour arp', all, 2, 2), mode: 'contour_arp' },
    { ...route('Folded', all, 3, 2), fold_pitch: 36 },
    ...all.map((v, i) => route(`Voice ${i + 1}`, [v], 4 + i, 2)),
  ];
  return [route('Lead', [0], 4), route('Middle', all.slice(1, -1), 5), route('Bass', [count - 1], 6)];
}

export function OutputPage({ output, onChange, config, onCompose }: { output: OutputSettings; onChange: (v: OutputSettings) => void; config: Record<string, unknown>; onCompose: () => void }) {
  const voiceCount = config?.fugue_mode ? Math.min(8, Math.max(3, Math.round(Number(config.fugue_voice_count) || 3))) : 5;
  const errors = useMemo(() => validateOutput(output), [output]);
  const unmapped = useMemo(() => unmappedVoices(output, voiceCount), [output, voiceCount]);
  const staleVoices = useMemo(() => outOfRangeVoices(output, voiceCount), [output, voiceCount]);
  const [clips, setClips] = useState<OutputClip[]>([]);
  const [previewMsg, setPreviewMsg] = useState('');
  const [previewContext, setPreviewContext] = useState<Record<string, unknown> | null>(null);
  const [trackNames, setTrackNames] = useState<string[]>([]);
  const [connection, setConnection] = useState('Checking Live connection…');
  const [previewing, setPreviewing] = useState(false);

  const refreshConnection = () => {
    setConnection('Checking Live connection…');
    fetch(`${API}/api/output/session`).then(async r => r.ok ? r.json() : Promise.reject(new Error((await r.json().catch(() => null))?.message || `HTTP ${r.status}`))).then(data => {
      const tracks = Array.isArray(data?.tracks) ? data.tracks : [];
      setTrackNames(tracks.map((t: unknown) => typeof t === 'string' ? t : String((t as { name?: unknown })?.name ?? '')));
      const canWrite = data?.script?.capabilities?.write_harmonizer_clips ?? data?.script?.capabilities?.includes?.('write_harmonizer_clips');
      setConnection(canWrite === false ? 'Live bridge connected · update the bridge to enable clip writing' : `Live bridge connected · ${tracks.length} tracks`);
    }).catch((e: unknown) => setConnection(`Live bridge unavailable: ${e instanceof Error ? e.message : String(e)}`));
  };
  useEffect(() => {
    refreshConnection();
  }, []);

  const changeOutput = (next: OutputSettings) => { setClips([]); setPreviewContext(null); setPreviewMsg('Routing changed — preview again to refresh the table.'); onChange(next); };
  const patchRoute = (index: number, patch: Partial<OutputRoute>) => changeOutput({ ...output, routes: output.routes.map((r, i) => i === index ? { ...r, ...patch } : r) });
  const preview = async () => {
    setPreviewing(true); setPreviewMsg('Building preview…'); setClips([]);
    try {
      const res = await fetch(`${API}/api/output/preview`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ output }) });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      if (data?.error) throw new Error(data.error);
      setClips(Array.isArray(data?.clips) ? data.clips : []); setPreviewContext(data?.source ?? data?.context ?? null); setPreviewMsg('Preview ready — nothing was written to Live.');
    } catch (e: unknown) { setPreviewMsg(`Preview unavailable: ${e instanceof Error ? e.message : String(e)}`); }
    setPreviewing(false);
  };
  const sendLast = async () => {
    setPreviewing(true); setPreviewMsg('Sending last render to Live…');
    try {
      const res = await fetch(`${API}/api/output/export`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ output }) });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json().catch(() => ({}));
      const failedClips = Array.isArray(data?.result?.clips) ? data.result.clips.filter((x: { status?: string }) => x.status !== 'written') : [];
      if (data?.status === 'failed') throw new Error(data.message || failedClips.map((x: { message?: string }) => x.message).filter(Boolean).join('; ') || 'Export failed');
      setPreviewMsg(data?.status === 'partial' ? `Partially sent: ${failedClips.map((x: { message?: string }) => x.message || 'a clip failed').join('; ')}` : (data?.message || 'Last render sent to Live.'));
    } catch (e: unknown) { setPreviewMsg(`Could not send: ${e instanceof Error ? e.message : String(e)}`); }
    setPreviewing(false);
  };

  return <main className="flex-1 min-h-0 overflow-y-auto bg-slate-950 p-6">
    <div className="mx-auto max-w-6xl space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div><p className="text-xs uppercase tracking-[0.25em] text-cyan-500">Output routing</p><h1 className="text-2xl font-semibold text-slate-100">Shape clips before they reach Live</h1><p className="mt-1 text-sm text-slate-500">Routes are saved separately from composition snapshots.</p></div>
        <label className="flex items-center gap-3 rounded-xl border border-slate-700 bg-slate-900 px-4 py-3 text-sm"><input type="checkbox" checked={output.enabled} onChange={e => changeOutput({ ...output, enabled: e.target.checked })} className="accent-cyan-500"/><span><b className="block text-slate-200">Write to Live on Generate</b><span className="text-xs text-slate-500">Off saves musical results without changing Live.</span></span></label>
      </div>
      <div className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-900/50 px-3 py-2 text-xs text-slate-400"><span>{connection}</span><button onClick={refreshConnection} className="text-cyan-400 hover:text-cyan-300">Refresh connection</button></div>
      <p className="text-xs text-slate-500">Track and slot numbers match Live and start at 1. Source Voice 1 corresponds to legacy engine voice 0.</p>
      <div className="flex flex-wrap gap-2 rounded-xl border border-slate-800 bg-slate-900/60 p-3">
        <span className="mr-2 self-center text-xs uppercase tracking-wider text-slate-500">Layouts</span>
        {[['raw','Raw separate voices'],['all','All voices / clip'],['mixed','Mixed'],['trance','Trance'],['contour','Contour'],['folded','Folded'],['single','Single parts'],['sections','Sections'],['legacy','Legacy inspired']].map(([key,label]) => <button key={key} onClick={() => changeOutput({ ...output, routes: preset(key, voiceCount) })} className="rounded-md border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs text-slate-300 hover:border-cyan-600 hover:text-cyan-300">{label}</button>)}
        <button onClick={() => changeOutput({ ...output, routes: [...output.routes, route(`Route ${output.routes.length + 1}`, [0], 4 + output.routes.length)] })} className="ml-auto rounded-md bg-cyan-700 px-3 py-1.5 text-xs font-semibold text-white hover:bg-cyan-600">+ Add route</button>
      </div>
      <div className="space-y-3">{output.routes.map((r, i) => <section key={i} className={`rounded-xl border p-4 ${r.enabled ? 'border-slate-700 bg-slate-900/80' : 'border-slate-800 bg-slate-900/30 opacity-70'}`}>
        <div className="grid gap-3 md:grid-cols-[auto_minmax(140px,1.4fr)_1fr_100px_100px_auto] md:items-center">
          <input aria-label={`Enable ${r.name}`} type="checkbox" checked={r.enabled} onChange={e => patchRoute(i,{enabled:e.target.checked})} className="accent-cyan-500"/>
          <input value={r.name} onChange={e => patchRoute(i,{name:e.target.value})} className="rounded border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm" placeholder="Clip name"/>
          <select value={r.mode} onChange={e => patchRoute(i,{mode:e.target.value as OutputMode})} className="rounded border border-slate-700 bg-slate-950 px-2 py-1.5 text-sm">{modes.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}</select>
          <label className="text-[10px] uppercase text-slate-500">Track<DraftNumber key={`track-${r.track_index}`} integer value={r.track_index + 1} onCommit={v => patchRoute(i,{track_index:Math.max(0,v-1)})}/></label>
          <label className="text-[10px] uppercase text-slate-500">Slot<DraftNumber key={`clip-${r.clip_index}`} integer value={r.clip_index + 1} onCommit={v => patchRoute(i,{clip_index:Math.max(0,v-1)})}/></label>
          <div className="flex gap-1"><button title="Duplicate route" onClick={() => changeOutput({...output,routes:[...output.routes.slice(0,i+1),{...r,name:`${r.name} copy`},...output.routes.slice(i+1)]})} className="rounded px-2 py-1 text-slate-400 hover:bg-slate-800">⧉</button><button title="Remove route" onClick={() => changeOutput({...output,routes:output.routes.filter((_,x)=>x!==i)})} className="rounded px-2 py-1 text-rose-400 hover:bg-rose-950">×</button></div>
        </div>
        {trackNames[r.track_index] && <p className="mt-1 pl-7 text-xs text-cyan-600">Live track: {trackNames[r.track_index]}</p>}
        <div className="mt-3 flex flex-wrap items-center gap-2"><span className="text-[10px] uppercase tracking-wider text-slate-500">Source voices</span>{Array.from({length:voiceCount},(_,v)=><button key={v} onClick={()=>patchRoute(i,{voices:r.voices.includes(v)?r.voices.filter(x=>x!==v):[...r.voices,v].sort((a,b)=>a-b)})} className={`rounded-full border px-2.5 py-1 text-xs ${r.voices.includes(v)?'border-cyan-500 bg-cyan-950 text-cyan-300':'border-slate-700 text-slate-500'}`}>Voice {v+1}</button>)}</div>
        <details className="mt-3 border-t border-slate-800 pt-2"><summary className="cursor-pointer text-xs uppercase tracking-wider text-slate-500">Advanced</summary><div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-5">
          {([['Transpose semitones','transpose',r.transpose],['Time scale ×','time_scale',r.time_scale],['Humanize beats','humanize_beats',r.humanize_beats]] as const).map(([label,key,val])=><label key={`${key}-${val}`} className="text-[10px] uppercase text-slate-500">{label}<DraftNumber value={val} onCommit={v=>patchRoute(i,{[key]:v})}/></label>)}
          {r.mode !== 'notes' && <><label className="text-[10px] uppercase text-slate-500">Initial MIDI pitch<DraftNumber integer value={r.initial_pitch} onCommit={v=>patchRoute(i,{initial_pitch:v})}/></label><label className="text-[10px] uppercase text-slate-500">Lead octave<DraftNumber integer value={r.lead_octave} onCommit={v=>patchRoute(i,{lead_octave:v})}/></label></>}
          <label className="text-[10px] uppercase text-slate-500">Fold register MIDI<input type="number" placeholder="Off" value={r.fold_pitch ?? ''} onChange={e=>patchRoute(i,{fold_pitch:e.target.value===''?null:Math.round(Number(e.target.value))})} className="mt-1 w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 text-sm text-slate-200"/></label>
          {r.mode !== 'notes' && <><label className="text-[10px] uppercase text-slate-500">Lead voice<DraftNumber integer value={r.lead_voice+1} onCommit={v=>patchRoute(i,{lead_voice:v-1})}/></label><label className="text-[10px] uppercase text-slate-500">Bass voice<DraftNumber integer value={r.bass_voice+1} onCommit={v=>patchRoute(i,{bass_voice:v-1})}/></label></>}
          {r.mode !== 'notes' && <label className="text-[10px] uppercase text-slate-500">Arp durations<DraftList key={r.arp_durations.join(',')} value={r.arp_durations} onCommit={v=>patchRoute(i,{arp_durations:v})}/></label>}{r.mode === 'voice_arp' && <label className="text-[10px] uppercase text-slate-500">Voice pattern<DraftList key={r.voice_pattern.join(',')} value={r.voice_pattern.map(v=>v+1)} onCommit={v=>patchRoute(i,{voice_pattern:v.map(x=>x-1)})}/></label>}
          <label className="col-span-2 flex items-center gap-2 text-xs text-slate-400"><input type="checkbox" checked={r.merge_adjacent} onChange={e=>patchRoute(i,{merge_adjacent:e.target.checked})} className="accent-cyan-500"/> Merge adjacent equal notes</label>
        </div></details>
      </section>)}</div>
      <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4"><div className="flex flex-wrap items-center justify-between gap-3"><div><h2 className="font-semibold text-slate-200">Clip preview</h2><p className="text-xs text-slate-500">Resolve clips and lengths without writing to Live. Generate once before previewing or sending.</p></div><div className="flex gap-2"><button disabled={previewing || errors.length>0} onClick={preview} className="rounded-lg bg-emerald-700 px-4 py-2 text-sm font-semibold disabled:opacity-40">{previewing?'Working…':'Preview clips'}</button><button disabled={previewing || errors.length>0} onClick={sendLast} className="rounded-lg border border-cyan-600 bg-cyan-950 px-4 py-2 text-sm font-semibold text-cyan-300 disabled:opacity-40">Send last render to Live</button></div></div>
        {errors.length>0 && <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-rose-400">{errors.map((e,i)=><li key={i}>{e}</li>)}</ul>}{unmapped.length>0 && <p className="mt-3 text-xs text-amber-400">Unmapped in current Compose settings: {unmapped.map(v=>`Voice ${v+1}`).join(', ')}. Partial arrangements are allowed.</p>}{staleVoices.length>0 && <p className="mt-2 text-xs text-amber-400">Saved routes also reference {staleVoices.map(v=>`Voice ${v+1}`).join(', ')}, beyond the current Compose voice count. Preview still uses the last render.</p>}{previewMsg && <p className="mt-3 text-xs text-slate-400">{previewMsg}</p>}
        {previewContext && <p className="mt-2 text-xs text-slate-500">Last render: {String(previewContext.voice_count ?? '—')} voices · {String(previewContext.length ?? previewContext.clip_length ?? '—')} beats</p>}
        {clips.length>0 && <div className="mt-3 overflow-x-auto"><table className="w-full text-left text-xs"><thead className="text-slate-500"><tr><th className="py-2">Target</th><th>Name</th><th>Length</th><th>Notes</th></tr></thead><tbody>{clips.map((c,i)=><tr key={i} className="border-t border-slate-800"><td className="py-2">Track {c.track_index+1} / Slot {c.clip_index+1}{trackNames[c.track_index]?` · ${trackNames[c.track_index]}`:''}</td><td>{c.name}</td><td>{c.length} beats</td><td>{c.note_count}</td></tr>)}</tbody></table></div>}
      </section>
      <button onClick={onCompose} className="text-sm text-cyan-400 hover:text-cyan-300">← Back to Compose</button>
    </div>
  </main>;
}
