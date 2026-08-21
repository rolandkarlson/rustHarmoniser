import { useState, useEffect, useRef, useMemo, type ReactNode } from 'react'
import { ContourEditor } from './ContourEditor'
import { saveSnapshot, listSnapshots, loadSnapshot, deleteSnapshot, renameSnapshot, toggleFavorite, type Snapshot } from './snapshots'
import CodeMirror, { keymap, Prec, EditorView, type Extension } from '@uiw/react-codemirror'
import { javascript } from '@codemirror/lang-javascript'
import { oneDark } from '@codemirror/theme-one-dark'
import './index.css'

// Shared CodeMirror-based script editor. Used both docked at the bottom and
// blown up to a fullscreen modal. `onRun` fires on ⌘/Ctrl+Enter.
function ScriptEditor({
  value,
  onChange,
  onRun,
  height,
  autoFocus,
}: {
  value: string;
  onChange: (v: string) => void;
  onRun: () => void;
  height: string;
  autoFocus?: boolean;
}) {
  // Keep the latest onRun in a ref so the keymap closure never goes stale
  // (extensions are memoised once and shouldn't be rebuilt on every keystroke).
  const runRef = useRef(onRun);
  runRef.current = onRun;
  const extensions = useMemo<Extension[]>(() => [
    javascript(),
    EditorView.lineWrapping,
    Prec.highest(keymap.of([
      { key: 'Mod-Enter', preventDefault: true, run: () => { runRef.current(); return true; } },
    ])),
  ], []);
  return (
    <CodeMirror
      value={value}
      onChange={onChange}
      height={height}
      theme={oneDark}
      extensions={extensions}
      autoFocus={autoFocus}
      basicSetup={{
        lineNumbers: true,
        highlightActiveLine: true,
        bracketMatching: true,
        closeBrackets: true,
        autocompletion: true,
        foldGutter: false,
        highlightActiveLineGutter: true,
      }}
      className="h-full text-sm rounded-lg overflow-hidden border border-slate-800 focus-within:border-cyan-500/50"
    />
  );
}

const MODE_NAMES = ['Ion', 'Dor', 'Phr', 'Lyd', 'Mix', 'Aeo', 'Loc'];

function generateModeFromSteps(mode: number): number[] {
  const stepPattern = [2, 2, 1, 2, 2, 2, 1];
  const m = ((mode % 7) + 7) % 7;
  const rotated = [...stepPattern.slice(m), ...stepPattern.slice(0, m)];
  rotated.pop();
  const notes = [0];
  let current = 0;
  for (const step of rotated) { current = (current + step) % 12; notes.push(current); }
  return notes.sort((a, b) => a - b);
}

function sameChord(a: number[], b: number[]): boolean {
  const norm = (arr: number[]) => [...new Set(arr.map(n => ((n % 12) + 12) % 12))].sort((x, y) => x - y);
  const sa = norm(a), sb = norm(b);
  return sa.length === sb.length && sa.every((v, i) => v === sb[i]);
}

const CHORD_LIST = [
  [0,1,2], [0,1,2,4,5], [0,1,2,4,5],
  [0,1,2,3,4], [0,1,2,3,4,5], [0,1,2,3,4,5,6]
];

// Harmony scoring matrix — 9 style rows × 12 interval columns.
// Mirrors the built-in default in src/harmonizer.rs (HARMONY_MATRIX).
const HARMONY_MATRIX_ROWS = [
  'Classical', 'Jazz', 'Tension', 'Ethereal', 'Dark', 'Bright', 'Aggressive', 'Ancient', 'Neutral'
];
const HARMONY_MATRIX_COLS = ['P1', 'm2', 'M2', 'm3', 'M3', 'P4', 'TT', 'P5', 'm6', 'M6', 'm7', 'M7'];
const DEFAULT_HARMONY_MATRIX: number[][] = [
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

// Seed pitches for the 5 generated voices (high → low; voice 0 = leading).
// Semitone offsets from the chord root, for the chord-structure builder.
const CHORD_DEGREE_LABELS = ['R', '\u266d2', '2', '\u266d3', '3', '4', '\u266d5', '5', '\u266d6', '6', '\u266d7', '7'];

// Named chord structures offered as one-click presets. Each is a set of
// pitch-class offsets from an arbitrary root — the backend expands every entry
// over all 12 roots, so these are chord TYPES, not specific chords.
const CHORD_PRESETS: { name: string; pcs: number[] }[] = [
  { name: 'Major', pcs: [0, 4, 7] },
  { name: 'Minor', pcs: [0, 3, 7] },
  { name: 'Diminished', pcs: [0, 3, 6] },
  { name: 'Augmented', pcs: [0, 4, 8] },
  { name: 'Sus2', pcs: [0, 2, 7] },
  { name: 'Sus4', pcs: [0, 5, 7] },
  { name: 'Major 7th', pcs: [0, 4, 7, 11] },
  { name: 'Dominant 7th', pcs: [0, 4, 7, 10] },
  { name: 'Minor 7th', pcs: [0, 3, 7, 10] },
  { name: 'Half-dim 7th', pcs: [0, 3, 6, 10] },
  { name: 'Dim 7th', pcs: [0, 3, 6, 9] },
  { name: 'Major 6th', pcs: [0, 4, 7, 9] },
  { name: 'Minor 6th', pcs: [0, 3, 7, 9] },
];

// Distinct pitch classes, in ascending order.
const normalizePcs = (pcs: number[]): number[] =>
  [...new Set(pcs.map((p) => ((Math.round(p) % 12) + 12) % 12))].sort((a, b) => a - b);

// Identity of a chord structure UP TO TRANSPOSITION, matching the backend, which
// expands each template over all 12 roots. So [0,4,7] and [0,3,8] are one and the
// same rule and must not both end up in the list.
const rotationKey = (pcs: number[]): string => {
  const set = normalizePcs(pcs);
  if (set.length === 0) return '';
  let best = '';
  for (let k = 0; k < 12; k++) {
    const rot = set.map((p) => (p + k) % 12).sort((a, b) => a - b).join(',');
    if (k === 0 || rot < best) best = rot;
  }
  return best;
};

const chordPresetFor = (pcs: number[]) => {
  const key = rotationKey(pcs);
  return key ? CHORD_PRESETS.find((p) => rotationKey(p.pcs) === key) : undefined;
};

// Store a rotation of a known chord in that chord's standard spelling, so the list
// reads musically: a set entered as [0,3,8] lands as the major triad it is.
const canonicalPcs = (pcs: number[]): number[] => {
  const preset = chordPresetFor(pcs);
  return preset ? [...preset.pcs] : normalizePcs(pcs);
};

const chordName = (pcs: number[]): string => chordPresetFor(pcs)?.name ?? 'Custom';

// A chord_templates entry, matching model::ChordTemplate: a bare offset list means
// "no stated preference", the object form carries a usage weight.
type ChordTemplate = number[] | { pcs: number[]; weight: number };

const tplPcs = (t: ChordTemplate): number[] => (Array.isArray(t) ? t : t.pcs);
const tplWeight = (t: ChordTemplate): number => (Array.isArray(t) ? 1 : t.weight);
// Weights only take effect once at least one entry states one — matching the
// backend, where a list of bare entries leaves every group free to pick any
// listed structure instead of being locked to an apportioned one.
const weightsActive = (list: ChordTemplate[]): boolean => list.some((t) => !Array.isArray(t));

// Mirrors DEFAULT_START_NOTES in src/model.rs.
const DEFAULT_START_NOTES = [70, 65, 60, 50, 34];

// MIDI number → note name (e.g. 60 → "C3"). Middle C (60) = C3 here, matching
// Ableton's octave numbering.
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
function midiName(n: number): string {
  if (!Number.isFinite(n)) return '—';
  const p = ((Math.round(n) % 12) + 12) % 12;
  const oct = Math.floor(Math.round(n) / 12) - 2;
  return `${NOTE_NAMES[p]}${oct}`;
}

// Text input for numeric config values. Keeps the raw typed string while editing
// (so intermediate states like "-", "." and "1." are allowed) and only commits a
// finite parsed number — never writes NaN. Syncs to the external value on blur.
function NumberField({ value, onChange, integer, className, title, disabled }: {
  value: number;
  onChange: (v: number) => void;
  integer?: boolean;
  className?: string;
  title?: string;
  disabled?: boolean;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const display = draft !== null ? draft : String(value ?? '');
  return (
    <input
      type="text"
      value={display}
      title={title}
      disabled={disabled}
      className={className}
      onChange={(e) => {
        const s = e.target.value;
        setDraft(s);
        const v = integer ? parseInt(s, 10) : parseFloat(s);
        if (Number.isFinite(v)) onChange(v);
      }}
      onBlur={() => setDraft(null)}
    />
  );
}

// A labelled cluster of related header parameters.
function ParamGroup({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1 bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800">
      <span className="text-[10px] uppercase tracking-widest text-slate-600 select-none">{label}</span>
      <div className="flex gap-3 items-center flex-wrap">{children}</div>
    </div>
  );
}

// Contour tabs that only feed the Schillinger candidate generator — hidden in
// Chromatic mode, where the progression layer is bypassed entirely.
const SCHILLINGER_TAB_IDS = ['schillinger', 'schillinger_ex', 'mode', 'chord'];

// ----- "Why this chord" inspector -----
// Mirrors core/src/trace.rs: the named score contributions the backend returns
// per chosen chord (in the /api/generate response and the render archive).
type ScoreTerm = { name: string; value: number };
type VoiceBd = {
  channel: number;
  pitch: number;
  previous_pitch: number | null;
  is_leader: boolean;
  terms: ScoreTerm[];
};
type GroupBd = {
  start: number;
  bar: number;
  root_pc: number | null;
  score: number;
  soft_score: number;
  hard_violation_count: number;
  hard_violations: { name: string; count: number }[];
  chord_terms: ScoreTerm[];
  voices: VoiceBd[];
  scored: boolean;
};

// Labels + tooltips per term name (names come from harmonizer's ScoreSink calls).
const TERM_INFO: Record<string, { label: string; tip: string }> = {
  // chord-level
  harmony_mean: { label: 'Harmony · mean', tip: 'Average pairwise consonance over the sonority (H-Matrix style blended with roughness), × the group harmony weight.' },
  harmony_worst: { label: 'Harmony · worst pair', tip: 'The single most disliked voice pair — one bad clash drags the whole chord.' },
  harmony_bass: { label: 'Harmony · vs bass', tip: 'Consonance of the pairs involving the bass note (inversion colour).' },
  harmony_quality: { label: 'Harmony · quality', tip: 'Pitch classes measured from the CHORD ROOT — the only harmony part that can tell major from minor (chord_quality_weight).' },
  root_position: { label: 'Root position', tip: 'Bass-on-root preference: full bonus in root position, 0.4 with the third in the bass, 0 for six-four, −0.3 for a non-chord tone (root_position_weight).' },
  root_doubling: { label: 'Doubling', tip: '+ for doubling the root, − per extra copy of the leading tone or chordal 7th (root_doubling_weight).' },
  interval_variety: { label: 'Interval variety', tip: 'Penalty per repeated interval class within the sonority; octave doublings count at any chord size (interval_exists_in_harmony).' },
  parallel_motion: { label: 'Parallel 5th/8ve', tip: 'Parallel fifths/octaves or antiparallel octaves against another moving voice (consecutive_octav_fift).' },
  same_direction: { label: 'Same direction', tip: 'An outer voice moving with the chord majority instead of against it (same_direction).' },
  common_tone_penalty: { label: 'Common tones', tip: 'Group-level penalty per voice holding its previous pitch (common_tone_penalty).' },
  // per-voice
  smoothness: { label: 'Smoothness', tip: 'Melodic distance from the previous pitch — hold is smoothest, leaps decay — × the group smoothness weight. Bass gets P4/P5/octave leaps floored.' },
  melody_force: { label: 'Melody force', tip: 'Recency-decayed penalty for pitches from the voice\'s last 5 notes, small reward for stepwise motion (melody_force).' },
  tendency: { label: 'Tendency tone', tip: 'Leading tone resolving up to the tonic / chordal 7th resolving down by step (tendency_weight).' },
  off_scale_hold: { label: 'Off-scale hold', tip: 'Fixed penalty for holding a pitch that has left the current scale — only survives when the voice-change budget forces it.' },
  contour_spring: { label: 'Contour spring', tip: 'Quadratic pull toward the voice contour\'s target pitch (voice_contour_weight).' },
  crossing_penalty: { label: 'Crossing', tip: 'Candidate too close to a neighbouring voice\'s register (no_crossing, applied per violated side).' },
  leader_history: { label: 'Leader history', tip: 'This chord\'s LEADER takes the repeat penalties (last_note_same, last_note_exist_in_voice) so it keeps moving.' },
  hold_stickiness: { label: 'Hold stickiness', tip: 'Non-leader bonus for keeping a common tone (same_note_bonus). Stagnant voices lose it.' },
  // hard violations
  unison_collision: { label: 'Unison collision', tip: 'Two voices on exactly the same pitch.' },
  forbidden_interval: { label: 'Forbidden interval', tip: 'An interval class the active H-Matrix row hard-forbids (cell ≤ −5).' },
  chord_template: { label: 'Chord whitelist', tip: 'The sonority\'s pitch-class set matches none of the allowed chord structures.' },
  voice_budget: { label: 'Voice budget', tip: 'Outside the min/max voices-changed budget.' },
};

const termLabel = (name: string) => TERM_INFO[name]?.label ?? name;
const termTip = (name: string) => TERM_INFO[name]?.tip ?? name;

function TermRow({ term, maxAbs }: { term: ScoreTerm; maxAbs: number }) {
  const pct = maxAbs > 0 ? Math.min(100, (Math.abs(term.value) / maxAbs) * 100) : 0;
  const pos = term.value >= 0;
  return (
    <div className="flex items-center gap-2 py-0.5" title={termTip(term.name)}>
      <span className="w-36 shrink-0 text-xs text-slate-400 truncate">{termLabel(term.name)}</span>
      <div className="flex-1 h-2 rounded bg-slate-950 overflow-hidden flex">
        <div className="w-1/2 flex justify-end">
          {!pos && <div className="h-full bg-rose-500/70 rounded-l" style={{ width: `${pct}%` }} />}
        </div>
        <div className="w-1/2 border-l border-slate-700">
          {pos && <div className="h-full bg-emerald-500/70 rounded-r" style={{ width: `${pct}%` }} />}
        </div>
      </div>
      <span className={`w-16 shrink-0 text-right text-xs font-mono ${pos ? 'text-emerald-300' : 'text-rose-300'}`}>
        {term.value >= 0 ? '+' : ''}{term.value.toFixed(2)}
      </span>
    </div>
  );
}

function ChordInspector({ breakdown, onClose }: { breakdown: GroupBd[]; onClose: () => void }) {
  const [sel, setSel] = useState(0);
  const g = breakdown[Math.min(sel, breakdown.length - 1)];

  // ←/→ step through chords; Escape closes.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { onClose(); return; }
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { e.preventDefault(); setSel(s => Math.min(breakdown.length - 1, s + 1)); }
      if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { e.preventDefault(); setSel(s => Math.max(0, s - 1)); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [breakdown.length, onClose]);

  if (!g) return null;

  const fmtPos = (grp: GroupBd) => {
    const beat = grp.start - grp.bar * 4;
    const beatStr = Number.isInteger(beat) ? String(beat + 1) : (beat + 1).toFixed(2);
    return `${grp.bar + 1}·${beatStr}`;
  };
  const chordNotes = (grp: GroupBd) =>
    grp.voices.slice().sort((a, b) => b.pitch - a.pitch).map(v => midiName(v.pitch)).join(' ');

  // One scale for every bar in this chord's detail view, so terms compare
  // visually across the chord and its voices.
  const maxAbs = Math.max(
    0.01,
    ...g.chord_terms.map(t => Math.abs(t.value)),
    ...g.voices.flatMap(v => v.terms.map(t => Math.abs(t.value))),
  );
  const byMagnitude = (a: ScoreTerm, b: ScoreTerm) => Math.abs(b.value) - Math.abs(a.value);

  return (
    <div
      className="fixed inset-0 z-[100] bg-black/70 flex items-center justify-center p-6"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-[95vw] h-[90vh] flex flex-col overflow-hidden">
        <div className="flex items-center justify-between gap-4 px-5 py-3 border-b border-slate-800 shrink-0">
          <div>
            <h2 className="text-lg font-bold text-emerald-300">Why This Chord</h2>
            <p className="text-xs text-slate-500">
              Named score contributions per chosen chord, from the last render. These are the exact numbers the beam search accumulated — not an approximation. Navigate with ←/→.
            </p>
          </div>
          <button
            onClick={onClose}
            className="px-3 py-1.5 bg-emerald-700 hover:bg-emerald-600 text-slate-100 text-sm font-bold rounded-lg transition-all active:scale-95"
          >
            Done
          </button>
        </div>

        <div className="flex flex-1 min-h-0">
          {/* Chord timeline */}
          <div className="w-72 shrink-0 border-r border-slate-800 overflow-y-auto">
            {breakdown.map((grp, i) => (
              <button
                key={i}
                onClick={() => setSel(i)}
                className={`w-full flex items-center gap-2 px-3 py-1.5 text-left border-b border-slate-800/60 transition-colors ${i === sel ? 'bg-slate-800 text-cyan-300' : 'text-slate-400 hover:bg-slate-800/50'}`}
              >
                <span className="w-10 shrink-0 text-xs font-mono text-slate-500">{fmtPos(grp)}</span>
                <span className="flex-1 text-xs font-mono truncate">{chordNotes(grp)}</span>
                {grp.hard_violation_count > 0 && (
                  <span className="shrink-0 px-1 rounded bg-rose-900 text-rose-200 text-[10px] font-mono" title={`${grp.hard_violation_count} hard violation(s)`}>
                    !{grp.hard_violation_count}
                  </span>
                )}
                <span className={`w-12 shrink-0 text-right text-xs font-mono ${grp.score >= 0 ? 'text-emerald-400/80' : 'text-rose-400/80'}`}>
                  {grp.score.toFixed(1)}
                </span>
              </button>
            ))}
          </div>

          {/* Detail */}
          <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-4">
            <div className="flex items-baseline gap-4 flex-wrap">
              <span className="text-xl font-bold text-slate-100 font-mono">{chordNotes(g)}</span>
              <span className="text-sm text-slate-400">bar {g.bar + 1}, beat {fmtPos(g).split('·')[1]}</span>
              {g.root_pc !== null && (
                <span className="text-sm text-slate-400">root <span className="text-cyan-300 font-mono">{NOTE_NAMES[((g.root_pc % 12) + 12) % 12]}</span></span>
              )}
              <span className="text-sm text-slate-400">
                score <span className={`font-mono ${g.score >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>{g.score.toFixed(3)}</span>
              </span>
              {g.hard_violation_count > 0 && (
                <span className="text-sm text-rose-300">soft {g.soft_score.toFixed(3)} − {g.hard_violation_count} × 1000</span>
              )}
            </div>

            {!g.scored && (
              <div className="text-sm text-amber-300 bg-amber-950/40 border border-amber-800 rounded-lg px-3 py-2">
                This group passed through unscored — no candidates were generated for it, so there is no breakdown.
              </div>
            )}

            {g.hard_violations.length > 0 && (
              <div className="flex gap-2 flex-wrap">
                {g.hard_violations.map(h => (
                  <span key={h.name} className="px-2 py-1 rounded bg-rose-950 border border-rose-800 text-rose-200 text-xs font-mono" title={termTip(h.name)}>
                    {termLabel(h.name)} ×{h.count}
                  </span>
                ))}
              </div>
            )}

            {g.chord_terms.length > 0 && (
              <div className="bg-slate-950/60 border border-slate-800 rounded-lg p-3">
                <div className="text-xs uppercase tracking-wider text-slate-500 mb-2">Chord terms</div>
                {g.chord_terms.slice().sort(byMagnitude).map(t => (
                  <TermRow key={t.name} term={t} maxAbs={maxAbs} />
                ))}
              </div>
            )}

            <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
              {g.voices.map((v, i) => (
                <div key={i} className={`bg-slate-950/60 border rounded-lg p-3 ${v.is_leader ? 'border-amber-600/60' : 'border-slate-800'}`}>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">V{v.channel}</span>
                    <span className="text-sm font-mono text-slate-200">
                      {v.previous_pitch !== null && v.previous_pitch !== undefined ? `${midiName(v.previous_pitch)} → ` : ''}{midiName(v.pitch)}
                    </span>
                    {v.is_leader && (
                      <span className="ml-auto px-1.5 py-0.5 rounded bg-amber-900/70 text-amber-200 text-[10px] uppercase tracking-wider" title="The leader carries the melodic motion this chord: it takes the repeat penalties instead of the hold bonus, so it stays free to move.">
                        Leader
                      </span>
                    )}
                  </div>
                  {v.terms.length > 0
                    ? v.terms.slice().sort(byMagnitude).map(t => <TermRow key={t.name} term={t} maxAbs={maxAbs} />)
                    : <div className="text-xs text-slate-600">no terms (fixed lead)</div>}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [config, setConfig] = useState<any>(null);
  const [activeTab, setActiveTab] = useState('harmony');
  const [selectedVoice, setSelectedVoice] = useState(0);
  const [isGenerating, setIsGenerating] = useState(false);
  const [message, setMessage] = useState('');
  const [snapshots, setSnapshots] = useState<Snapshot[]>([]);
  const [showSnapshots, setShowSnapshots] = useState(false);
  const [showMatrix, setShowMatrix] = useState(false);
  const [showStartNotes, setShowStartNotes] = useState(false);
  const [showChords, setShowChords] = useState(false);
  // Pitch classes currently ticked in the modal's custom-chord builder.
  const [chordDraft, setChordDraft] = useState<number[]>([0, 4, 7]);
  const [snTrack, setSnTrack] = useState(0);
  const [snClip, setSnClip] = useState(0);
  const [fetchingChord, setFetchingChord] = useState(false);
  const [chordMsg, setChordMsg] = useState<{ text: string; error: boolean } | null>(null);
  const [favoritesOnly, setFavoritesOnly] = useState(false);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const [broadcastVoices, setBroadcastVoices] = useState(false);
  const [showConsole, setShowConsole] = useState(false);
  const [consoleFullscreen, setConsoleFullscreen] = useState(false);
  const [script, setScript] = useState<string>(() => localStorage.getItem('contourScript') || '');
  const [scriptMsg, setScriptMsg] = useState<{ text: string; error: boolean } | null>(null);
  // Score breakdown of the LAST render (from the generate response); powers the
  // "Why?" inspector. Cleared implicitly by never persisting across reloads.
  const [breakdown, setBreakdown] = useState<GroupBd[]>([]);
  const [showInspector, setShowInspector] = useState(false);
  // Collapsible row holding the preset/randomise buttons; folded by default.
  const [showGenerators, setShowGenerators] = useState<boolean>(
    () => localStorage.getItem('showGenerators') === '1',
  );
  const toggleGenerators = () => setShowGenerators(s => {
    localStorage.setItem('showGenerators', s ? '0' : '1');
    return !s;
  });
  const snapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const snaps = listSnapshots();
    setSnapshots(snaps);
    if (snaps.length > 0) {
      // Restore the most recent snapshot (newest is first) on reload.
      setConfig(snaps[0].config);
      setMessage(`Restored snapshot ${snaps[0].name}`);
      return;
    }
    fetch('http://127.0.0.1:3000/api/config')
      .then(res => res.json())
      .then(data => setConfig(data))
      .catch(err => setMessage(`Error loading config: ${err}`));
  }, []);

  // Expose current config to the browser console for debugging.
  // In DevTools: `config` (live snapshot), `setConfig(obj)` to overwrite, `patchConfig({key: val})` to merge.
  useEffect(() => {
    const w = window as any;
    w.config = config;
    w.setConfig = setConfig;
    w.patchConfig = (patch: any) => setConfig((c: any) => ({ ...c, ...patch }));
  }, [config]);

  // Chromatic mode hides the Schillinger-only tabs; bounce off one if active.
  useEffect(() => {
    if (config && !(config.schillinger_progression ?? true) && SCHILLINGER_TAB_IDS.includes(activeTab)) {
      setActiveTab('harmony');
    }
  }, [config, activeTab]);

  // Close snapshot menu on outside click
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (snapRef.current && !snapRef.current.contains(e.target as Node)) setShowSnapshots(false);
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  // Global keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) return;
      switch (e.key) {
        case 'd': handleDuplicate(); break;
        case 'g': handleGenerate(); break;
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  });

  // Close the fullscreen script editor on Escape.
  useEffect(() => {
    if (!consoleFullscreen) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setConsoleFullscreen(false); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [consoleFullscreen]);

  // Broadcast-to-all-voices modifier (hold Shift while editing a per-voice contour)
  useEffect(() => {
    const onDown = (e: KeyboardEvent) => { if (e.key === 'Shift') setBroadcastVoices(true); };
    const onUp = (e: KeyboardEvent) => { if (e.key === 'Shift') setBroadcastVoices(false); };
    const onBlur = () => setBroadcastVoices(false);
    window.addEventListener('keydown', onDown);
    window.addEventListener('keyup', onUp);
    window.addEventListener('blur', onBlur);
    return () => {
      window.removeEventListener('keydown', onDown);
      window.removeEventListener('keyup', onUp);
      window.removeEventListener('blur', onBlur);
    };
  }, []);

  const writeVoiceContour = (current: number[][] | undefined, d: number[]): number[][] => {
    const base = current ? [...current] : [];
    while (base.length < 16) base.push([]);
    if (!broadcastVoices) {
      base[selectedVoice] = d;
      return base;
    }
    // Broadcast mode: copy only the indices that actually changed in d
    // (compared to the selected voice's previous row) into every voice.
    const prev = base[selectedVoice] || [];
    const changed: number[] = [];
    for (let i = 0; i < d.length; i++) {
      if (d[i] !== prev[i]) changed.push(i);
    }
    if (changed.length === 0) {
      base[selectedVoice] = d;
      return base;
    }
    return base.map(row => {
      const nr = [...(row || [])];
      for (const i of changed) {
        while (nr.length <= i) nr.push(d[i]);
        nr[i] = d[i];
      }
      return nr;
    });
  };

  const handleGenerate = async () => {
    if (!config) return;
    setIsGenerating(true);
    setMessage('Generating MIDI...');
    saveSnapshot(config);
    setSnapshots(listSnapshots());
    try {
      const res = await fetch('http://127.0.0.1:3000/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text);
      }
      
      const data = await res.json();
      setMessage(data.message || data.status);
      setBreakdown(Array.isArray(data.breakdown) ? data.breakdown : []);
    } catch (err: any) {
      setMessage(`Error: ${err.message || err}`);
    }
    setIsGenerating(false);
  };

  const updateConfig = (key: string, value: any) => {
    setConfig({ ...config, [key]: value });
  };

  // ----- chord_templates: the chord-structure whitelist -----
  // Empty list = unconstrained. Each entry is a set of pitch-class offsets from
  // an arbitrary root; the backend admits it on all 12 roots.
  const chordTemplates = (): ChordTemplate[] =>
    Array.isArray(config?.chord_templates) ? config.chord_templates : [];

  const addChordTemplate = (pcs: number[]) => {
    const next = canonicalPcs(pcs);
    if (next.length === 0) return;
    const list = chordTemplates();
    // Rotation-equivalent duplicates would be a no-op for the backend but make
    // the list look like it holds two different rules.
    if (list.some((t) => rotationKey(tplPcs(t)) === rotationKey(next))) return;
    // Match the list's current mode, so adding a chord never silently switches
    // weighting on or off for the entries already there.
    const entry: ChordTemplate = weightsActive(list) ? { pcs: next, weight: 1 } : next;
    updateConfig('chord_templates', [...list, entry]);
  };

  const removeChordTemplate = (idx: number) =>
    updateConfig('chord_templates', chordTemplates().filter((_, i) => i !== idx));

  const clearChordTemplates = () => updateConfig('chord_templates', []);

  const setChordWeight = (idx: number, weight: number) =>
    updateConfig('chord_templates', chordTemplates().map((t, i) =>
      i === idx ? { pcs: tplPcs(t), weight: Math.max(0, weight) } : { pcs: tplPcs(t), weight: tplWeight(t) }));

  // Switching weights off drops back to bare offset lists — the backend reads
  // that as "any listed structure, optimiser's choice", not as equal weights.
  const toggleChordWeights = () => {
    const list = chordTemplates();
    updateConfig('chord_templates', weightsActive(list)
      ? list.map((t) => tplPcs(t))
      : list.map((t) => ({ pcs: tplPcs(t), weight: 1 })));
  };

  // Share of chords each entry gets, as the backend apportions it.
  const chordWeightShare = (idx: number): number => {
    const list = chordTemplates();
    const total = list.reduce((a, t) => a + Math.max(0, tplWeight(t)), 0);
    return total > 0 ? (Math.max(0, tplWeight(list[idx])) / total) * 100 : 0;
  };

  const toggleChordDraft = (pc: number) =>
    setChordDraft((d) => (d.includes(pc) ? d.filter((x) => x !== pc) : [...d, pc].sort((a, b) => a - b)));

  // Which config field (and whether it's per-voice) backs the active tab — used
  // to resolve the `$` shortcut in the script console to the on-screen contour.
  const CONTOUR_FIELDS: Record<string, { field: string; perVoice: boolean }> = {
    harmony: { field: 'harmony_distance_contour', perVoice: false },
    mode: { field: 'mode_contour', perVoice: false },
    chord: { field: 'chord_structure_contour', perVoice: true },
    voice: { field: 'voice_contour', perVoice: true },
    rhythm: { field: 'voice_rhythm_contour', perVoice: true },
    schillinger: { field: 'schillinger_sequence', perVoice: false },
    schillinger_ex: { field: 'schillinger_ex_contour', perVoice: true },
    harmony_matrix: { field: 'harmony_matrix_contour', perVoice: false },
    melody_force: { field: 'melody_force_contour', perVoice: true },
  };

  // Execute a user script against a mutable clone of the live config. Exposes
  // `config` (the draft), `$` (the active tab+voice contour, with write-back),
  // and a few helpers. Sloppy-mode `with` lets bare names resolve against the
  // scope proxy first, then fall back to globals (Math, Array, JSON, …).
  const runScript = (code: string) => {
    if (!config) return;
    let draft: any;
    try {
      draft = structuredClone(config);
    } catch {
      draft = JSON.parse(JSON.stringify(config));
    }
    const ref = CONTOUR_FIELDS[activeTab];
    const getDollar = () => {
      if (!ref) return undefined;
      return ref.perVoice ? (draft[ref.field]?.[selectedVoice]) : draft[ref.field];
    };
    const setDollar = (v: any) => {
      if (!ref) return;
      if (ref.perVoice) {
        if (!Array.isArray(draft[ref.field])) draft[ref.field] = [];
        while (draft[ref.field].length < 16) draft[ref.field].push([]);
        draft[ref.field][selectedVoice] = v;
      } else {
        draft[ref.field] = v;
      }
    };
    const range = (n: number) => Array.from({ length: Math.max(0, Math.floor(n)) }, (_, i) => i);
    const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
    const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
    const steps = (getDollar() as number[] | undefined)?.length ?? 0;
    const helpers: Record<string, any> = {
      range, clamp, lerp, steps, voice: selectedVoice, v: selectedVoice,
      res: config.voice_contour_resolution,
    };
    const scope = new Proxy({}, {
      has: () => true,
      get: (_t, key: any) => {
        if (key === Symbol.unscopables) return undefined;
        if (key === '$') return getDollar();
        if (key === 'config') return draft;
        if (key in helpers) return helpers[key];
        return (window as any)[key];
      },
      set: (_t, key: any, val: any) => {
        if (key === '$') { setDollar(val); return true; }
        if (key === 'config') { draft = val; return true; }
        helpers[key] = val;
        return true;
      },
    });
    try {
      // eslint-disable-next-line no-new-func
      const fn = new Function('__scope', `with(__scope){\n${code}\n}`);
      fn(scope);
      setConfig(draft);
      localStorage.setItem('contourScript', code);
      setScriptMsg({ text: 'Applied ✓', error: false });
    } catch (e: any) {
      setScriptMsg({ text: String(e?.message || e), error: true });
    }
  };

  // Current scoring matrix (falls back to defaults if backend didn't supply one).
  const getMatrix = (): number[][] => {
    const m = config?.harmony_matrix;
    if (Array.isArray(m) && m.length === 9 && m.every((r: any) => Array.isArray(r) && r.length === 12)) {
      return m;
    }
    return DEFAULT_HARMONY_MATRIX.map(r => [...r]);
  };

  const updateMatrixCell = (row: number, col: number, value: number) => {
    const m = getMatrix().map(r => [...r]);
    m[row][col] = Number.isFinite(value) ? value : 0;
    updateConfig('harmony_matrix', m);
  };

  const resetMatrix = () => {
    updateConfig('harmony_matrix', DEFAULT_HARMONY_MATRIX.map(r => [...r]));
  };

  // Start notes — the 5 seed pitches (high → low) for the generated voices.
  // Always normalised to exactly 5 slots, filling gaps from the defaults.
  const getStartNotes = (): number[] => {
    const s = config?.start_notes;
    return Array.from({ length: 5 }, (_, i) =>
      Array.isArray(s) && Number.isFinite(s[i]) ? s[i] : DEFAULT_START_NOTES[i]
    );
  };

  const updateStartNote = (i: number, v: number) => {
    const s = getStartNotes();
    s[i] = Number.isFinite(v) ? Math.round(v) : DEFAULT_START_NOTES[i];
    updateConfig('start_notes', s);
  };

  const resetStartNotes = () => {
    updateConfig('start_notes', [...DEFAULT_START_NOTES]);
    setChordMsg(null);
  };

  // Fetch the last chord of an Ableton clip and drop its pitches (high → low)
  // into the start-note slots — the "gen a phrase, seed the next one from the
  // resolving chord" loop. Fewer pitches than slots leaves the rest untouched.
  const fetchLastChord = async () => {
    setFetchingChord(true);
    setChordMsg(null);
    try {
      const res = await fetch(`http://127.0.0.1:3000/api/last-chord/${snTrack}/${snClip}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.message || `HTTP ${res.status}`);
      const notes: number[] = Array.isArray(data.notes) ? data.notes : [];
      if (notes.length === 0) {
        setChordMsg({ text: 'Clip has no notes', error: true });
        return;
      }
      const s = getStartNotes();
      for (let i = 0; i < 5 && i < notes.length; i++) s[i] = Math.round(notes[i]);
      updateConfig('start_notes', s);
      setChordMsg({
        text: `Loaded ${Math.min(notes.length, 5)} note(s): ${notes.slice(0, 5).map(midiName).join(' ')}`,
        error: false,
      });
    } catch (err: any) {
      setChordMsg({ text: `Fetch failed: ${err.message || err}`, error: true });
    }
    setFetchingChord(false);
  };

  const handleDuplicate = () => {
    if (!config) return;
    
    // Explicitly mirror arrays filling gaps enforcing perfect duplication limits natively
    const duplicateArray = (arr: any[], steps: number, fallback: any) => {
      const clean = Array.from({ length: steps }).map((_, i) => arr && arr[i] !== undefined ? arr[i] : fallback);
      return [...clean, ...clean];
    };

    const newRL = config.render_length * 2;
    const stdSteps = Math.ceil((config.pl * 4 * config.render_length) / config.voice_contour_resolution);
    const schillingerSteps = config.pl * config.render_length;

    const newConfig = { ...config };
    newConfig.render_length = newRL;

    if (newConfig.schillinger_sequence) newConfig.schillinger_sequence = duplicateArray(newConfig.schillinger_sequence, schillingerSteps, 0);
    if (newConfig.harmony_distance_contour) newConfig.harmony_distance_contour = duplicateArray(newConfig.harmony_distance_contour, stdSteps, 0.2);
    if (newConfig.mode_contour) newConfig.mode_contour = duplicateArray(newConfig.mode_contour, stdSteps, 0);
    if (newConfig.chord_structure_contour) {
      newConfig.chord_structure_contour = newConfig.chord_structure_contour.map((track: any[]) => duplicateArray(track, stdSteps, 0));
    }
    if (newConfig.schillinger_ex_contour) {
      newConfig.schillinger_ex_contour = newConfig.schillinger_ex_contour.map((track: any[]) => duplicateArray(track, stdSteps, 2));
    }
    if (newConfig.harmony_matrix_contour) newConfig.harmony_matrix_contour = duplicateArray(newConfig.harmony_matrix_contour, stdSteps, 0);
    if (newConfig.melody_force_contour) {
      newConfig.melody_force_contour = newConfig.melody_force_contour.map((track: any[]) => duplicateArray(track, stdSteps, newConfig.melody_force ?? 0));
    }

    if (newConfig.voice_contour) {
      newConfig.voice_contour = newConfig.voice_contour.map((track: any[]) => duplicateArray(track, stdSteps, 0));
    }
    if (newConfig.voice_rhythm_contour) {
      newConfig.voice_rhythm_contour = newConfig.voice_rhythm_contour.map((track: any[]) => duplicateArray(track, stdSteps, 4.0));
    }

    setConfig(newConfig);
  };

  const resampleContour = (data: number[], oldRes: number, newRes: number): number[] => {
    if (oldRes === newRes || data.length === 0) return data;
    const totalBeats = data.length * oldRes;
    const newLen = Math.ceil(totalBeats / newRes);
    return Array.from({ length: newLen }, (_, i) => {
      const oldIdx = Math.min(Math.floor((i * newRes) / oldRes), data.length - 1);
      return data[oldIdx];
    });
  };

  const handleResolutionChange = (newRes: number) => {
    if (!config) return;
    const oldRes = config.voice_contour_resolution;
    if (oldRes === newRes) return;

    const nc = { ...config, voice_contour_resolution: newRes };
    if (nc.harmony_distance_contour) nc.harmony_distance_contour = resampleContour(nc.harmony_distance_contour, oldRes, newRes);
    if (nc.mode_contour) nc.mode_contour = resampleContour(nc.mode_contour, oldRes, newRes);
    if (nc.chord_structure_contour) nc.chord_structure_contour = nc.chord_structure_contour.map((t: number[]) => resampleContour(t, oldRes, newRes));
    if (nc.schillinger_ex_contour) nc.schillinger_ex_contour = nc.schillinger_ex_contour.map((t: number[]) => resampleContour(t, oldRes, newRes));
    if (nc.harmony_matrix_contour) nc.harmony_matrix_contour = resampleContour(nc.harmony_matrix_contour, oldRes, newRes);
    if (nc.melody_force_contour) nc.melody_force_contour = nc.melody_force_contour.map((t: number[]) => resampleContour(t, oldRes, newRes));
    if (nc.voice_contour) nc.voice_contour = nc.voice_contour.map((t: number[]) => resampleContour(t, oldRes, newRes).map(Math.round));
    if (nc.voice_rhythm_contour) nc.voice_rhythm_contour = nc.voice_rhythm_contour.map((t: number[]) => resampleContour(t, oldRes, newRes));
    setConfig(nc);
  };

  const generateMarkovProgression = (totalLength: number, mode: number, phraseLength: number): number[] => {
    if (totalLength === 0 || phraseLength === 0) return [];
    
    type Transition = [number, number]; // [target_chord, weight]
    const allTransitions: Transition[][][] = [
      /* Ionian */ [
        [[4, 40], [3, 30], [5, 20], [1, 5], [2, 5]], // 0: I -> V, IV, vi
        [[4, 70], [6, 30]],                          // 1: ii -> V, vii
        [[5, 60], [3, 40]],                          // 2: iii -> vi, IV
        [[0, 30], [4, 50], [1, 20]],                 // 3: IV -> V, I, ii
        [[0, 70], [5, 30]],                          // 4: V -> I, vi
        [[3, 40], [1, 40], [4, 20]],                 // 5: vi -> IV, ii, V
        [[0, 100]]                                   // 6: vii -> I
      ],
      /* Dorian */ [
        [[3, 50], [6, 30], [1, 20]], // i -> IV, VII, ii
        [[0, 70], [3, 30]],          // ii -> i, IV
        [[3, 60], [4, 40]],          // III -> IV, v
        [[0, 60], [6, 40]],          // IV -> i, VII
        [[0, 50], [3, 50]],          // v -> i, IV
        [[6, 50], [3, 50]],          // vi° -> VII, IV
        [[0, 70], [3, 30]]           // VII -> i, IV
      ],
      /* Phrygian */ [
        [[1, 50], [3, 30], [5, 20]], // i -> II, iv, VI
        [[0, 100]],                  // II -> i
        [[1, 60], [3, 40]],          // III -> II, iv
        [[0, 60], [1, 40]],          // iv -> i, II
        [[1, 50], [5, 50]],          // v° -> II, VI
        [[1, 60], [0, 40]],          // VI -> II, i
        [[0, 60], [2, 40]]           // vii -> i, III
      ],
      /* Lydian */ [
        [[1, 50], [4, 30], [2, 20]], // I -> II, V, iii
        [[0, 60], [4, 40]],          // II -> I, V
        [[0, 50], [1, 50]],          // iii -> I, II
        [[4, 60], [2, 40]],          // iv° -> V, iii
        [[0, 60], [1, 40]],          // V -> I, II
        [[1, 50], [4, 50]],          // vi -> II, V
        [[0, 70], [2, 30]]           // vii -> I, iii
      ],
      /* Mixolydian */ [
        [[3, 40], [6, 40], [4, 20]], // I -> IV, VII, v
        [[0, 50], [3, 50]],          // ii -> I, IV
        [[3, 60], [5, 40]],          // iii° -> IV, vi
        [[0, 60], [6, 40]],          // IV -> I, VII
        [[0, 60], [3, 40]],          // v -> I, IV
        [[3, 50], [1, 50]],          // vi -> IV, ii
        [[0, 60], [3, 40]]           // VII -> I, IV
      ],
      /* Aeolian */ [
        [[3, 30], [4, 30], [5, 20], [6, 10], [2, 10]], // i -> iv, v, VI, VII, III
        [[4, 60], [6, 40]],                            // ii° -> v, VII
        [[5, 60], [3, 40]],                            // III -> VI, iv
        [[4, 40], [0, 30], [6, 20], [1, 10]],          // iv -> v, i, VII, ii°
        [[0, 70], [5, 30]],                            // v -> i, VI
        [[3, 40], [4, 40], [1, 20]],                   // VI -> iv, v, ii°
        [[2, 60], [0, 40]]                             // VII -> III, i
      ],
      /* Locrian */ [
        [[1, 40], [3, 30], [5, 30]], // i° -> II, iv, VI
        [[0, 60], [3, 40]],          // II -> i°, iv
        [[1, 50], [5, 50]],          // iii -> II, VI
        [[0, 60], [1, 40]],          // iv -> i°, II
        [[1, 50], [5, 50]],          // V -> II, VI
        [[0, 60], [1, 40]],          // VI -> i°, II
        [[0, 50], [5, 50]]           // vii -> i°, VI
      ],
    ];

    const transitions = allTransitions[Math.max(0, Math.min(6, mode))];
    // End each phrase on V (scale degree 5). Because the sequence loops,
    // the V -> I resolution happens across the loop seam (bar N -> bar 1).
    const target = 4;
    const phraseStart = 0; // Each phrase opens on I (tonic)

    const findPathPhrase = (current: number, remaining: number, path: number[]): boolean => {
      if (remaining === 1) {
        if (current === target) {
          path.push(current);
          return true;
        }
        return false;
      }

      path.push(current);
      const opts = transitions[current];
      const choices: number[] = [];
      for (const [tgt, weight] of opts) {
        for (let i = 0; i < weight; i++) {
          choices.push(tgt);
        }
      }

      // Shuffle
      for (let i = choices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [choices[i], choices[j]] = [choices[j], choices[i]];
      }

      const tried = Array(7).fill(false);
      for (const nextChord of choices) {
        if (nextChord >= 7 || tried[nextChord]) continue;
        tried[nextChord] = true;
        
        if (findPathPhrase(nextChord, remaining - 1, path)) return true;
      }

      path.pop();
      return false;
    };

    const generatePhrase = (len: number, startChord: number): number[] => {
      if (len === 0) return [];
      const path: number[] = [];
      if (findPathPhrase(startChord, len, path)) return path;
      
      const fallback = Array(len).fill(0);
      fallback[0] = startChord;
      if (len > 0) fallback[len - 1] = target;
      for (let i = 1; i < len - 1; i++) {
        fallback[i] = transitions[fallback[i - 1]][0][0];
      }
      return fallback;
    };

    const progression: number[] = [];
    const numPhrases = Math.floor(totalLength / phraseLength);

    for (let i = 0; i < numPhrases; i++) {
      const block = generatePhrase(phraseLength, phraseStart);
      progression.push(...block);
    }

    const remainder = totalLength % phraseLength;
    if (remainder > 0) {
      const block = generatePhrase(remainder, phraseStart);
      progression.push(...block);
    }

    return progression;
  };

  const applyPreset = (preset: 'jazz' | 'classical' | 'ambient' | 'trance') => {
    if (!config) return;
    const nc = { ...config };

    if (preset === 'jazz') {
      // Jazz: Dorian/Mixolydian modes, rich voicings, relaxed voice-leading, syncopation
      nc.pl = 4;
      nc.main_pitch = Math.floor(Math.random() * 12); // random key
      nc.mode = 1; // Dorian
      nc.lookahead_depth = 3;
      nc.harmony_distance_balance = 0.35;
      nc.rng_seed = Math.floor(Math.random() * 999999);

      // Relaxed classical rules — jazz embraces parallel motion & voice
      // overlap. Repeat/melodic terms live on the rescaled ±1 scale.
      nc.last_note_exist_in_voice = 1.0;
      nc.same_direction = 0.3;
      nc.consecutive_octav_fift = 0.0; // parallel 5ths (and antiparallel octaves) are fine in jazz
      nc.no_crossing = 1.5; // voice overlap tolerated
      nc.last_note_same = 0.5;
      nc.same_note_bonus = 0.5; // light hold bias: comping breathes, lines keep moving
      nc.common_tone_penalty = 0.0;
      nc.max_voices_changed = -1;
      nc.min_voices_changed = -1;
      nc.interval_exists_in_harmony = 0.5;
      nc.chord_structure = [0, 1, 2, 3, 4, 5]; // full 7th chord voicings

      // Root-aware harmony: jazz plays inversions and rootless voicings freely
      // and avoids octave doublings, but guide-tone resolution (7th → 3rd) is
      // the whole point of a ii-V-I, so tendency stays strong.
      nc.root_position_weight = 0.3;
      nc.root_doubling_weight = 0.0;
      nc.chord_quality_weight = 1.0;
      nc.tendency_weight = 0.8;
      nc.roughness_weight = 0.4;

      // The preset writes its own progression — the generated progression
      // would silently override it, and a custom matrix would override the
      // style rows the matrix contour points at.
      nc.schillinger_progression = true;
      nc.use_generated_progression = false;
      nc.harmony_matrix = DEFAULT_HARMONY_MATRIX.map(r => [...r]);

      // Jazz progression: loop-friendly turnarounds — each 4-bar phrase
      // starts on I and lands on V so the I resolution falls on bar 1 of
      // the next iteration (across the loop seam).
      // 0=I, 1=ii, 2=iii, 3=IV, 4=V, 5=vi, 6=vii
      const jazzBars = nc.pl * nc.render_length; // 32 bars
      const jazzPatterns = [
        [0, 5, 1, 4],   // I-vi-ii-V (classic turnaround)
        [0, 3, 1, 4],   // I-IV-ii-V (rhythm changes)
        [0, 2, 5, 4],   // I-iii-vi-V
        [0, 6, 1, 4],   // I-vii-ii-V
        [0, 3, 5, 4],   // I-IV-vi-V
        [0, 1, 3, 4],   // I-ii-IV-V
        [0, 5, 3, 4],   // I-vi-IV-V
        [0, 2, 1, 4],   // I-iii-ii-V
      ];
      nc.schillinger_sequence = [];
      for (let i = 0; i < jazzBars; i += 4) {
        const pat = jazzPatterns[Math.floor(Math.random() * jazzPatterns.length)];
        nc.schillinger_sequence.push(...pat.slice(0, Math.min(4, jazzBars - i)));
      }

      // Contours
      const steps = Math.ceil((nc.pl * 4 * nc.render_length) / nc.voice_contour_resolution);

      // Harmony distance: medium-high tension with jazz push-pull
      nc.harmony_distance_contour = Array.from({ length: steps }, (_, i) => {
        const phase = i / steps;
        return parseFloat((0.25 + 0.15 * Math.sin(phase * Math.PI * 3) + 0.05 * Math.sin(phase * Math.PI * 7)).toFixed(2));
      });

      // Mode: alternate Dorian(1) / Mixolydian(4) / Dorian / Aeolian(5) for color
      nc.mode_contour = Array.from({ length: steps }, (_, i) => {
        const section = Math.floor((i / steps) * 4);
        return [1, 4, 1, 5][section] ?? 1;
      });

      // Chord structure: rich voicings throughout, thicker at climax — per voice
      nc.chord_structure_contour = Array.from({ length: 16 }, () =>
        Array.from({ length: steps }, (_, i) => {
          const phase = i / steps;
          return Math.round(3 + 2 * Math.sin(phase * Math.PI));
        })
      );

      // Schillinger expansion: moderate, jazzier wider voicings — per voice, slight offset per voice
      nc.schillinger_ex_contour = Array.from({ length: 16 }, (_, v) =>
        Array.from({ length: steps }, (_, i) => {
          const phase = i / steps;
          return Math.round(3 + Math.sin(phase * Math.PI * 2 + v * 0.3));
        })
      );

      // Rhythm: syncopated — mix of 8ths, dotted quarters, swung feel
      const jazzSnaps = [0.5, 0.75, 1.0, 0.5, 0.75, 1.0, 2.0];
      nc.voice_rhythm_contour = Array.from({ length: 16 }, (_, v) =>
        Array.from({ length: steps }, (_, i) => {
          const phase = i / steps;
          const localPhase = (i % nc.pl) / nc.pl;
          // Walking bass (channel 4 is the bass): steady quarter notes
          if (v === 4) return 1.0;
          // Lead (channel 0): longer, more lyrical
          if (v === 0) return [1.0, 2.0, 0.75, 1.0][Math.floor(i % 4)];
          // Comping voices: syncopated
          return jazzSnaps[Math.floor((phase * 7 + localPhase * 3 + v) % jazzSnaps.length)];
        })
      );

      // Voice pitch contour: gentle melodic motion around each voice's own
      // register (channel 0 = lead on top, channel 4 = bass).
      nc.voice_contour = Array.from({ length: 16 }, (_, v) =>
        Array.from({ length: steps }, (_, i) => {
          const phase = i / steps;
          const base = v === 0 ? 3 : v === 4 ? -3 : 0;
          return Math.round(base + 3 * Math.sin(phase * Math.PI * (2 + v * 0.3)));
        })
      );

      // Harmony matrix: Jazz(1) base, dipping into Tension(2) at phrase peaks, Ethereal(3) for ballad sections
      nc.harmony_matrix_contour = Array.from({ length: steps }, (_, i) => {
        const phase = i / steps;
        const section = Math.floor(phase * 4);
        return [1, 2, 1, 3][section] ?? 1; // Jazz → Tension → Jazz → Ethereal
      });

      // Melody force: moderate-to-strong moving lines, peaking mid-phrase (0-1)
      nc.melody_force_contour = Array.from({ length: 16 }, () => Array.from({ length: steps }, (_, i) => {
        const tension = Math.sin((i / Math.max(steps - 1, 1)) * Math.PI);
        return parseFloat((0.6 + tension * 0.3).toFixed(2));
      }));

    } else if (preset === 'classical') {
      // Classical: Ionian/Aeolian modes, strict voice-leading, balanced phrases
      nc.pl = 4;
      nc.main_pitch = Math.floor(Math.random() * 12); // random key
      nc.mode = 0; // Ionian (major)
      nc.lookahead_depth = 3;
      nc.harmony_distance_balance = 0.2;
      nc.rng_seed = Math.floor(Math.random() * 999999);

      // Strict classical voice-leading rules (rescaled ±1 melodic terms;
      // rule penalties a few × the soft-score range are already prohibitive)
      nc.last_note_exist_in_voice = 1.0;
      nc.same_direction = 1.0; // encourage contrary motion
      nc.consecutive_octav_fift = 5.0; // parallel AND antiparallel 5ths/octaves effectively banned
      nc.no_crossing = 10.0; // strict voice separation
      nc.last_note_same = 0.5;
      nc.same_note_bonus = 1.0; // chorale hold bias: non-leaders keep common tones
      nc.common_tone_penalty = 0.0;
      nc.max_voices_changed = -1;
      nc.min_voices_changed = -1;
      // Lowered from 2.0: root doubling is now rewarded explicitly (below), and
      // this term pushes the other way on exactly those octaves.
      nc.interval_exists_in_harmony = 1.0;
      nc.chord_structure = [0, 1, 2, 4]; // triads & simple 7ths

      // Root-aware harmony: functional bass, classical doubling policy (double
      // the root, never the leading tone), and strong tendency-tone resolution.
      nc.root_position_weight = 2.0;
      nc.root_doubling_weight = 1.0;
      nc.chord_quality_weight = 1.5;
      nc.tendency_weight = 1.5;
      // 0.4, not higher: pure roughness ranks 4ths/5ths above 3rds, and at
      // high weight it drains the thirds out of the texture entirely.
      nc.roughness_weight = 0.4;

      // The preset writes its own progression and points at built-in matrix rows.
      nc.schillinger_progression = true;
      nc.use_generated_progression = false;
      nc.harmony_matrix = DEFAULT_HARMONY_MATRIX.map(r => [...r]);

      // Classical progressions: each 4-bar phrase starts on I and ends on V
      // so the authentic cadence (V -> I) resolves across the loop boundary.
      const classBars = nc.pl * nc.render_length;
      const classicalPatterns = [
        [0, 3, 0, 4],   // I-IV-I-V (prolonged tonic → dominant)
        [0, 5, 3, 4],   // I-vi-IV-V (50s progression → cadence)
        [0, 1, 3, 4],   // I-ii-IV-V (predominant approach)
        [0, 3, 1, 4],   // I-IV-ii-V (circle of fifths)
        [0, 2, 3, 4],   // I-iii-IV-V
        [0, 5, 1, 4],   // I-vi-ii-V
        [0, 2, 5, 4],   // I-iii-vi-V
        [0, 4, 5, 4],   // I-V-vi-V (deceptive, returns to V)
      ];
      nc.schillinger_sequence = [];
      for (let i = 0; i < classBars; i += 4) {
        // Structured: exposition, development, recapitulation feel
        const section = i / classBars;
        let poolIdx: number;
        if (section < 0.3) poolIdx = Math.floor(Math.random() * 3); // stable openings
        else if (section < 0.7) poolIdx = 3 + Math.floor(Math.random() * 3); // development
        else poolIdx = Math.floor(Math.random() * 2); // return to tonic
        nc.schillinger_sequence.push(...classicalPatterns[poolIdx].slice(0, Math.min(4, classBars - i)));
      }
      // Final bar = V so the loop seam delivers the V -> I authentic cadence.
      nc.schillinger_sequence[nc.schillinger_sequence.length - 1] = 4;

      const steps = Math.ceil((nc.pl * 4 * nc.render_length) / nc.voice_contour_resolution);

      // Harmony distance: classical arch — low tension at start/end, peak in middle
      nc.harmony_distance_contour = Array.from({ length: steps }, (_, i) => {
        const phase = i / steps;
        return parseFloat((0.15 + 0.25 * Math.sin(phase * Math.PI)).toFixed(2));
      });

      // Mode: primarily Ionian(0), brief Aeolian(5) in development
      nc.mode_contour = Array.from({ length: steps }, (_, i) => {
        const phase = i / steps;
        // Development section goes minor
        return (phase > 0.35 && phase < 0.65) ? 5 : 0;
      });

      // Chord structure: triads expanding to 7ths at climax — per voice
      nc.chord_structure_contour = Array.from({ length: 16 }, () =>
        Array.from({ length: steps }, (_, i) => {
          const phase = i / steps;
          return Math.round(1 + 3 * Math.sin(phase * Math.PI));
        })
      );

      // Schillinger expansion: moderate, wider at climax — per voice
      nc.schillinger_ex_contour = Array.from({ length: 16 }, (_, v) =>
        Array.from({ length: steps }, (_, i) => {
          const phase = i / steps;
          return Math.round(2 + 2 * Math.sin(phase * Math.PI) + (v % 2 === 0 ? 0 : 0.5));
        })
      );

      // Rhythm: stately — mostly quarter and half notes, faster at climax
      nc.voice_rhythm_contour = Array.from({ length: 16 }, (_, v) =>
        Array.from({ length: steps }, (_, i) => {
          const phase = i / steps;
          const tension = Math.sin(phase * Math.PI);
          // Higher voices move faster at climax
          if (v <= 1) {
            // Soprano/alto: quarters at rest, eighths at climax
            return tension > 0.6 ? 0.5 : 1.0;
          }
          // Lower voices: half/whole notes, slightly faster at climax
          return tension > 0.7 ? 1.0 : 2.0;
        })
      );

      // Voice pitch contour: smooth, arched (channel 0 = soprano, 4 = bass;
      // the bass arcs against the upper voices for contrary large-scale motion)
      nc.voice_contour = Array.from({ length: 16 }, (_, v) =>
        Array.from({ length: steps }, (_, i) => {
          const phase = i / steps;
          const base = v === 0 ? 4 : v === 3 ? -2 : v === 4 ? -5 : 0;
          return Math.round(base + 4 * Math.sin(phase * Math.PI) * (v >= 3 ? -0.5 : 1));
        })
      );

      // Harmony matrix: Strict Classical(0) base, Dark(4) in development, back to Classical(0)
      nc.harmony_matrix_contour = Array.from({ length: steps }, (_, i) => {
        const phase = i / steps;
        if (phase > 0.35 && phase < 0.65) return 4; // Dark/Melancholic for development
        return 0; // Strict Classical for exposition/recap
      });

      // Melody force: gentle stepwise pressure, easing at phrase boundaries (0-1)
      nc.melody_force_contour = Array.from({ length: 16 }, () => Array.from({ length: steps }, (_, i) => {
        const tension = Math.sin((i / Math.max(steps - 1, 1)) * Math.PI);
        return parseFloat((0.3 + tension * 0.2).toFixed(2));
      }));
    } else if (preset === 'ambient') {
      // Ambient: Lydian float over the Ethereal matrix row, glacial harmonic
      // rhythm, deep hold bias with a 1-voice change budget — the texture
      // drifts one voice at a time instead of progressing.
      nc.pl = 4;
      nc.main_pitch = Math.floor(Math.random() * 12);
      nc.mode = 3; // Lydian
      nc.lookahead_depth = 2;
      nc.harmony_distance_balance = 0.1;
      nc.rng_seed = Math.floor(Math.random() * 999999);

      nc.last_note_exist_in_voice = 0.3;
      nc.same_direction = 0.0;
      nc.consecutive_octav_fift = 0.0;
      nc.no_crossing = 2.0;
      nc.last_note_same = 0.1;
      nc.same_note_bonus = 3.0; // deep hold bias — the pad sound
      nc.common_tone_penalty = 0.0;
      nc.interval_exists_in_harmony = 0.0; // octave doublings welcome
      nc.max_voices_changed = 1; // at most one voice moves per chord
      nc.min_voices_changed = -1;
      nc.chord_structure = [0, 1, 2];

      nc.root_position_weight = 0.3;
      nc.root_doubling_weight = 0.0;
      nc.chord_quality_weight = 0.5;
      nc.tendency_weight = 0.0; // no functional pull — stasis is the point
      nc.roughness_weight = 0.6;

      nc.schillinger_progression = true;
      nc.use_generated_progression = false;
      nc.harmony_matrix = DEFAULT_HARMONY_MATRIX.map(r => [...r]);

      // Two-chord oscillations, four bars each — I↔II is the Lydian float,
      // never a functional cadence.
      const ambBars = nc.pl * nc.render_length;
      const ambPatterns = [
        [0, 0, 1, 1],   // I - II (the Lydian signature)
        [0, 0, 5, 5],   // I - vi
        [0, 1, 0, 1],   // faster I-II rocking
        [0, 0, 4, 4],   // I - V as color, not cadence
      ];
      nc.schillinger_sequence = [];
      for (let i = 0; i < ambBars; i += 4) {
        const pat = ambPatterns[Math.floor(Math.random() * ambPatterns.length)];
        nc.schillinger_sequence.push(...pat.slice(0, Math.min(4, ambBars - i)));
      }

      const steps = Math.ceil((nc.pl * 4 * nc.render_length) / nc.voice_contour_resolution);

      // Barely-moving harmony weight with one slow swell.
      nc.harmony_distance_contour = Array.from({ length: steps }, (_, i) =>
        parseFloat((0.1 + 0.1 * Math.sin((i / Math.max(steps - 1, 1)) * Math.PI)).toFixed(2)));
      nc.mode_contour = Array.from({ length: steps }, () => 3);
      nc.harmony_matrix_contour = Array.from({ length: steps }, () => 3); // Ethereal throughout
      // Open voicings: bare triads, spread wide by the expansion.
      nc.chord_structure_contour = Array.from({ length: 16 }, () =>
        Array.from({ length: steps }, () => 0));
      nc.schillinger_ex_contour = Array.from({ length: 16 }, (_, v) =>
        Array.from({ length: steps }, () => 3 + (v % 2)));
      // Long tones phasing against each other: 2- and 4-beat notes, offset per voice.
      nc.voice_rhythm_contour = Array.from({ length: 16 }, (_, v) =>
        Array.from({ length: steps }, (_, i) => ((i + v) % 3 === 0 ? 2.0 : 4.0)));
      nc.voice_contour = Array.from({ length: 16 }, (_, v) =>
        Array.from({ length: steps }, (_, i) =>
          Math.round(2 * Math.sin((i / Math.max(steps - 1, 1)) * Math.PI + v))));
      // Almost no line pressure: lines are allowed to sleep.
      nc.melody_force_contour = Array.from({ length: 16 }, () =>
        Array.from({ length: steps }, () => 0.1));
    } else {
      // Trance: Aeolian anthem loops (i-VI-III-VII family), pumping bass and
      // 8th-note lead over held pads, Dark matrix lifting to Bright at the
      // climax. Parallel octaves/5ths and octave doubling ARE the genre.
      nc.pl = 4;
      nc.main_pitch = Math.floor(Math.random() * 12);
      nc.mode = 5; // Aeolian
      nc.lookahead_depth = 3;
      nc.harmony_distance_balance = 0.3;
      nc.rng_seed = Math.floor(Math.random() * 999999);

      nc.last_note_exist_in_voice = 0.5;
      nc.same_direction = 0.0;
      nc.consecutive_octav_fift = 0.0;
      nc.no_crossing = 3.0;
      nc.last_note_same = 0.2;
      nc.same_note_bonus = 2.0; // pads hold until the chord change forces them
      nc.common_tone_penalty = 0.0;
      nc.interval_exists_in_harmony = 0.0;
      nc.max_voices_changed = -1;
      nc.min_voices_changed = -1;
      nc.chord_structure = [0, 1, 2];

      nc.root_position_weight = 2.0; // the bass hammers chord roots
      nc.root_doubling_weight = 1.0;
      nc.chord_quality_weight = 1.0;
      nc.tendency_weight = 0.3;
      nc.roughness_weight = 0.5;

      nc.schillinger_progression = true;
      nc.use_generated_progression = false;
      nc.harmony_matrix = DEFAULT_HARMONY_MATRIX.map(r => [...r]);

      // Anthem loops. Degree-4 bars get the harmonic-minor leading tone
      // automatically (Aeolian V), so patterns ending on V deliver real lift.
      const trBars = nc.pl * nc.render_length;
      const trPatterns = [
        [0, 5, 2, 6],   // i-VI-III-VII (the anthem)
        [0, 6, 5, 6],   // i-VII-VI-VII
        [0, 5, 6, 4],   // i-VI-VII-V (V carries the raised leading tone)
        [0, 3, 5, 6],   // i-iv-VI-VII
      ];
      nc.schillinger_sequence = [];
      for (let i = 0; i < trBars; i += 4) {
        const pat = trPatterns[Math.floor(Math.random() * trPatterns.length)];
        nc.schillinger_sequence.push(...pat.slice(0, Math.min(4, trBars - i)));
      }

      const steps = Math.ceil((nc.pl * 4 * nc.render_length) / nc.voice_contour_resolution);

      // Build: harmony weight rises steadily toward the end of the loop.
      nc.harmony_distance_contour = Array.from({ length: steps }, (_, i) =>
        parseFloat((0.2 + 0.15 * (i / Math.max(steps - 1, 1))).toFixed(2)));
      nc.mode_contour = Array.from({ length: steps }, () => 5);
      nc.harmony_matrix_contour = Array.from({ length: steps }, (_, i) => {
        const section = Math.floor((i / steps) * 4);
        return [4, 4, 5, 4][section] ?? 4; // Dark → Dark → Bright climax → Dark
      });
      // Triads, thickening briefly at the climax.
      nc.chord_structure_contour = Array.from({ length: 16 }, () =>
        Array.from({ length: steps }, (_, i) => {
          const phase = i / steps;
          return phase > 0.5 && phase < 0.75 ? 2 : 0;
        }));
      nc.schillinger_ex_contour = Array.from({ length: 16 }, () =>
        Array.from({ length: steps }, () => 2));
      // Lead (ch 0) and bass (ch 4) run 8ths; inner pads hold long tones.
      nc.voice_rhythm_contour = Array.from({ length: 16 }, (_, v) =>
        Array.from({ length: steps }, () => {
          if (v === 0 || v === 4) return 0.5;
          if (v === 3) return 2.0;
          return 4.0;
        }));
      // Lead climbs through each half of the loop and resets — a riser.
      nc.voice_contour = Array.from({ length: 16 }, (_, v) =>
        Array.from({ length: steps }, (_, i) => {
          if (v !== 0) return 0;
          const half = Math.max(Math.ceil(steps / 2), 1);
          const phase = (i % half) / Math.max(half - 1, 1);
          return Math.round(phase * 7);
        }));
      // Line pressure on the lead only; pads stay glued to their tones.
      nc.melody_force_contour = Array.from({ length: 16 }, (_, v) =>
        Array.from({ length: steps }, (_, i) => {
          if (v !== 0) return 0.15;
          const tension = Math.sin((i / Math.max(steps - 1, 1)) * Math.PI);
          return parseFloat((0.8 + 0.4 * tension).toFixed(2));
        }));
    }

    setConfig(nc);
    setMessage(`${preset.charAt(0).toUpperCase() + preset.slice(1)} preset loaded`);
  };

  const handleInit = async () => {
    // Re-pull the server's default config — the reasonable baseline (rescaled
    // penalties, common-tone control, default H-matrix). Snapshots are untouched.
    try {
      const res = await fetch('http://127.0.0.1:3000/api/config');
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setConfig(data);
      setMessage('Reset to default config');
    } catch (err: any) {
      setMessage(`Error loading defaults: ${err.message || err}`);
    }
  };

  const handleRandomise = () => {
    if (!config) return;
    const steps = Math.ceil((config.pl * 4 * config.render_length) / config.voice_contour_resolution);

    // Smooth random: interpolate between random anchor points
    const smoothRandom = (min: number, max: number, pts: number = 6): number[] => {
      const anchors = Array.from({ length: pts }, (_, i) => ({
        x: i === 0 ? 0 : i === pts - 1 ? steps - 1 : Math.floor(Math.random() * steps),
        y: min + Math.random() * (max - min)
      })).sort((a, b) => a.x - b.x);
      return Array.from({ length: steps }, (_, i) => {
        let l = anchors[0], r = anchors[anchors.length - 1];
        for (let j = 0; j < anchors.length - 1; j++) {
          if (i >= anchors[j].x && i <= anchors[j + 1].x) { l = anchors[j]; r = anchors[j + 1]; break; }
        }
        if (l.x === r.x) return l.y;
        const t = (i - l.x) / (r.x - l.x);
        return l.y + t * (r.y - l.y);
      });
    };

    const snaps = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0];
    const snapNearest = (v: number) => snaps.reduce((a, b) => Math.abs(b - v) < Math.abs(a - v) ? b : a);

    const nc = { ...config };
    nc.main_pitch = Math.floor(Math.random() * 12);
    nc.harmony_distance_contour = smoothRandom(-0.2, 0.5).map(v => parseFloat(v.toFixed(2)));

    // Mode contour: pick 2-4 modes and assign them to sections at phrase boundaries
    // Musically coherent transitions — modes change per phrase, not per beat
    const beatsPerPhrase = nc.pl * 4;
    const stepsPerPhrase = Math.ceil(beatsPerPhrase / nc.voice_contour_resolution);
    const totalPhrases = nc.render_length;
    const numSections = 2 + Math.floor(Math.random() * 3); // 2-4 sections
    // Pick modes that are related (neighbors on circle of fifths: 0↔4↔1↔5↔2↔6↔3)
    const relatedModes = [[0, 4], [0, 5], [1, 4], [1, 5], [0, 3, 4], [1, 4, 0], [5, 1, 4, 0], [0, 5, 1, 4]];
    const modeSet = relatedModes[Math.floor(Math.random() * relatedModes.length)];
    const sectionModes: number[] = [];
    for (let s = 0; s < numSections; s++) {
      sectionModes.push(modeSet[Math.floor(Math.random() * modeSet.length)]);
    }
    nc.mode_contour = Array.from({ length: steps }, (_, i) => {
      const phraseIdx = Math.floor(i / stepsPerPhrase);
      const sectionIdx = Math.min(Math.floor((phraseIdx / totalPhrases) * numSections), numSections - 1);
      return sectionModes[sectionIdx];
    });

    nc.chord_structure_contour = Array.from({ length: 16 }, () => smoothRandom(0, 5).map(v => Math.round(v)));
    nc.schillinger_ex_contour = Array.from({ length: 16 }, () => smoothRandom(2, 5).map(v => Math.round(v)));
    // Harmony matrix: smooth transitions between random context rows (0-7)
    nc.harmony_matrix_contour = smoothRandom(0, 8).map(v => Math.round(v));
    nc.melody_force_contour = Array.from({ length: 16 }, () => smoothRandom(0, 1).map(v => parseFloat(v.toFixed(2))));
    nc.voice_contour = Array.from({ length: 16 }, () => smoothRandom(-12, 12).map(v => Math.round(v)));
    nc.voice_rhythm_contour = Array.from({ length: 16 }, () => smoothRandom(0.25, 4).map(snapNearest));
    // Use the first mode from the contour for Markov progression coherence
    nc.schillinger_sequence = generateMarkovProgression(nc.pl * nc.render_length, sectionModes[0], nc.pl);
    setConfig(nc);
  };

  const handleRandomiseRhythm = () => {
    if (!config) return;
    const steps = Math.ceil((config.pl * 4 * config.render_length) / config.voice_contour_resolution);
    const snaps = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0];
    const snapNearest = (v: number) => snaps.reduce((a, b) => Math.abs(b - v) < Math.abs(a - v) ? b : a);

    // Fractal / 1-over-f rhythm: sum octaves of sine at halving amplitude so the
    // envelope is self-similar — zoom into any phrase and the same pulse shape
    // emerges at a smaller scale. A universe that nests inside itself.
    const octaves = 5;
    const fractalCurve = (phaseOffset: number, seed: number): number[] => {
      return Array.from({ length: steps }, (_, i) => {
        let v = 0;
        let amp = 1;
        for (let o = 0; o < octaves; o++) {
          const freq = Math.pow(2, o);
          const t = (i / steps) * Math.PI * 2 * freq + phaseOffset + seed * (o + 1);
          v += Math.sin(t) * amp;
          amp *= 0.5;
        }
        const norm = (v + 2) / 4;
        return 0.25 + Math.max(0, Math.min(1, norm)) * (4 - 0.25);
      });
    };

    const nc = { ...config };
    nc.voice_rhythm_contour = Array.from({ length: 16 }, (_, v) =>
      fractalCurve(v * 0.618, Math.random() * 6.28).map(snapNearest)
    );
    setConfig(nc);
    setMessage('Rhythm randomised (fractal)');
  };

  // Precompute mode highlights: which modes share the same chord as the current mode at each step
  const modeHighlights = useMemo(() => {
    if (!config || activeTab !== 'mode') return [];
    const modeContour: number[] = config.mode_contour || [];
    const chordContour: number[] = (config.chord_structure_contour && config.chord_structure_contour[selectedVoice]) || [];
    const exContour: number[] = (config.schillinger_ex_contour && config.schillinger_ex_contour[selectedVoice]) || [];
    const seq: number[] = config.schillinger_sequence || [0];
    const res = config.voice_contour_resolution;
    const fallbackChord = config.chord_structure || [0,1,2,4,5];
    const steps = Math.ceil((config.pl * 4 * config.render_length) / res);

    const highlights: Array<{x: number, y: number, color: string}> = [];
    for (let xi = 0; xi < Math.min(steps, modeContour.length); xi++) {
      const barIdx = Math.floor((xi * res) / 4);
      const seqRoot = seq[barIdx % seq.length] ?? 0;
      const expansion = Math.round(exContour[xi] ?? 2);
      const csIdx = Math.round(chordContour[xi] ?? 0);
      const chordStruct = (csIdx >= 0 && csIdx < CHORD_LIST.length) ? CHORD_LIST[csIdx] : fallbackChord;
      const currentMode = ((Math.round(modeContour[xi] ?? config.mode) % 7) + 7) % 7;

      const currentScale = generateModeFromSteps(currentMode);
      const chordIndices = chordStruct.map((item: number) => (item * expansion) + seqRoot);
      const modShim = (v: number, len: number) => ((v % len) + len) % len;
      const currentChord = chordIndices.map((idx: number) => currentScale[modShim(idx, currentScale.length)] % 12);

      for (let m = 0; m < 7; m++) {
        if (m === currentMode) continue;
        const otherScale = generateModeFromSteps(m);
        const otherChord = chordIndices.map((idx: number) => otherScale[modShim(idx, otherScale.length)] % 12);
        if (sameChord(currentChord, otherChord)) {
          highlights.push({ x: xi, y: m, color: 'rgba(253, 224, 71, 0.3)' });
        }
      }
    }
    return highlights;
  }, [config, activeTab, selectedVoice]);

  if (!config) return <div className="p-8 text-xl text-slate-400">Loading Configuration...</div>;

  const renderActiveEditor = () => {
    const xMax = config.pl * 4 * config.render_length;
    
    switch (activeTab) {
      case 'harmony':
        return <ContourEditor
          label="Harmony Distance Contour"
          data={config.harmony_distance_contour || []}
          yMin={-0.2} yMax={0.5} xMax={xMax}
          resolution={config.voice_contour_resolution}
          pl={config.pl}
          onChange={(d) => updateConfig('harmony_distance_contour', d)}
          onResolutionChange={handleResolutionChange}
          color="#c084fc"
        />;
      case 'mode':
        return <ContourEditor
          label="Mode Contour"
          data={config.mode_contour || []}
          yMin={0} yMax={6} xMax={xMax}
          resolution={config.voice_contour_resolution}
          pl={config.pl}
          onChange={(d) => updateConfig('mode_contour', d)}
          onResolutionChange={handleResolutionChange}
          yLabelFormatter={(v) => MODE_NAMES[v] ?? String(v)}
          cellHighlights={modeHighlights}
          color="#fde047"
        />;
      case 'chord':
        const chordData = config.chord_structure_contour ? config.chord_structure_contour[selectedVoice] : [];
        return <ContourEditor
          label={`Voice ${selectedVoice} Chord Structure Contour`}
          data={chordData || []}
          yMin={0} yMax={5} xMax={xMax}
          resolution={config.voice_contour_resolution}
          pl={config.pl}
          onChange={(d) => updateConfig('chord_structure_contour', writeVoiceContour(config.chord_structure_contour, d))}
          onResolutionChange={handleResolutionChange}
          color="#86efac"
        />;
      case 'voice':
        const voiceData = config.voice_contour ? config.voice_contour[selectedVoice] : [];
        return <ContourEditor
          label={`Voice ${selectedVoice} Pitch Shift Contour`}
          data={voiceData}
          yMin={-12} yMax={12} xMax={xMax}
          resolution={config.voice_contour_resolution}
          pl={config.pl}
          onChange={(d) => updateConfig('voice_contour', writeVoiceContour(config.voice_contour, d.map(n => Math.round(n))))}
          onResolutionChange={handleResolutionChange}
          color="#67e8f9"
        />;
      case 'rhythm':
        const rhythmData = config.voice_rhythm_contour ? config.voice_rhythm_contour[selectedVoice] : [];
        return <ContourEditor
          label={`Voice ${selectedVoice} Rhythm Contour`}
          data={rhythmData}
          yMin={0} yMax={4} xMax={xMax}
          resolution={config.voice_contour_resolution}
          pl={config.pl}
          snaps={[0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0]}
          onChange={(d) => updateConfig('voice_rhythm_contour', writeVoiceContour(config.voice_rhythm_contour, d))}
          onResolutionChange={handleResolutionChange}
          color="#f87171"
        />;
      case 'schillinger':
        return <ContourEditor
          label="Schillinger Sequence (1 Block = 1 Bar)"
          data={config.schillinger_sequence || []}
          yMin={0} yMax={6} xMax={xMax}
          resolution={4.0}
          pl={1}
          onChange={(d) => updateConfig('schillinger_sequence', d.map(n => Math.round(n)))}
          color="#fb923c"
          yLabelOffset={1}
        />;
      case 'schillinger_ex':
        const exData = config.schillinger_ex_contour ? config.schillinger_ex_contour[selectedVoice] : [];
        return <ContourEditor
          label={`Voice ${selectedVoice} Schillinger Expansion Contour (ex)`}
          data={exData || []}
          yMin={2} yMax={5} xMax={xMax}
          resolution={config.voice_contour_resolution}
          pl={config.pl}
          onResolutionChange={handleResolutionChange}
          onChange={(d) => updateConfig('schillinger_ex_contour', writeVoiceContour(config.schillinger_ex_contour, d.map(n => Math.round(n))))}
          color="#f43f5e"
        />;
      case 'harmony_matrix':
        return <ContourEditor
          label="Harmony Matrix — 0:Classical 1:Jazz 2:Tension 3:Ethereal 4:Dark 5:Bright 6:Aggressive 7:Ancient 8:Neutral"
          data={config.harmony_matrix_contour || []}
          yMin={0} yMax={8} xMax={xMax}
          resolution={config.voice_contour_resolution}
          pl={config.pl}
          onResolutionChange={handleResolutionChange}
          onChange={(d) => updateConfig('harmony_matrix_contour', d)}
          color="#a78bfa"
        />;
      case 'melody_force':
        const melodyForceData = config.melody_force_contour ? config.melody_force_contour[selectedVoice] : [];
        return <ContourEditor
          label={`Voice ${selectedVoice} Melody Force Contour — line-shaping pressure per step (0 = off … 1 = max). Overrides the Melody Force number when drawn.`}
          data={melodyForceData || []}
          yMin={0} yMax={1} xMax={xMax}
          yStep={0.01}
          resolution={config.voice_contour_resolution}
          pl={config.pl}
          onResolutionChange={handleResolutionChange}
          onChange={(d) => updateConfig('melody_force_contour', writeVoiceContour(config.melody_force_contour, d))}
          color="#34d399"
        />;
      default: return null;
    }
  };

  // Which candidate generator feeds the search — drives which controls and
  // tabs are shown (root-aware terms need the Schillinger progression's roots).
  const sch: boolean = config.schillinger_progression ?? true;

  const tabs = [
    { id: 'schillinger', label: 'Schillinger', tip: 'Chord progression sequence — scale degree indices per bar that define the harmonic framework.' },
    { id: 'schillinger_ex', label: 'Sch. EXP', tip: 'Expansion multiplier — stretches or compresses chord voicing intervals. Higher = wider voicings.' },
    { id: 'harmony', label: 'Harmony', tip: 'Harmony/Smoothness balance over time. High = favor consonance, Low = favor smooth stepwise voice leading.' },
    { id: 'mode', label: 'Mode', tip: 'Scale mode over time (0=Ionian → 6=Lydian). Adjacent values differ by 1 note (circle of fifths order).' },
    { id: 'chord', label: 'Chord', tip: 'Chord voicing structure over time. Selects from preset voicings (triads to full 7-note chords).' },
    { id: 'voice', label: 'Voice Pitch', tip: 'Per-voice pitch offset in semitones over time. Shifts the target pitch the harmonizer aims for.' },
    { id: 'rhythm', label: 'Voice Rhythm', tip: 'Per-voice note duration over time. Values in beats (0.25=16th, 0.5=8th, 1=quarter, 4=whole). Clamped at bar boundaries.' },
    { id: 'harmony_matrix', label: 'H. Matrix', tip: 'Harmony style profile over time. 0=Classical, 1=Jazz, 2=Tension, 3=Ethereal, 4=Dark, 5=Bright, 6=Aggressive, 7=Ancient, 8=Neutral. Fractional values interpolate between rows.' },
    { id: 'melody_force', label: 'Melody Force', tip: 'Line-shaping pressure over time, applied to every voice. Penalizes recently-used pitches and rewards stepwise motion. 0=off, 2-3=strongly forces moving lines. Overrides the Melody Force number when drawn.' },
  ];

  return (
    <div className="h-screen w-screen bg-slate-950 text-slate-200 flex flex-col font-sans overflow-hidden selection:bg-cyan-900">
      {/* Top Header */}
      <header className="flex flex-col gap-3 p-4 bg-slate-900/80 border-b border-slate-800 shrink-0 shadow-lg">
        <div className="flex items-center gap-3 flex-wrap">
          {message && (
            <span className="text-cyan-400 text-sm bg-slate-900/50 px-3 py-1 rounded border border-slate-700">
              {message}
            </span>
          )}

          <div className="flex gap-2 items-stretch flex-wrap">
            <ParamGroup label="Algorithm">
              <div className="flex rounded-lg overflow-hidden border border-slate-700">
                <button
                  onClick={() => updateConfig('schillinger_progression', true)}
                  title="Candidates come from the Schillinger progression's per-bar scales — chords have roots, so the root-aware terms (Quality, Root Pos, Root Dbl, chordal-7th tendency) and the progression tabs apply."
                  className={`px-3 py-1 text-xs font-bold transition-colors ${sch ? 'bg-cyan-600 text-slate-950' : 'bg-slate-900 text-slate-400 hover:bg-slate-800'}`}
                >
                  Schillinger
                </button>
                <button
                  onClick={() => updateConfig('schillinger_progression', false)}
                  title="Candidates are any pitch within ± Cand Range semitones of the voice's previous note — no progression layer, no chord roots; the root-aware terms and progression tabs are off."
                  className={`px-3 py-1 text-xs font-bold transition-colors ${!sch ? 'bg-cyan-600 text-slate-950' : 'bg-slate-900 text-slate-400 hover:bg-slate-800'}`}
                >
                  Chromatic
                </button>
              </div>
              {sch ? (
                <>
                  <div className="flex items-center gap-2">
                    <span className="text-xs uppercase tracking-wider text-slate-500" title="Snap each Schillinger scale note UP to the nearest chord_structure pitch class (mod 12). Mutually exclusive with Floor.">Ceil:</span>
                    <input type="checkbox" checked={config.use_ceiling ?? false} onChange={e => updateConfig('use_ceiling', e.target.checked)} className="accent-cyan-500" />
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs uppercase tracking-wider text-slate-500" title="Snap each Schillinger scale note DOWN to the nearest chord_structure pitch class (mod 12). Ignored if Ceil is on.">Floor:</span>
                    <input type="checkbox" checked={config.use_floor ?? false} onChange={e => updateConfig('use_floor', e.target.checked)} className="accent-cyan-500" />
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs uppercase tracking-wider text-slate-500" title="At bar boundaries (first/last bar of each pl-length phrase), restrict candidate notes by channel: channel 4 → root only, channel 0 → third, others → root/third/fifth.">Resolve:</span>
                    <input type="checkbox" checked={config.use_resolve ?? false} onChange={e => updateConfig('use_resolve', e.target.checked)} className="accent-cyan-500" />
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs uppercase tracking-wider text-slate-500" title="Generate the chord progression from the mode's own chord-transition table instead of using the Schillinger Sequence literally: one phrase of pl bars per render_length, each closing V → I. Overrides Schillinger Sequence while on.">Cadence:</span>
                    <input type="checkbox" checked={config.use_generated_progression ?? false} onChange={e => updateConfig('use_generated_progression', e.target.checked)} className="accent-cyan-500" />
                  </div>
                </>
              ) : (
                <div className="flex items-center gap-2">
                  <span className="text-xs uppercase tracking-wider text-slate-500" title="Pitch search window: each voice considers its previous pitch ± this many semitones. Wider = more freedom to escape a register, slower search.">Cand Range:</span>
                  <NumberField integer value={config.candidate_range ?? 3} onChange={v => updateConfig('candidate_range', v)} className="w-12 bg-transparent text-sm focus:text-cyan-400 outline-none" />
                </div>
              )}
            </ParamGroup>

            <ParamGroup label="Structure">
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Phrase Length — number of bars per phrase. Controls harmonic progression length and contour grid spacing.">PL:</span>
                <NumberField integer value={config.pl} onChange={v => updateConfig('pl', v)} className="w-12 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Render Length — total number of phrases to generate. Total bars = PL × Render Len.">Render Len:</span>
                <NumberField integer value={config.render_length} onChange={v => updateConfig('render_length', v)} className="w-12 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Main Pitch — MIDI note offset added to all output pitches. 60 = Middle C.">Pitch:</span>
                <NumberField integer value={config.main_pitch ?? 60} onChange={v => updateConfig('main_pitch', v)} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Root — pitch class (0-11) used as the tonal center: the Schillinger scale root, and the tonic the tendency-tone term resolves to. 0 = C.">Root:</span>
                <NumberField integer value={config.root ?? 0} onChange={v => updateConfig('root', v)} className="w-12 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Random seed — same seed produces identical output. Change for a different arrangement.">Seed:</span>
                <NumberField integer value={config.rng_seed} onChange={v => updateConfig('rng_seed', v)} className="w-24 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
            </ParamGroup>

            <ParamGroup label="Search">
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Beam Width — how many rival progressions are kept alive, and how many alternative voicings each chord branches into. 1 = greedy (fastest, and Lookahead does nothing). Cost grows as Beam² × (1 + Lookahead).">Beam:</span>
                <NumberField integer value={config.beam_width ?? 3} onChange={v => updateConfig('beam_width', v)} className="w-12 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Lookahead Depth — how many future chords each candidate progression is scored on. Only has an effect when Beam > 1.">Lookahead:</span>
                <NumberField integer value={config.lookahead_depth ?? 0} onChange={v => updateConfig('lookahead_depth', v)} className="w-12 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
            </ParamGroup>

            <ParamGroup label="Voice Leading">
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Penalizes pitches already in voice history. Higher = more variety.">Note Repeat:</span>
                <NumberField value={config.last_note_exist_in_voice ?? 100} onChange={v => updateConfig('last_note_exist_in_voice', v)} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Melody force: applied to EVERY voice (Note Repeat / Same Note only act on the per-chord leader). Penalizes pitches the voice used in its last 5 notes, recency-decayed — so A-B-A-B circling is caught, not just immediate repeats — and slightly rewards stepwise motion (1-2 st). 0 = off. Start ~1.0, raise to 2-3 to strongly force moving lines. Pair with Hold Bias ≤ 0.">Melody Force:</span>
                <NumberField value={config.melody_force ?? 0} onChange={v => updateConfig('melody_force', v)} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Per-voice: penalizes a voice repeating its own immediate previous note. Higher = more melodic movement within a line. (Scale ≈ ±1 now.)">Same Note:</span>
                <NumberField value={config.last_note_same ?? 0.5} onChange={v => updateConfig('last_note_same', v)} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Stickiness: bonus a NON-leader voice gets for holding its previous pitch (common tone). Higher = voices keep common tones unless moving is clearly more consonant. The per-chord leader is excluded so it stays free to move.">Hold Bias:</span>
                <NumberField value={config.same_note_bonus ?? 2.0} onChange={v => updateConfig('same_note_bonus', v)} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Voice-change budget: MINIMUM voices that must change pitch between consecutive chords (-1 = off). Forces the most-worthwhile holders to move, so chords are never fully static.">Min Δ:</span>
                <NumberField integer value={config.min_voices_changed ?? -1} onChange={v => updateConfig('min_voices_changed', v)} className="w-12 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Voice-change budget: MAXIMUM voices that may change pitch between consecutive chords (-1 = off). Holds the least-worthwhile movers on their previous pitch (common tones) for parsimonious voice leading.">Max Δ:</span>
                <NumberField integer value={config.max_voices_changed ?? -1} onChange={v => updateConfig('max_voices_changed', v)} className="w-12 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Penalizes outer voices moving in the same direction as harmony. Encourages contrary motion.">Same Dir:</span>
                <NumberField value={config.same_direction ?? 1} onChange={v => updateConfig('same_direction', v)} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Penalizes parallel fifths and unisons. Classical voice-leading rule. 0 = disabled.">Par 5th/Oct:</span>
                <NumberField value={config.consecutive_octav_fift ?? 0} onChange={v => updateConfig('consecutive_octav_fift', v)} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Prevents voice crossing — voices must stay in their register. Higher = stricter.">No Cross:</span>
                <NumberField value={config.no_crossing ?? 100} onChange={v => updateConfig('no_crossing', v)} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="How strongly each voice's pitch contour pulls its notes toward the target pitch. 1 = original strength, 0 = contour ignored.">Contour Wt:</span>
                <NumberField value={config.voice_contour_weight ?? 1} onChange={v => updateConfig('voice_contour_weight', v)} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
            </ParamGroup>

            <ParamGroup label="Harmony">
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Penalizes duplicate intervals in chord. Adds harmonic variety.">Dup Interval:</span>
                <NumberField value={config.interval_exists_in_harmony ?? 1} onChange={v => updateConfig('interval_exists_in_harmony', v)} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Blend between the H-Matrix style preference and register-aware sensory roughness (harmonic-spectrum Plomp-Levelt). 0 = pure style/pitch-class, 1 = pure psychoacoustics. Roughness is what makes the same interval muddier in the bass and a m2 harsher than a m9.">Roughness:</span>
                <NumberField value={config.roughness_weight ?? 0.5} onChange={v => updateConfig('roughness_weight', v)} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Tendency tones: rewards the leading tone stepping up to the tonic and (Schillinger mode) a chordal minor 7th falling by step, penalizes abandoning either. Only fires in modes that actually have a leading tone. This is what gives cadences their pull. 0 = off.">Tendency:</span>
                <NumberField value={config.tendency_weight ?? 0.5} onChange={v => updateConfig('tendency_weight', v)} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              {sch && (
                <>
                  <div className="flex items-center gap-2">
                    <span className="text-xs uppercase tracking-wider text-slate-500" title="Chord quality: scores each pitch class by its interval ABOVE THE CHORD ROOT, using the same H-Matrix row. This is the only term that can tell a major triad from a minor one — pairwise intervals give {3,4,7} for both. Raise it to make the H-Matrix row steer chord colour, not just interval colour. 0 = off.">Quality:</span>
                    <NumberField value={config.chord_quality_weight ?? 1} onChange={v => updateConfig('chord_quality_weight', v)} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs uppercase tracking-wider text-slate-500" title="Bass on the chord root: full bonus in root position, 0.4 with the third in the bass, 0 with the fifth (six-four), -0.3 for a non-chord tone. Raise for solid functional harmony, drop to 0 to let inversions float freely.">Root Pos:</span>
                    <NumberField value={config.root_position_weight ?? 0.5} onChange={v => updateConfig('root_position_weight', v)} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs uppercase tracking-wider text-slate-500" title="Rewards doubling the chord root (the first doubling only) and penalizes each extra voice on the key's leading tone. The classical doubling policy — pair with Dup Interval, which pushes the other way. 0 = off.">Root Dbl:</span>
                    <NumberField value={config.root_doubling_weight ?? 0.5} onChange={v => updateConfig('root_doubling_weight', v)} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
                  </div>
                </>
              )}
            </ParamGroup>

            <ParamGroup label="Lead Clip">
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Use the pitches from an Ableton clip as the leading voice (channel 0). Pitches are cycled through voice_rhythm timing.">On:</span>
                <input type="checkbox" checked={config.use_leading_voice ?? false} onChange={e => updateConfig('use_leading_voice', e.target.checked)} className="accent-cyan-500" />
                <NumberField integer value={config.leading_voice_track ?? 0} onChange={v => updateConfig('leading_voice_track', v)} title="Ableton track index" className="w-12 bg-transparent text-sm focus:text-cyan-400 outline-none" disabled={!config.use_leading_voice} />
                <span className="text-xs text-slate-600">/</span>
                <NumberField integer value={config.leading_voice_clip ?? 1} onChange={v => updateConfig('leading_voice_clip', v)} title="Ableton clip index" className="w-12 bg-transparent text-sm focus:text-cyan-400 outline-none" disabled={!config.use_leading_voice} />
              </div>
            </ParamGroup>
          </div>

          <div className="flex gap-2 items-center ml-auto">
            <button
              onClick={toggleGenerators}
              title="Presets and randomisers — Jazz/Classical/Ambient/Trance, Init, Randomise, Rhythm"
              className={`px-4 py-2 font-bold rounded-lg shadow transition-all active:scale-95 border ${showGenerators ? 'bg-slate-800 border-cyan-500/50 text-cyan-300' : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'}`}
            >
              {showGenerators ? '▾' : '▸'} Generators
            </button>
            <button
              onClick={handleDuplicate}
              title="Duplicate all contours to double the length [D]"
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-slate-100 font-bold rounded-lg shadow transition-all active:scale-95"
            >
              Duplicate
            </button>
            <button
              onClick={() => setShowMatrix(true)}
              title="Edit the harmony scoring matrix — consonance weight per interval for each style row"
              className="px-4 py-2 bg-violet-700 hover:bg-violet-600 text-slate-100 font-bold rounded-lg shadow transition-all active:scale-95"
            >
              H. Matrix
            </button>
            <button
              onClick={() => setShowChords(true)}
              title="Restrict which chord structures may be built — the constraint the H-Matrix cannot express, since it scores voice pairs rather than the whole sonority"
              className="px-4 py-2 bg-fuchsia-700 hover:bg-fuchsia-600 text-slate-100 font-bold rounded-lg shadow transition-all active:scale-95"
            >
              Chords
              {chordTemplates().length > 0 && (
                <span className="ml-2 px-1.5 py-0.5 rounded bg-fuchsia-900 text-fuchsia-200 text-xs font-mono">
                  {chordTemplates().length}
                </span>
              )}
            </button>
            <button
              onClick={() => setShowStartNotes(true)}
              title="Set the 5 starting seed notes for the generated voices, or fetch them from the last chord of an Ableton clip"
              className="px-4 py-2 bg-teal-700 hover:bg-teal-600 text-slate-100 font-bold rounded-lg shadow transition-all active:scale-95"
            >
              Start Notes
            </button>
            <div className="relative" ref={snapRef}>
              <button
                onClick={() => setShowSnapshots(!showSnapshots)}
                title="Snapshots — saved on each Generate"
                className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 font-bold rounded-lg shadow transition-all active:scale-95"
              >
                Snapshots{snapshots.length > 0 && ` (${snapshots.length})`}
              </button>
              {showSnapshots && (() => {
                const visible = favoritesOnly ? snapshots.filter(s => s.favorite) : snapshots;
                const commitRename = (id: string) => {
                  const trimmed = renameValue.trim();
                  if (trimmed) {
                    renameSnapshot(id, trimmed);
                    setSnapshots(listSnapshots());
                  }
                  setRenamingId(null);
                };
                return (
                <div className="absolute right-0 top-full mt-1 w-80 max-h-80 overflow-y-auto bg-slate-900 border border-slate-700 rounded-lg shadow-xl z-50">
                  <div className="flex items-center justify-between px-3 py-2 border-b border-slate-800 sticky top-0 bg-slate-900">
                    <label className="flex items-center gap-2 text-xs text-slate-400 cursor-pointer select-none">
                      <input
                        type="checkbox"
                        checked={favoritesOnly}
                        onChange={(e) => setFavoritesOnly(e.target.checked)}
                        className="accent-yellow-400"
                      />
                      Favorites only
                    </label>
                    <span className="text-xs text-slate-600">{visible.length} / {snapshots.length}</span>
                  </div>
                  {visible.length === 0 ? (
                    <div className="p-3 text-sm text-slate-500">
                      {snapshots.length === 0
                        ? 'No snapshots yet. Press Generate to create one.'
                        : 'No favorites yet. Click the star to favorite a snapshot.'}
                    </div>
                  ) : visible.map(s => (
                    <div key={s.id} className="flex items-center gap-2 px-3 py-2 hover:bg-slate-800 border-b border-slate-800 last:border-0">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleFavorite(s.id);
                          setSnapshots(listSnapshots());
                        }}
                        className={`text-base shrink-0 ${s.favorite ? 'text-yellow-400' : 'text-slate-600 hover:text-yellow-400'}`}
                        title={s.favorite ? 'Unfavorite' : 'Favorite'}
                      >
                        {s.favorite ? '★' : '☆'}
                      </button>
                      {renamingId === s.id ? (
                        <input
                          autoFocus
                          value={renameValue}
                          onChange={(e) => setRenameValue(e.target.value)}
                          onBlur={() => commitRename(s.id)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') commitRename(s.id);
                            else if (e.key === 'Escape') setRenamingId(null);
                          }}
                          className="flex-1 min-w-0 bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm text-slate-200 focus:outline-none focus:border-cyan-500"
                        />
                      ) : (
                        <button
                          onClick={() => {
                            const cfg = loadSnapshot(s.id);
                            if (cfg) { setConfig(cfg); setMessage(`Loaded snapshot ${s.name}`); setShowSnapshots(false); }
                          }}
                          onDoubleClick={(e) => {
                            e.stopPropagation();
                            setRenameValue(s.name);
                            setRenamingId(s.id);
                          }}
                          className="text-sm text-slate-300 hover:text-cyan-400 truncate text-left flex-1 min-w-0"
                          title="Click to load · Double-click to rename"
                        >
                          {s.name}
                        </button>
                      )}
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setRenameValue(s.name);
                          setRenamingId(s.id);
                        }}
                        className="text-xs text-slate-600 hover:text-cyan-400 shrink-0"
                        title="Rename snapshot"
                      >
                        ✎
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteSnapshot(s.id);
                          setSnapshots(listSnapshots());
                        }}
                        className="text-xs text-slate-600 hover:text-red-400 shrink-0"
                        title="Delete snapshot"
                      >
                        x
                      </button>
                    </div>
                  ))}
                </div>
                );
              })()}
            </div>
            <button
              onClick={handleGenerate}
              disabled={isGenerating}
              title="Generate MIDI output [G]"
              className="px-6 py-2 bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold rounded-lg shadow-[0_0_15px_rgba(34,211,238,0.4)] transition-all active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isGenerating ? 'Generating...' : 'Generate'}
            </button>
            <button
              onClick={() => setShowInspector(true)}
              disabled={breakdown.length === 0}
              title={breakdown.length > 0
                ? 'Inspect the last render: named score contributions per chosen chord'
                : 'Generate first — the inspector shows the last render\'s score breakdown'}
              className="px-4 py-2 bg-emerald-700 hover:bg-emerald-600 text-slate-100 font-bold rounded-lg shadow transition-all active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Why?
            </button>
          </div>
        </div>

        {/* Generators — presets & randomisers, folded away by default */}
        {showGenerators && (
          <div className="flex gap-2 items-center flex-wrap bg-slate-950/60 border border-slate-800 rounded-lg px-3 py-2">
            <span className="text-[10px] uppercase tracking-widest text-slate-600 select-none mr-1">Presets</span>
            <button
              onClick={() => applyPreset('jazz')}
              title="Jazz preset — Dorian/Mixolydian, ii-V-I turnarounds, syncopated rhythms, relaxed voice-leading"
              className="px-4 py-1.5 bg-yellow-700 hover:bg-yellow-600 text-slate-100 font-bold rounded-lg shadow transition-all active:scale-95"
            >
              Jazz
            </button>
            <button
              onClick={() => applyPreset('classical')}
              title="Classical preset — Ionian/Aeolian, I-IV-V-I, strict counterpoint rules, arch-form dynamics"
              className="px-4 py-1.5 bg-rose-800 hover:bg-rose-700 text-slate-100 font-bold rounded-lg shadow transition-all active:scale-95"
            >
              Classical
            </button>
            <button
              onClick={() => applyPreset('ambient')}
              title="Ambient preset — Lydian float over the Ethereal matrix, glacial 2-chord oscillations, deep hold bias, one voice drifts at a time"
              className="px-4 py-1.5 bg-teal-800 hover:bg-teal-700 text-slate-100 font-bold rounded-lg shadow transition-all active:scale-95"
            >
              Ambient
            </button>
            <button
              onClick={() => applyPreset('trance')}
              title="Trance preset — Aeolian anthem loops (i-VI-III-VII), pumping 8th-note bass and lead over held pads, Dark→Bright climax"
              className="px-4 py-1.5 bg-violet-800 hover:bg-violet-700 text-slate-100 font-bold rounded-lg shadow transition-all active:scale-95"
            >
              Trance
            </button>
            <div className="border-l border-slate-700 h-5 mx-2"></div>
            <span className="text-[10px] uppercase tracking-widest text-slate-600 select-none mr-1">Reset / Randomise</span>
            <button
              onClick={handleInit}
              title="Reset to the server's default config (reasonable baseline). Does not delete snapshots."
              className="px-4 py-1.5 bg-slate-600 hover:bg-slate-500 text-slate-100 font-bold rounded-lg shadow transition-all active:scale-95"
            >
              Init
            </button>
            <button
              onClick={handleRandomise}
              title="Randomise all contours"
              className="px-4 py-1.5 bg-amber-600 hover:bg-amber-500 text-slate-100 font-bold rounded-lg shadow transition-all active:scale-95"
            >
              Randomise
            </button>
            <button
              onClick={handleRandomiseRhythm}
              title="Randomise only rhythm — fractal / self-similar pulse across octaves"
              className="px-4 py-1.5 bg-pink-600 hover:bg-pink-500 text-slate-100 font-bold rounded-lg shadow transition-all active:scale-95"
            >
              Rhythm
            </button>
          </div>
        )}

        {/* Tabs Row */}
        <div className="flex justify-between items-end">
          <div className="flex gap-2">
            {tabs.filter(t => sch || !SCHILLINGER_TAB_IDS.includes(t.id)).map(t => (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                title={t.tip}
                className={`px-4 py-2 text-sm rounded-lg transition-colors border ${activeTab === t.id ? 'bg-slate-800 border-cyan-500/50 text-cyan-400' : 'bg-transparent border-slate-800 text-slate-400 hover:bg-slate-900 border-dashed'}`}
              >
                {t.label}
              </button>
            ))}
          </div>

          {(activeTab === 'voice' || activeTab === 'rhythm' || activeTab === 'schillinger_ex' || activeTab === 'chord' || activeTab === 'melody_force') && (
            <div className={`flex items-center gap-4 px-3 py-1.5 rounded-lg border transition-colors ${broadcastVoices ? 'bg-amber-950/50 border-amber-500/50' : 'bg-slate-950 border-slate-800'}`} title="Hold Shift while editing to apply to all voices">
              <span className={`text-xs uppercase ${broadcastVoices ? 'text-amber-400' : 'text-slate-500'}`}>
                {broadcastVoices ? 'All Voices (Shift)' : 'Voice Target:'}
              </span>
              <div className="flex gap-1 flex-wrap">
                {Array.from({ length: 16 }).map((_, i) => i < 5 ? (
                  <button
                    key={i}
                    onClick={() => setSelectedVoice(i)}
                    className={`w-7 h-7 rounded text-xs font-mono transition-colors ${broadcastVoices ? 'bg-amber-800/60 text-amber-200' : selectedVoice === i ? 'bg-cyan-600 text-slate-950' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}
                  >
                    {i}
                  </button>
                ) : null)}
              </div>
            </div>
          )}
        </div>
      </header>

      {/* Main Editor Canvas Full Screen */}
      <div className="flex-1 overflow-hidden p-6 bg-slate-950 flex flex-col">
          {renderActiveEditor()}
      </div>

      {/* Script Console */}
      <div className="shrink-0 border-t border-slate-800 bg-slate-900/80">
        <div className="flex items-center justify-between px-4 py-1.5">
          <button
            onClick={() => setShowConsole(s => !s)}
            className="text-xs font-mono uppercase tracking-wide text-slate-400 hover:text-cyan-400 transition-colors"
            title="Run JS against the live config. `config` = full config (mutable), `$` = current tab+voice contour, helpers: range(n), clamp, lerp, steps, voice, res."
          >
            {showConsole ? '▾' : '▸'} ⌨ Script Console
          </button>
          <div className="flex items-center gap-3">
            {scriptMsg && (
              <span className={`text-xs font-mono ${scriptMsg.error ? 'text-rose-400' : 'text-emerald-400'}`}>
                {scriptMsg.text}
              </span>
            )}
            <button
              onClick={() => { setShowConsole(true); setConsoleFullscreen(true); }}
              className="text-xs font-mono text-slate-400 hover:text-cyan-400 transition-colors border border-slate-700 hover:border-cyan-500/50 rounded px-2 py-0.5"
              title="Open the full-screen code editor"
            >
              ⛶ Full screen
            </button>
          </div>
        </div>
        {showConsole && (
          <div className="px-4 pb-3 flex flex-col gap-2">
            <ScriptEditor value={script} onChange={setScript} onRun={() => runScript(script)} height="160px" />
            <div className="flex items-center gap-3">
              <button
                onClick={() => runScript(script)}
                className="px-4 py-1.5 text-sm rounded-lg bg-cyan-600 text-slate-950 font-medium hover:bg-cyan-500 transition-colors"
              >
                Run <span className="opacity-60 text-xs">⌘↵</span>
              </button>
              <span className="text-xs text-slate-500 font-mono">
                Targeting <span className="text-cyan-400">{CONTOUR_FIELDS[activeTab]?.field ?? '—'}</span>
                {CONTOUR_FIELDS[activeTab]?.perVoice && <span className="text-cyan-400">[{selectedVoice}]</span>} for <span className="text-cyan-400">$</span>
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Full-screen Script Editor */}
      {consoleFullscreen && (
        <div className="fixed inset-0 z-[110] bg-slate-950 flex flex-col">
          <div className="flex items-center justify-between gap-4 px-5 py-3 border-b border-slate-800 bg-slate-900 shrink-0">
            <div className="flex items-baseline gap-3">
              <h2 className="text-lg font-bold text-cyan-300">⌨ Script Editor</h2>
              <span className="text-xs text-slate-500 font-mono">
                Targeting <span className="text-cyan-400">{CONTOUR_FIELDS[activeTab]?.field ?? '—'}</span>
                {CONTOUR_FIELDS[activeTab]?.perVoice && <span className="text-cyan-400">[{selectedVoice}]</span>} for <span className="text-cyan-400">$</span>
              </span>
            </div>
            <div className="flex items-center gap-3">
              {scriptMsg && (
                <span className={`text-xs font-mono ${scriptMsg.error ? 'text-rose-400' : 'text-emerald-400'}`}>
                  {scriptMsg.text}
                </span>
              )}
              <button
                onClick={() => runScript(script)}
                className="px-4 py-1.5 text-sm rounded-lg bg-cyan-600 text-slate-950 font-medium hover:bg-cyan-500 transition-colors"
              >
                Run <span className="opacity-60 text-xs">⌘↵</span>
              </button>
              <button
                onClick={() => setConsoleFullscreen(false)}
                className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-200 text-sm font-bold rounded-lg transition-all active:scale-95"
                title="Close (Esc)"
              >
                Close <span className="opacity-60 text-xs">Esc</span>
              </button>
            </div>
          </div>
          <div className="flex-1 min-h-0 overflow-hidden p-4">
            <ScriptEditor value={script} onChange={setScript} onRun={() => runScript(script)} height="100%" autoFocus />
          </div>
          <div className="shrink-0 px-5 py-2 border-t border-slate-800 bg-slate-900 text-xs text-slate-500 font-mono">
            <span className="text-slate-400">config</span> = live config (mutable) &nbsp;·&nbsp; <span className="text-slate-400">$</span> = current tab+voice contour &nbsp;·&nbsp; helpers: <span className="text-slate-400">range(n), clamp(x,lo,hi), lerp(a,b,t), steps, voice, res</span>
          </div>
        </div>
      )}

      {/* Chord Structure Whitelist Modal */}
      {/* "Why this chord" score inspector */}
      {showInspector && breakdown.length > 0 && (
        <ChordInspector breakdown={breakdown} onClose={() => setShowInspector(false)} />
      )}

      {showChords && (
        <div
          className="fixed inset-0 z-[100] bg-black/70 flex items-center justify-center p-6"
          onMouseDown={(e) => { if (e.target === e.currentTarget) setShowChords(false); }}
        >
          <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-[46rem] max-w-[95vw] max-h-[90vh] overflow-auto">
            <div className="flex items-center justify-between gap-4 px-5 py-3 border-b border-slate-800 sticky top-0 bg-slate-900 z-10">
              <div>
                <h2 className="text-lg font-bold text-fuchsia-300">Chord Structures</h2>
                <p className="text-xs text-slate-500 max-w-[34rem]">
                  Only these structures may be built, on any root. Doublings are free — the rule is about which
                  distinct pitch classes appear, not how they are voiced. An empty list leaves chord structure
                  unconstrained. Unlike the H-Matrix, which scores voice <em>pairs</em>, this sees the whole sonority,
                  so it can tell a major triad from an augmented one. Turn on <em>Weight by usage</em> to fix how often
                  each structure appears — 0.75 / 0.25 gives three major chords to every minor one.
                </p>
              </div>
              <div className="flex gap-2 items-center shrink-0">
                <button
                  onClick={clearChordTemplates}
                  title="Remove every restriction — any chord structure allowed"
                  className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-200 text-sm font-bold rounded-lg transition-all active:scale-95"
                >
                  Clear all
                </button>
                <button
                  onClick={() => setShowChords(false)}
                  className="px-3 py-1.5 bg-fuchsia-700 hover:bg-fuchsia-600 text-slate-100 text-sm font-bold rounded-lg transition-all active:scale-95"
                >
                  Done
                </button>
              </div>
            </div>

            <div className="p-5 space-y-6">
              {/* Current list */}
              <div>
                <div className="flex items-baseline justify-between mb-2">
                  <h3 className="text-xs uppercase tracking-wider text-slate-500">
                    Allowed structures ({chordTemplates().length})
                  </h3>
                  {chordTemplates().length > 0 && (
                    <label className="flex items-center gap-2 text-xs text-slate-400 cursor-pointer select-none">
                      <input
                        type="checkbox"
                        checked={weightsActive(chordTemplates())}
                        onChange={toggleChordWeights}
                        className="accent-fuchsia-500"
                      />
                      <span title="Off: any listed structure, whichever scores best each chord. On: each structure gets its share of the chords, spread evenly through the render.">
                        Weight by usage
                      </span>
                    </label>
                  )}
                </div>
                {chordTemplates().length === 0 ? (
                  <p className="text-sm text-slate-500 italic bg-slate-950 border border-slate-800 rounded-lg px-3 py-3">
                    No restriction — the harmonizer may build any chord the H-Matrix and voice leading permit.
                  </p>
                ) : (
                  <ul className="space-y-1.5">
                    {chordTemplates().map((t, i) => {
                      const pcs = normalizePcs(tplPcs(t));
                      return (
                        <li key={i} className="flex items-center gap-3 bg-slate-950 border border-slate-800 rounded-lg px-3 py-2">
                          <span className="text-sm font-bold text-slate-200 w-28 shrink-0">{chordName(tplPcs(t))}</span>
                          <span className="flex gap-1">
                            {pcs.map((pc) => (
                              <span key={pc} className="px-1.5 py-0.5 rounded bg-fuchsia-950 text-fuchsia-300 text-xs font-mono">
                                {CHORD_DEGREE_LABELS[pc]}
                              </span>
                            ))}
                          </span>
                          <span className="text-xs text-slate-600 font-mono">[{pcs.join(', ')}]</span>
                          {pcs.length > 5 && (
                            <span className="text-xs text-amber-400" title="More pitch classes than there are voices, so no voicing can satisfy this entry">
                              needs {pcs.length} voices
                            </span>
                          )}
                          {weightsActive(chordTemplates()) && (
                            <span className="ml-auto flex items-center gap-2 shrink-0">
                              <NumberField
                                value={tplWeight(t)}
                                onChange={(v) => setChordWeight(i, v)}
                                title="Relative weight. Shares are normalised, so 3/1 and 0.75/0.25 mean the same thing. 0 = listed but never used."
                                className="w-16 bg-slate-950 border border-slate-800 rounded px-1.5 py-1 text-sm text-right font-mono text-emerald-300 outline-none focus:border-fuchsia-500"
                              />
                              <span
                                className={`text-xs font-mono w-12 text-right ${tplWeight(t) <= 0 ? 'text-slate-600' : 'text-emerald-400'}`}
                                title="Share of chords this structure gets"
                              >
                                {chordWeightShare(i).toFixed(0)}%
                              </span>
                            </span>
                          )}
                          <button
                            onClick={() => removeChordTemplate(i)}
                            className={`${weightsActive(chordTemplates()) ? '' : 'ml-auto '}px-2 py-1 bg-slate-800 hover:bg-rose-800 text-slate-300 hover:text-rose-100 text-xs font-bold rounded transition-all active:scale-95`}
                          >
                            Remove
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </div>

              {/* Presets */}
              <div>
                <h3 className="text-xs uppercase tracking-wider text-slate-500 mb-2">Add a preset</h3>
                <div className="flex flex-wrap gap-2">
                  {CHORD_PRESETS.map((preset) => {
                    const already = chordTemplates().some((t) => rotationKey(tplPcs(t)) === rotationKey(preset.pcs));
                    return (
                      <button
                        key={preset.name}
                        onClick={() => addChordTemplate(preset.pcs)}
                        disabled={already}
                        title={already ? 'Already in the list' : `Add ${preset.name} — [${preset.pcs.join(', ')}]`}
                        className={`px-3 py-1.5 text-sm font-bold rounded-lg transition-all active:scale-95 ${
                          already
                            ? 'bg-slate-800 text-slate-600 cursor-not-allowed'
                            : 'bg-slate-700 hover:bg-fuchsia-700 text-slate-200'
                        }`}
                      >
                        {preset.name}
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Custom builder */}
              <div>
                <h3 className="text-xs uppercase tracking-wider text-slate-500 mb-2">Build a custom structure</h3>
                <p className="text-xs text-slate-600 mb-2">Pick the semitone offsets from the root.</p>
                <div className="flex flex-wrap gap-1 mb-3">
                  {CHORD_DEGREE_LABELS.map((label, pc) => {
                    const on = chordDraft.includes(pc);
                    return (
                      <button
                        key={pc}
                        onClick={() => toggleChordDraft(pc)}
                        title={`${pc} semitone${pc === 1 ? '' : 's'} above the root`}
                        className={`w-12 py-1.5 text-sm font-mono rounded transition-all active:scale-95 ${
                          on ? 'bg-fuchsia-700 text-slate-100 font-bold' : 'bg-slate-800 hover:bg-slate-700 text-slate-400'
                        }`}
                      >
                        {label}
                      </button>
                    );
                  })}
                </div>
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => { addChordTemplate(chordDraft); }}
                    disabled={chordDraft.length === 0}
                    className={`px-4 py-2 text-sm font-bold rounded-lg transition-all active:scale-95 ${
                      chordDraft.length === 0
                        ? 'bg-slate-800 text-slate-600 cursor-not-allowed'
                        : 'bg-fuchsia-700 hover:bg-fuchsia-600 text-slate-100'
                    }`}
                  >
                    Add to list
                  </button>
                  <span className="text-sm text-slate-500">
                    {chordDraft.length === 0
                      ? 'Nothing selected'
                      : `${chordName(chordDraft)} — [${normalizePcs(chordDraft).join(', ')}]`}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Harmony Scoring Matrix Editor Modal */}
      {showMatrix && (
        <div
          className="fixed inset-0 z-[100] bg-black/70 flex items-center justify-center p-6"
          onMouseDown={(e) => { if (e.target === e.currentTarget) setShowMatrix(false); }}
        >
          <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl max-w-[95vw] max-h-[90vh] overflow-auto">
            <div className="flex items-center justify-between gap-4 px-5 py-3 border-b border-slate-800 sticky top-0 bg-slate-900 z-10">
              <div>
                <h2 className="text-lg font-bold text-violet-300">Harmony Scoring Matrix</h2>
                <p className="text-xs text-slate-500">Per-style consonance preference per interval. Soft values live in ≈ −1…+1 (higher = more favoured). Any cell ≤ −5 is a hard "forbidden" constraint (e.g. the default −100s).</p>
              </div>
              <div className="flex gap-2 items-center">
                <button
                  onClick={resetMatrix}
                  title="Restore all values to the built-in defaults"
                  className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-200 text-sm font-bold rounded-lg transition-all active:scale-95"
                >
                  Reset to default
                </button>
                <button
                  onClick={() => setShowMatrix(false)}
                  className="px-3 py-1.5 bg-violet-700 hover:bg-violet-600 text-slate-100 text-sm font-bold rounded-lg transition-all active:scale-95"
                >
                  Done
                </button>
              </div>
            </div>
            <div className="p-5">
              <table className="border-collapse">
                <thead>
                  <tr>
                    <th className="sticky left-0 bg-slate-900 px-2 py-1 text-left text-xs uppercase tracking-wider text-slate-500">Style \ Interval</th>
                    {HARMONY_MATRIX_COLS.map((c, ci) => (
                      <th key={ci} className="px-1 py-1 text-center text-xs font-mono text-slate-400" title={`Interval ${ci} semitones`}>{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {HARMONY_MATRIX_ROWS.map((rowName, ri) => (
                    <tr key={ri} className="hover:bg-slate-800/40">
                      <td className="sticky left-0 bg-slate-900 px-2 py-1 text-sm text-slate-300 whitespace-nowrap">
                        <span className="text-slate-600 font-mono mr-1">{ri}</span>{rowName}
                      </td>
                      {HARMONY_MATRIX_COLS.map((_, ci) => {
                        const val = getMatrix()[ri][ci];
                        return (
                          <td key={ci} className="px-0.5 py-0.5">
                            <NumberField
                              value={val}
                              onChange={(v) => updateMatrixCell(ri, ci, v)}
                              className={`w-16 bg-slate-950 border border-slate-800 rounded px-1.5 py-1 text-sm text-right font-mono outline-none focus:border-violet-500 ${val <= -10 ? 'text-red-400' : val < 0 ? 'text-rose-300' : val > 0 ? 'text-emerald-300' : 'text-slate-500'}`}
                            />
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Start Notes Editor Modal */}
      {showStartNotes && (
        <div
          className="fixed inset-0 z-[100] bg-black/70 flex items-center justify-center p-6"
          onMouseDown={(e) => { if (e.target === e.currentTarget) setShowStartNotes(false); }}
        >
          <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-[440px] max-w-[95vw] max-h-[90vh] overflow-auto">
            <div className="flex items-center justify-between gap-4 px-5 py-3 border-b border-slate-800">
              <div>
                <h2 className="text-lg font-bold text-teal-300">Start Notes</h2>
                <p className="text-xs text-slate-500">Seed pitch for each of the 5 voices (high → low). Voice 0 is the leading voice.</p>
              </div>
              <div className="flex gap-2 items-center">
                <button
                  onClick={resetStartNotes}
                  title="Restore the default seed pitches (70 65 60 50 34)"
                  className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-200 text-sm font-bold rounded-lg transition-all active:scale-95"
                >
                  Reset
                </button>
                <button
                  onClick={() => setShowStartNotes(false)}
                  className="px-3 py-1.5 bg-teal-700 hover:bg-teal-600 text-slate-100 text-sm font-bold rounded-lg transition-all active:scale-95"
                >
                  Done
                </button>
              </div>
            </div>
            <div className="p-5 space-y-4">
              <div className="space-y-2">
                {getStartNotes().map((n, i) => (
                  <div key={i} className="flex items-center gap-3">
                    <span className="text-xs uppercase tracking-wider text-slate-500 w-16">
                      Voice {i}{i === 0 ? ' *' : ''}
                    </span>
                    <NumberField
                      integer
                      value={n}
                      onChange={(v) => updateStartNote(i, v)}
                      className="w-20 bg-slate-950 border border-slate-800 rounded px-2 py-1 text-sm text-right font-mono text-teal-300 outline-none focus:border-teal-500"
                    />
                    <span className="text-sm font-mono text-slate-400 w-12">{midiName(n)}</span>
                  </div>
                ))}
              </div>

              <div className="border-t border-slate-800 pt-4">
                <p className="text-xs uppercase tracking-wider text-slate-500 mb-2">Fetch last chord from Ableton clip</p>
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-xs text-slate-500">Track</span>
                  <NumberField
                    integer
                    value={snTrack}
                    onChange={setSnTrack}
                    title="Ableton track index"
                    className="w-14 bg-slate-950 border border-slate-800 rounded px-2 py-1 text-sm text-right font-mono outline-none focus:border-teal-500"
                  />
                  <span className="text-xs text-slate-500">Clip</span>
                  <NumberField
                    integer
                    value={snClip}
                    onChange={setSnClip}
                    title="Ableton clip slot index"
                    className="w-14 bg-slate-950 border border-slate-800 rounded px-2 py-1 text-sm text-right font-mono outline-none focus:border-teal-500"
                  />
                  <button
                    onClick={fetchLastChord}
                    disabled={fetchingChord}
                    title="Read the clip, take the pitches of its final chord (high → low), and load them into the slots above"
                    className="px-3 py-1.5 bg-cyan-700 hover:bg-cyan-600 disabled:opacity-50 text-slate-100 text-sm font-bold rounded-lg transition-all active:scale-95"
                  >
                    {fetchingChord ? 'Fetching…' : 'Fetch last chord'}
                  </button>
                </div>
                {chordMsg && (
                  <p className={`mt-2 text-xs font-mono ${chordMsg.error ? 'text-rose-400' : 'text-emerald-400'}`}>
                    {chordMsg.text}
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
