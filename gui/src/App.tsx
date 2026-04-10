import { useState, useEffect } from 'react'
import { ContourEditor } from './ContourEditor'
import './index.css'

function App() {
  const [config, setConfig] = useState<any>(null);
  const [activeTab, setActiveTab] = useState('harmony');
  const [selectedVoice, setSelectedVoice] = useState(0);
  const [isGenerating, setIsGenerating] = useState(false);
  const [message, setMessage] = useState('');

  useEffect(() => {
    fetch('http://127.0.0.1:3000/api/config')
      .then(res => res.json())
      .then(data => setConfig(data))
      .catch(err => setMessage(`Error loading config: ${err}`));
  }, []);

  const handleGenerate = async () => {
    if (!config) return;
    setIsGenerating(true);
    setMessage('Generating MIDI...');
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
    } catch (err: any) {
      setMessage(`Error: ${err.message || err}`);
    }
    setIsGenerating(false);
  };

  const updateConfig = (key: string, value: any) => {
    setConfig({ ...config, [key]: value });
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
    if (newConfig.chord_structure_contour) newConfig.chord_structure_contour = duplicateArray(newConfig.chord_structure_contour, stdSteps, 0);
    if (newConfig.schillinger_ex_contour) newConfig.schillinger_ex_contour = duplicateArray(newConfig.schillinger_ex_contour, stdSteps, 2);
    
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
    if (nc.chord_structure_contour) nc.chord_structure_contour = resampleContour(nc.chord_structure_contour, oldRes, newRes);
    if (nc.schillinger_ex_contour) nc.schillinger_ex_contour = resampleContour(nc.schillinger_ex_contour, oldRes, newRes);
    if (nc.voice_contour) nc.voice_contour = nc.voice_contour.map((t: number[]) => resampleContour(t, oldRes, newRes));
    if (nc.voice_rhythm_contour) nc.voice_rhythm_contour = nc.voice_rhythm_contour.map((t: number[]) => resampleContour(t, oldRes, newRes));
    setConfig(nc);
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
    nc.harmony_distance_contour = smoothRandom(-0.2, 0.5).map(v => parseFloat(v.toFixed(2)));
    nc.mode_contour = smoothRandom(0, 6).map(v => Math.round(v));
    nc.chord_structure_contour = smoothRandom(0, 5).map(v => Math.round(v));
    nc.schillinger_ex_contour = smoothRandom(2, 5).map(v => Math.round(v));
    nc.voice_contour = Array.from({ length: 16 }, () => smoothRandom(-12, 12).map(v => parseFloat(v.toFixed(1))));
    nc.voice_rhythm_contour = Array.from({ length: 16 }, () => smoothRandom(0.25, 4).map(snapNearest));
    setConfig(nc);
  };

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
          yMin={0} yMax={7} xMax={xMax}
          resolution={config.voice_contour_resolution}
          pl={config.pl}
          onChange={(d) => updateConfig('mode_contour', d)}
          onResolutionChange={handleResolutionChange}
          color="#fde047"
        />;
      case 'chord':
        return <ContourEditor
          label="Chord Structure Contour"
          data={config.chord_structure_contour || []}
          yMin={0} yMax={5} xMax={xMax}
          resolution={config.voice_contour_resolution}
          pl={config.pl}
          onChange={(d) => updateConfig('chord_structure_contour', d)}
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
          onChange={(d) => {
            const nv = [...config.voice_contour];
            nv[selectedVoice] = d;
            updateConfig('voice_contour', nv);
          }}
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
          onChange={(d) => {
            const nv = [...config.voice_rhythm_contour];
            nv[selectedVoice] = d;
            updateConfig('voice_rhythm_contour', nv);
          }}
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
        />;
      case 'schillinger_ex':
        return <ContourEditor
          label="Schillinger Expansion Contour (ex)"
          data={config.schillinger_ex_contour || []}
          yMin={2} yMax={5} xMax={xMax}
          resolution={config.voice_contour_resolution}
          pl={config.pl}
          onResolutionChange={handleResolutionChange}
          onChange={(d) => updateConfig('schillinger_ex_contour', d.map(n => Math.round(n)))}
          color="#f43f5e"
        />;
      default: return null;
    }
  };

  const tabs = [
    { id: 'schillinger', label: 'Schillinger', tip: 'Chord progression sequence — scale degree indices per bar that define the harmonic framework.' },
    { id: 'schillinger_ex', label: 'Sch. EXP', tip: 'Expansion multiplier — stretches or compresses chord voicing intervals. Higher = wider voicings.' },
    { id: 'harmony', label: 'Harmony', tip: 'Harmony/Smoothness balance over time. High = favor consonance, Low = favor smooth stepwise voice leading.' },
    { id: 'mode', label: 'Mode', tip: 'Scale mode over time (0=Ionian → 6=Lydian). Adjacent values differ by 1 note (circle of fifths order).' },
    { id: 'chord', label: 'Chord', tip: 'Chord voicing structure over time. Selects from preset voicings (triads to full 7-note chords).' },
    { id: 'voice', label: 'Voice Pitch', tip: 'Per-voice pitch offset in semitones over time. Shifts the target pitch the harmonizer aims for.' },
    { id: 'rhythm', label: 'Voice Rhythm', tip: 'Per-voice note duration over time. Values in beats (0.25=16th, 0.5=8th, 1=quarter, 4=whole). Clamped at bar boundaries.' },
  ];

  return (
    <div className="h-screen w-screen bg-slate-950 text-slate-200 flex flex-col font-sans overflow-hidden selection:bg-cyan-900">
      {/* Top Header */}
      <header className="flex flex-col gap-4 p-4 bg-slate-900/80 border-b border-slate-800 shrink-0 shadow-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-6">
            <h1 className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-indigo-400 bg-clip-text text-transparent">Rust Harmoniser</h1>
            
            {message && (
              <span className="text-cyan-400 text-sm bg-slate-900/50 px-3 py-1 rounded border border-slate-700">
                {message}
              </span>
            )}
          </div>
          
            <div className="flex gap-6 items-center bg-slate-950 px-4 py-2 rounded-lg border border-slate-800">
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Phrase Length — number of bars per phrase. Controls harmonic progression length and contour grid spacing.">PL:</span>
                <input type="number" value={config.pl} onChange={e => updateConfig('pl', parseInt(e.target.value))} className="w-12 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Render Length — total number of phrases to generate. Total bars = PL × Render Len.">Render Len:</span>
                <input type="number" value={config.render_length} onChange={e => updateConfig('render_length', parseInt(e.target.value))} className="w-12 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Lookahead Depth — how many future steps the harmonizer evaluates. Higher = more coherent voice leading but slower.">Lookahead:</span>
                <input type="number" value={config.lookahead_depth ?? 0} onChange={e => updateConfig('lookahead_depth', parseInt(e.target.value))} className="w-12 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="When enabled, the last note of the piece resolves back to match the first note.">Same End:</span>
                <input type="checkbox" checked={config.last_note_same > 0} onChange={e => updateConfig('last_note_same', e.target.checked ? 10.0 : 0.0)} className="accent-cyan-500" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Random seed — same seed produces identical output. Change for a different arrangement.">Seed:</span>
                <input type="number" value={config.rng_seed} onChange={e => updateConfig('rng_seed', parseInt(e.target.value))} className="w-24 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
            </div>

            <div className="flex gap-4 items-center flex-wrap bg-slate-950 px-4 py-2 rounded-lg border border-slate-800">
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Penalizes pitches already in voice history">Note Repeat:</span>
                <input type="number" step="1" value={config.last_note_exist_in_voice ?? 100} onChange={e => updateConfig('last_note_exist_in_voice', parseFloat(e.target.value))} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Penalizes same melodic direction as harmony">Same Dir:</span>
                <input type="number" step="0.1" value={config.same_direction ?? 1} onChange={e => updateConfig('same_direction', parseFloat(e.target.value))} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Penalizes parallel fifths and unisons">Par 5th/Oct:</span>
                <input type="number" step="0.1" value={config.consecutive_octav_fift ?? 0} onChange={e => updateConfig('consecutive_octav_fift', parseFloat(e.target.value))} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Prevents voice crossing between registers">No Cross:</span>
                <input type="number" step="1" value={config.no_crossing ?? 100} onChange={e => updateConfig('no_crossing', parseFloat(e.target.value))} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Penalizes repeating the immediate previous note">Same Note:</span>
                <input type="number" step="0.1" value={config.last_note_same ?? 10} onChange={e => updateConfig('last_note_same', parseFloat(e.target.value))} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Penalizes duplicate intervals in chord">Dup Interval:</span>
                <input type="number" step="0.1" value={config.interval_exists_in_harmony ?? 1} onChange={e => updateConfig('interval_exists_in_harmony', parseFloat(e.target.value))} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
            </div>

          <div className="flex gap-4 items-center">
            <button
              onClick={handleRandomise}
              title="Randomise all contours with smooth interpolated curves matching current render length."
              className="px-4 py-2 bg-amber-600 hover:bg-amber-500 text-slate-100 font-bold rounded-lg shadow transition-all active:scale-95"
            >
              Randomise
            </button>
            <button
              onClick={handleDuplicate}
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-slate-100 font-bold rounded-lg shadow transition-all active:scale-95"
            >
              Duplicate (x2)
            </button>
            <button 
              onClick={handleGenerate}
              disabled={isGenerating}
              className="px-6 py-2 bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold rounded-lg shadow-[0_0_15px_rgba(34,211,238,0.4)] transition-all active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isGenerating ? 'Generating...' : 'Generate MIDI'}
            </button>
          </div>
        </div>

        {/* Tabs Row */}
        <div className="flex justify-between items-end">
          <div className="flex gap-2">
            {tabs.map(t => (
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

          {(activeTab === 'voice' || activeTab === 'rhythm') && (
            <div className="flex items-center gap-4 bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800">
              <span className="text-xs text-slate-500 uppercase">Voice Target:</span>
              <div className="flex gap-1 flex-wrap">
                {Array.from({ length: 16 }).map((_, i) => i < 5 ? (
                  <button 
                    key={i} 
                    onClick={() => setSelectedVoice(i)}
                    className={`w-7 h-7 rounded text-xs font-mono transition-colors ${selectedVoice === i ? 'bg-cyan-600 text-slate-950' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}
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
    </div>
  )
}

export default App
