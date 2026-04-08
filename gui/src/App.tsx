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
      default: return null;
    }
  };

  const tabs = [
    { id: 'schillinger', label: 'Schillinger' },
    { id: 'harmony', label: 'Harmony' },
    { id: 'mode', label: 'Mode' },
    { id: 'chord', label: 'Chord' },
    { id: 'voice', label: 'Voice Pitch' },
    { id: 'rhythm', label: 'Voice Rhythm' },
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
                <span className="text-xs uppercase tracking-wider text-slate-500">PL:</span>
                <input type="number" value={config.pl} onChange={e => updateConfig('pl', parseInt(e.target.value))} className="w-12 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500">Render Len:</span>
                <input type="number" value={config.render_length} onChange={e => updateConfig('render_length', parseInt(e.target.value))} className="w-12 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500">Lookahead:</span>
                <input type="number" value={config.lookahead_depth ?? 0} onChange={e => updateConfig('lookahead_depth', parseInt(e.target.value))} className="w-12 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500" title="Last Note Same">Same End:</span>
                <input type="checkbox" checked={config.last_note_same > 0} onChange={e => updateConfig('last_note_same', e.target.checked ? 10.0 : 0.0)} className="accent-cyan-500" />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-500">Seed:</span>
                <input type="number" value={config.rng_seed} onChange={e => updateConfig('rng_seed', parseInt(e.target.value))} className="w-24 bg-transparent text-sm focus:text-cyan-400 outline-none" />
              </div>
            </div>

          <button 
            onClick={handleGenerate}
            disabled={isGenerating}
            className="px-6 py-2 bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold rounded-lg shadow-[0_0_15px_rgba(34,211,238,0.4)] transition-all active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isGenerating ? 'Generating...' : 'Generate MIDI'}
          </button>
        </div>

        {/* Tabs Row */}
        <div className="flex justify-between items-end">
          <div className="flex gap-2">
            {tabs.map(t => (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
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
