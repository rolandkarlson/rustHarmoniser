import { useState, useEffect, useRef, useMemo } from 'react'
import { ContourEditor } from './ContourEditor'
import { saveSnapshot, listSnapshots, loadSnapshot, deleteSnapshot, renameSnapshot, toggleFavorite, type Snapshot } from './snapshots'
import './index.css'

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

function App() {
  const [config, setConfig] = useState<any>(null);
  const [activeTab, setActiveTab] = useState('harmony');
  const [selectedVoice, setSelectedVoice] = useState(0);
  const [isGenerating, setIsGenerating] = useState(false);
  const [message, setMessage] = useState('');
  const [snapshots, setSnapshots] = useState<Snapshot[]>([]);
  const [showSnapshots, setShowSnapshots] = useState(false);
  const [favoritesOnly, setFavoritesOnly] = useState(false);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const [broadcastVoices, setBroadcastVoices] = useState(false);
  const snapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch('http://127.0.0.1:3000/api/config')
      .then(res => res.json())
      .then(data => setConfig(data))
      .catch(err => setMessage(`Error loading config: ${err}`));
    setSnapshots(listSnapshots());
  }, []);

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
        case 'r': handleRandomise(); break;
        case 'd': handleDuplicate(); break;
        case 'g': handleGenerate(); break;
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  });

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
    const snap = saveSnapshot(config);
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
    if (newConfig.chord_structure_contour) {
      newConfig.chord_structure_contour = newConfig.chord_structure_contour.map((track: any[]) => duplicateArray(track, stdSteps, 0));
    }
    if (newConfig.schillinger_ex_contour) {
      newConfig.schillinger_ex_contour = newConfig.schillinger_ex_contour.map((track: any[]) => duplicateArray(track, stdSteps, 2));
    }
    if (newConfig.harmony_matrix_contour) newConfig.harmony_matrix_contour = duplicateArray(newConfig.harmony_matrix_contour, stdSteps, 0);

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

  const applyPreset = (preset: 'jazz' | 'classical') => {
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

      // Relaxed classical rules — jazz embraces parallel motion & voice overlap
      nc.last_note_exist_in_voice = 60.0;
      nc.same_direction = 0.3;
      nc.consecutive_octav_fift = 0.0; // parallel 5ths are fine in jazz
      nc.no_crossing = 30.0;
      nc.last_note_same = 5.0;
      nc.interval_exists_in_harmony = 0.5;
      nc.chord_structure = [0, 1, 2, 3, 4, 5]; // full 7th chord voicings

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
          // Walking bass (voice 0): steady quarter notes
          if (v === 0) return 1.0;
          // Comping voices: syncopated
          if (v <= 2) return jazzSnaps[Math.floor((phase * 7 + localPhase * 3 + v) % jazzSnaps.length)];
          // Upper voices: longer, more lyrical
          return [1.0, 2.0, 0.75, 1.0][Math.floor((i + v) % 4)];
        })
      );

      // Voice pitch contour: gentle melodic motion, voice 0 lower (bass)
      nc.voice_contour = Array.from({ length: 16 }, (_, v) =>
        Array.from({ length: steps }, (_, i) => {
          const phase = i / steps;
          const base = v === 0 ? -8 : v <= 2 ? 0 : 4;
          return Math.round(base + 3 * Math.sin(phase * Math.PI * (2 + v * 0.3)));
        })
      );

      // Harmony matrix: Jazz(1) base, dipping into Tension(2) at phrase peaks, Ethereal(3) for ballad sections
      nc.harmony_matrix_contour = Array.from({ length: steps }, (_, i) => {
        const phase = i / steps;
        const section = Math.floor(phase * 4);
        return [1, 2, 1, 3][section] ?? 1; // Jazz → Tension → Jazz → Ethereal
      });

    } else {
      // Classical: Ionian/Aeolian modes, strict voice-leading, balanced phrases
      nc.pl = 4;
      nc.main_pitch = Math.floor(Math.random() * 12); // random key
      nc.mode = 0; // Ionian (major)
      nc.lookahead_depth = 3;
      nc.harmony_distance_balance = 0.2;
      nc.rng_seed = Math.floor(Math.random() * 999999);

      // Strict classical voice-leading rules
      nc.last_note_exist_in_voice = 120.0;
      nc.same_direction = 2.0; // encourage contrary motion
      nc.consecutive_octav_fift = 50.0; // strictly avoid parallel 5ths/octaves
      nc.no_crossing = 150.0; // strict voice separation
      nc.last_note_same = 15.0;
      nc.interval_exists_in_harmony = 2.0;
      nc.chord_structure = [0, 1, 2, 4]; // triads & simple 7ths

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
      const classSnaps = [0.5, 1.0, 2.0, 4.0];
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

      // Voice pitch contour: smooth, arched, voice 0 = bass
      nc.voice_contour = Array.from({ length: 16 }, (_, v) =>
        Array.from({ length: steps }, (_, i) => {
          const phase = i / steps;
          const base = v === 0 ? -10 : v === 1 ? -4 : v === 2 ? 0 : 5;
          return Math.round(base + 4 * Math.sin(phase * Math.PI) * (v < 2 ? -0.5 : 1));
        })
      );

      // Harmony matrix: Strict Classical(0) base, Dark(4) in development, back to Classical(0)
      nc.harmony_matrix_contour = Array.from({ length: steps }, (_, i) => {
        const phase = i / steps;
        if (phase > 0.35 && phase < 0.65) return 4; // Dark/Melancholic for development
        return 0; // Strict Classical for exposition/recap
      });
    }

    setConfig(nc);
    setMessage(`${preset === 'jazz' ? 'Jazz' : 'Classical'} preset loaded`);
  };

  const handleInit = () => {
    if (!config) return;
    const nc = { ...config, render_length: 2 };
    const steps = Math.ceil((nc.pl * 4 * nc.render_length) / nc.voice_contour_resolution);
    nc.harmony_distance_contour = new Array(steps).fill(0);
    nc.mode_contour = new Array(steps).fill(0);
    nc.chord_structure_contour = Array.from({ length: 16 }, () => new Array(steps).fill(0));
    nc.schillinger_ex_contour = Array.from({ length: 16 }, () => new Array(steps).fill(2));
    nc.voice_contour = Array.from({ length: 16 }, () => new Array(steps).fill(0));
    nc.voice_rhythm_contour = Array.from({ length: 16 }, () => new Array(steps).fill(1.0));
    if (nc.harmony_matrix_contour) nc.harmony_matrix_contour = new Array(steps).fill(0);
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
      const chordIndices = chordStruct.map(item => (item * expansion) + seqRoot);
      const modShim = (v: number, len: number) => ((v % len) + len) % len;
      const currentChord = chordIndices.map(idx => currentScale[modShim(idx, currentScale.length)] % 12);

      for (let m = 0; m < 7; m++) {
        if (m === currentMode) continue;
        const otherScale = generateModeFromSteps(m);
        const otherChord = chordIndices.map(idx => otherScale[modShim(idx, otherScale.length)] % 12);
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
    { id: 'harmony_matrix', label: 'H. Matrix', tip: 'Harmony style profile over time. 0=Classical, 1=Jazz, 2=Tension, 3=Ethereal, 4=Dark, 5=Bright, 6=Aggressive, 7=Ancient, 8=Neutral. Fractional values interpolate between rows.' },
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

          <div className="flex gap-4 items-center flex-wrap bg-slate-950 px-4 py-2 rounded-lg border border-slate-800">
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
              <span className="text-xs uppercase tracking-wider text-slate-500" title="Random seed — same seed produces identical output. Change for a different arrangement.">Seed:</span>
              <input type="number" value={config.rng_seed} onChange={e => updateConfig('rng_seed', parseInt(e.target.value))} className="w-24 bg-transparent text-sm focus:text-cyan-400 outline-none" />
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs uppercase tracking-wider text-slate-500" title="Main Pitch — MIDI note offset added to all output pitches. 60 = Middle C.">Pitch:</span>
              <input type="number" value={config.main_pitch ?? 60} onChange={e => updateConfig('main_pitch', parseInt(e.target.value))} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs uppercase tracking-wider text-slate-500" title="When enabled, candidate notes are constrained to Schillinger-derived scale degrees. When off, notes can be any pitch within range.">Schillinger:</span>
              <input type="checkbox" checked={config.schillinger_progression ?? true} onChange={e => updateConfig('schillinger_progression', e.target.checked)} className="accent-cyan-500" />
            </div>
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
              <span className="text-xs uppercase tracking-wider text-slate-500" title="Use the pitches from an Ableton clip as the leading voice (channel 0). Pitches are cycled through voice_rhythm timing.">Lead Clip:</span>
              <input type="checkbox" checked={config.use_leading_voice ?? false} onChange={e => updateConfig('use_leading_voice', e.target.checked)} className="accent-cyan-500" />
              <input type="number" value={config.leading_voice_track ?? 0} onChange={e => updateConfig('leading_voice_track', parseInt(e.target.value))} title="Ableton track index" className="w-12 bg-transparent text-sm focus:text-cyan-400 outline-none" disabled={!config.use_leading_voice} />
              <span className="text-xs text-slate-600">/</span>
              <input type="number" value={config.leading_voice_clip ?? 1} onChange={e => updateConfig('leading_voice_clip', parseInt(e.target.value))} title="Ableton clip index" className="w-12 bg-transparent text-sm focus:text-cyan-400 outline-none" disabled={!config.use_leading_voice} />
            </div>
            <div className="border-l border-slate-700 h-5"></div>
            <div className="flex items-center gap-2">
              <span className="text-xs uppercase tracking-wider text-slate-500" title="Penalizes pitches already in voice history. Higher = more variety.">Note Repeat:</span>
              <input type="number" step="1" value={config.last_note_exist_in_voice ?? 100} onChange={e => updateConfig('last_note_exist_in_voice', parseFloat(e.target.value))} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs uppercase tracking-wider text-slate-500" title="Penalizes repeating the immediate previous note. Higher = more melodic movement.">Same Note:</span>
              <input type="number" step="0.1" value={config.last_note_same ?? 10} onChange={e => updateConfig('last_note_same', parseFloat(e.target.value))} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs uppercase tracking-wider text-slate-500" title="Penalizes outer voices moving in the same direction as harmony. Encourages contrary motion.">Same Dir:</span>
              <input type="number" step="0.1" value={config.same_direction ?? 1} onChange={e => updateConfig('same_direction', parseFloat(e.target.value))} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs uppercase tracking-wider text-slate-500" title="Penalizes parallel fifths and unisons. Classical voice-leading rule. 0 = disabled.">Par 5th/Oct:</span>
              <input type="number" step="0.1" value={config.consecutive_octav_fift ?? 0} onChange={e => updateConfig('consecutive_octav_fift', parseFloat(e.target.value))} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs uppercase tracking-wider text-slate-500" title="Prevents voice crossing — voices must stay in their register. Higher = stricter.">No Cross:</span>
              <input type="number" step="1" value={config.no_crossing ?? 100} onChange={e => updateConfig('no_crossing', parseFloat(e.target.value))} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs uppercase tracking-wider text-slate-500" title="Penalizes duplicate intervals in chord. Adds harmonic variety.">Dup Interval:</span>
              <input type="number" step="0.1" value={config.interval_exists_in_harmony ?? 1} onChange={e => updateConfig('interval_exists_in_harmony', parseFloat(e.target.value))} className="w-14 bg-transparent text-sm focus:text-cyan-400 outline-none" />
            </div>
          </div>

          <div className="flex gap-2 items-center ml-auto">
            <button
              onClick={() => applyPreset('jazz')}
              title="Jazz preset — Dorian/Mixolydian, ii-V-I turnarounds, syncopated rhythms, relaxed voice-leading"
              className="px-4 py-2 bg-yellow-700 hover:bg-yellow-600 text-slate-100 font-bold rounded-lg shadow transition-all active:scale-95"
            >
              Jazz
            </button>
            <button
              onClick={() => applyPreset('classical')}
              title="Classical preset — Ionian/Aeolian, I-IV-V-I, strict counterpoint rules, arch-form dynamics"
              className="px-4 py-2 bg-rose-800 hover:bg-rose-700 text-slate-100 font-bold rounded-lg shadow transition-all active:scale-95"
            >
              Classical
            </button>
            <button
              onClick={handleInit}
              title="Reset render length to 2 and zero all contours (rhythm = 1 beat)"
              className="px-4 py-2 bg-slate-600 hover:bg-slate-500 text-slate-100 font-bold rounded-lg shadow transition-all active:scale-95"
            >
              Init
            </button>
            <button
              onClick={handleRandomise}
              title="Randomise all contours [R]"
              className="px-4 py-2 bg-amber-600 hover:bg-amber-500 text-slate-100 font-bold rounded-lg shadow transition-all active:scale-95"
            >
              Randomise
            </button>
            <button
              onClick={handleRandomiseRhythm}
              title="Randomise only rhythm — fractal / self-similar pulse across octaves"
              className="px-4 py-2 bg-pink-600 hover:bg-pink-500 text-slate-100 font-bold rounded-lg shadow transition-all active:scale-95"
            >
              Rhythm
            </button>
            <button
              onClick={handleDuplicate}
              title="Duplicate all contours to double the length [D]"
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-slate-100 font-bold rounded-lg shadow transition-all active:scale-95"
            >
              Duplicate
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

          {(activeTab === 'voice' || activeTab === 'rhythm' || activeTab === 'schillinger_ex' || activeTab === 'chord') && (
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
    </div>
  )
}

export default App
