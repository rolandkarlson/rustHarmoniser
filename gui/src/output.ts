export type OutputMode = 'notes' | 'trance_arp' | 'contour_arp' | 'voice_arp';

export type OutputRoute = {
  enabled: boolean;
  name: string;
  track_index: number;
  clip_index: number;
  voices: number[];
  mode: OutputMode;
  transpose: number;
  time_scale: number;
  merge_adjacent: boolean;
  humanize_beats: number;
  fold_pitch: number | null;
  arp_durations: number[];
  voice_pattern: number[];
  initial_pitch: number;
  lead_voice: number;
  bass_voice: number;
  lead_octave: number;
};

export type OutputSettings = { enabled: boolean; routes: OutputRoute[] };
export type OutputClip = { track_index: number; clip_index: number; name: string; length: number; note_count: number };

export const OUTPUT_STORAGE_KEY = 'harmoniser_output_v1';
export const parseNumberList = (value: string): number[] => value.split(',').map(Number).filter(Number.isFinite);

export function route(name: string, voices: number[], trackIndex = 4, clipIndex = 2): OutputRoute {
  return {
    enabled: true, name, track_index: trackIndex, clip_index: clipIndex, voices,
    mode: 'notes', transpose: 0, time_scale: 1, merge_adjacent: true,
    humanize_beats: 0, fold_pitch: null, arp_durations: [0.25], voice_pattern: [],
    initial_pitch: 60, lead_voice: 0, bass_voice: Math.max(0, voices.at(-1) ?? 0), lead_octave: 1,
  };
}

export function rawRoutes(voiceCount: number, legacyTracks?: unknown, legacyClip?: unknown): OutputRoute[] {
  const tracks = Array.isArray(legacyTracks) ? legacyTracks : [];
  const clip = Number.isFinite(Number(legacyClip)) ? Math.max(0, Math.round(Number(legacyClip))) : 2;
  return Array.from({ length: voiceCount }, (_, voice) =>
    ({ ...route(`Voice ${voice + 1}`, [voice], Number.isFinite(Number(tracks[voice])) ? Math.round(Number(tracks[voice])) : 4 + voice, clip), merge_adjacent: false }));
}

export function initialOutput(config: Record<string, unknown>): OutputSettings {
  try {
    const saved = JSON.parse(localStorage.getItem(OUTPUT_STORAGE_KEY) || 'null');
    if (saved && typeof saved.enabled === 'boolean' && Array.isArray(saved.routes)) return saved;
  } catch { /* use migration defaults */ }
  const count = config?.fugue_mode ? Math.min(8, Math.max(3, Math.round(Number(config.fugue_voice_count) || 3))) : 5;
  return { enabled: false, routes: rawRoutes(count, config?.fugue_output_tracks, config?.fugue_output_clip) };
}

export function validateOutput(output: OutputSettings): string[] {
  const errors: string[] = [];
  const targets = new Map<string, string>();
  output.routes.forEach((r, i) => {
    if (!r.enabled) return;
    const label = r.name.trim() || `Route ${i + 1}`;
    if (r.track_index < 0 || r.clip_index < 0) errors.push(`${label}: track and clip must be 1 or higher.`);
    if (r.voices.length === 0) errors.push(`${label}: choose at least one voice.`);
    const key = `${r.track_index}:${r.clip_index}`;
    if (targets.has(key)) errors.push(`${label} collides with ${targets.get(key)} at Track ${r.track_index + 1}, Slot ${r.clip_index + 1}.`);
    else targets.set(key, label);
  });
  return errors;
}

export function unmappedVoices(output: OutputSettings, voiceCount: number): number[] {
  const mapped = new Set(output.routes.filter(r => r.enabled).flatMap(r => r.voices));
  return Array.from({ length: voiceCount }, (_, i) => i).filter(i => !mapped.has(i));
}

export function outOfRangeVoices(output: OutputSettings, voiceCount: number): number[] {
  return [...new Set(output.routes.filter(r => r.enabled).flatMap(r => r.voices).filter(v => v < 0 || v >= voiceCount))];
}

export function saveOutput(output: OutputSettings) {
  localStorage.setItem(OUTPUT_STORAGE_KEY, JSON.stringify(output));
}
