const STORAGE_KEY = 'harmoniser_snapshots';

// Snapshots hold full configs (large per-voice contour arrays), so they eat
// localStorage fast. Cap the number of auto-saved (non-favorite) entries;
// favorites are never counted against the cap and never auto-evicted.
const MAX_AUTO_SNAPSHOTS = 40;

export interface Snapshot {
  id: string;
  name: string;
  config: any;
  created_at: string;
  favorite: boolean;
}

function readAll(): Snapshot[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return parsed.map((s: any) => ({ favorite: false, ...s }));
  } catch {
    return [];
  }
}

// Keep all favorites plus the newest MAX_AUTO_SNAPSHOTS non-favorites.
// Input is expected newest-first, so this drops the oldest auto-saves.
function enforceCap(all: Snapshot[]): Snapshot[] {
  let kept = 0;
  return all.filter(s => {
    if (s.favorite) return true;
    kept++;
    return kept <= MAX_AUTO_SNAPSHOTS;
  });
}

function tryPersist(snapshots: Snapshot[]): boolean {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshots));
    return true;
  } catch {
    return false; // QuotaExceededError (or storage disabled)
  }
}

// Persist, degrading gracefully on quota errors: evict the oldest non-favorite
// snapshot and retry until it fits or only favorites remain. Never throws.
function writeAll(snapshots: Snapshot[]) {
  let list = snapshots;
  if (tryPersist(list)) return;
  while (true) {
    // Oldest non-favorite (list is newest-first, so scan from the end).
    let dropAt = -1;
    for (let i = list.length - 1; i >= 0; i--) {
      if (!list[i].favorite) { dropAt = i; break; }
    }
    if (dropAt === -1) return; // only favorites left; nothing more we can shed
    list = list.filter((_, i) => i !== dropAt);
    if (tryPersist(list)) return;
  }
}

export function saveSnapshot(config: any): Snapshot {
  const now = new Date();
  const snap: Snapshot = {
    id: crypto.randomUUID(),
    name: now.toLocaleString('en-GB', {
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
    }).replace(',', ''),
    config: structuredClone(config),
    created_at: now.toISOString(),
    favorite: false,
  };
  const all = readAll();
  all.unshift(snap);
  writeAll(enforceCap(all));
  return snap;
}

export function listSnapshots(): Snapshot[] {
  return readAll();
}

export function loadSnapshot(id: string): any | null {
  const snap = readAll().find(s => s.id === id);
  return snap ? snap.config : null;
}

export function deleteSnapshot(id: string) {
  writeAll(readAll().filter(s => s.id !== id));
}

export function renameSnapshot(id: string, name: string) {
  const all = readAll();
  const snap = all.find(s => s.id === id);
  if (snap) {
    snap.name = name;
    writeAll(all);
  }
}

export function toggleFavorite(id: string) {
  const all = readAll();
  const snap = all.find(s => s.id === id);
  if (snap) {
    snap.favorite = !snap.favorite;
    writeAll(all);
  }
}
