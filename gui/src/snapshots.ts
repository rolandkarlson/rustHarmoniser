const STORAGE_KEY = 'harmoniser_snapshots';

export interface Snapshot {
  id: string;
  name: string;
  config: any;
  created_at: string;
}

function readAll(): Snapshot[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function writeAll(snapshots: Snapshot[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshots));
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
  };
  const all = readAll();
  all.unshift(snap);
  writeAll(all);
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
