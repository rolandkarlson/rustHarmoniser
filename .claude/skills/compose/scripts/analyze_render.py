#!/usr/bin/env python3
"""Musical health report for a rustHarmoniser render archive.

Usage:
    python3 analyze_render.py [render/render_<ts>.json]

With no argument, analyzes the newest archive in render/.
Reads the archived config + notes + breakdown and prints the measurements the
compose skill iterates against: per-bar dissonance, bass-on-root, voice motion,
velocity arc, loop seam, cadence-bar quality, hard violations.
"""
import glob
import itertools
import json
import statistics
import sys

DISSONANT_ICS = {1, 2, 6, 10, 11}  # m2/M2/TT/m7/M7 pitch-interval classes


def sounding(notes, t):
    return sorted({n["pitch"] for n in notes
                   if n["start"] <= t + 1e-6 and n["start"] + n["duration"] > t + 1e-6})


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob("render/render_*.json"))
    doc = json.load(open(path))
    cfg = doc["input"]["config"]
    notes = doc["output"]["notes"]
    bd = doc["output"].get("breakdown", [])
    pl = max(cfg.get("pl", 4), 1)
    bars = pl * max(cfg.get("render_length", 1), 1)

    print(f"archive: {path}")
    print(f"config: mode={cfg.get('mode')} root={cfg.get('root')} main_pitch={cfg.get('main_pitch')} "
          f"pl={pl} bars={bars} loop_wrap={cfg.get('loop_wrap_weight')} "
          f"cadence_w={cfg.get('cadence_weight')} echo={cfg.get('phrase_echo_weight')} "
          f"rgens={cfg.get('rhythm_generators')} cand_range={cfg.get('candidate_range')}")
    print(f"notes: {len(notes)} on channels {sorted({n['channel'] for n in notes})}")

    # Hard violations (beat-0 voice_budget is a known cosmetic artifact).
    hard = [(g["start"], g["hard_violations"]) for g in bd if g.get("hard_violation_count", 0) > 0]
    real = [h for h in hard if not (h[0] == 0.0 and all(v["name"] == "voice_budget" for v in h[1]))]
    print(f"hard violations: {len(real)} real, {len(hard) - len(real)} beat-0 artifact")
    for start, vs in real[:5]:
        print(f"  beat {start}: {vs}")

    # Per-bar dissonance share (sampled off-downbeat so sustains count).
    print("\nbar | dissonant-pair share | mean velocity")
    for bar in range(bars):
        ics, vels = [], []
        for beat in (0.5, 1.5, 2.5, 3.5):
            ps = sounding(notes, bar * 4 + beat)
            ics += [abs(a - b) % 12 for a, b in itertools.combinations(ps, 2)]
        vels = [n["velocity"] for n in notes if bar * 4 <= n["start"] < bar * 4 + 4]
        share = sum(1 for i in ics if i in DISSONANT_ICS) / max(1, len(ics))
        cad = "  <- phrase-final" if (bar + 1) % pl == 0 else ""
        print(f"{bar:3} | {share:20.2f} | {statistics.mean(vels) if vels else 0:13.0f}{cad}")

    # Bass-on-root, from the archived Schillinger stacks when present, else breakdown roots.
    roots = {}
    for g in bd:
        if g.get("root_pc") is not None:
            roots[g["bar"]] = g["root_pc"]
    if roots:
        hits = tot = 0
        for n in notes:
            if n["channel"] != 4:
                continue
            r = roots.get(int(n["start"] // 4))
            if r is None:
                continue
            tot += 1
            hits += n["pitch"] % 12 == r
        print(f"\nbass on chord root: {hits}/{tot} ({hits / max(1, tot):.0%})")

    # Voice motion + loop seam.
    print("\nvoice | moves | max jump | seam (last->first)")
    seam_total = 0
    for ch in sorted({n["channel"] for n in notes}):
        line = sorted((n for n in notes if n["channel"] == ch), key=lambda n: n["start"])
        moves = [abs(b["pitch"] - a["pitch"]) for a, b in zip(line, line[1:]) if a["pitch"] != b["pitch"]]
        seam = abs(line[-1]["pitch"] - line[0]["pitch"])
        seam_total += seam
        print(f"{ch:5} | {len(moves):5} | {max(moves) if moves else 0:8} | {seam}")
    print(f"total loop-seam distance: {seam_total}")

    # Cadence arrivals: phrase-final downbeat chords.
    print("\nphrase-final downbeats:")
    for bar in range(pl - 1, bars, pl):
        t = bar * 4.0
        chord = [n for n in notes if abs(n["start"] - t) < 1e-6]
        if not chord:
            print(f"  bar {bar}: no downbeat onset")
            continue
        pcs = sorted({n["pitch"] % 12 for n in chord})
        r = roots.get(bar)
        stat = ""
        if r is not None:
            bass = min(n["pitch"] for n in chord)
            third = ((r + 3) % 12 in pcs) or ((r + 4) % 12 in pcs)
            stat = f" root_pc={r} bass_on_root={bass % 12 == r} third_present={third}"
        print(f"  bar {bar}: pcs={pcs}{stat}")


if __name__ == "__main__":
    main()
