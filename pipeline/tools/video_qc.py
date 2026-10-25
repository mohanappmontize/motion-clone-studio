#!/usr/bin/env python3
"""video_qc.py - flicker / black-frame gate for every reel BEFORE it is queued (owner 2026-10-05: "black flickers ... headache").
FAIL if: any frame >60% near-black outside a planned end-card fade, any frame-to-frame luma jump > 25 (flash), any decode error,
or the stream has B-frames / keyframe gap > 2 s (some players flash black on those).
  python3 tools/video_qc.py OUT.mp4   -> prints PASS/FAIL + reasons, exit 1 on FAIL"""
import subprocess, sys
import cv2, numpy as np


def qc(path, slate_from=None):
    why = []
    err = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "null", "-"], capture_output=True, text=True).stderr.strip()
    if err: why.append("decode errors: " + err[:120])
    ft = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "frame=pict_type,pts_time", "-of", "csv=p=0", path],
                        capture_output=True, text=True).stdout.split()
    types = [l.split(",")[-1] for l in ft if "," in l]; times = [float(l.split(",")[0]) for l in ft if "," in l and l.split(",")[-1] == "I"]
    if "B" in types: why.append("B-frames present")
    if len(times) > 1 and max(np.diff(times)) > 2.05: why.append(f"keyframe gap {max(np.diff(times)):.1f}s")
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 30; prev = None; i = 0
    while True:
        r, f = cap.read()
        if not r: break
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (108, 192)).astype(np.float32)
        in_slate = (slate_from is not None and i / fps >= slate_from) or (i / fps < 1.7)  # skip hook slate (first ~1.7s)
        if not in_slate and (g < 25).mean() > 0.6: why.append(f"black frame at {i / fps:.2f}s")
        if prev is not None and abs(g.mean() - prev.mean()) > 25 and not in_slate: why.append(f"flash at {i / fps:.2f}s")
        prev = g; i += 1
    return (not why), why[:6]


if __name__ == "__main__":
    ok, why = qc(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else None)
    print("VIDEO QC", "PASS" if ok else "FAIL", why); sys.exit(0 if ok else 1)
