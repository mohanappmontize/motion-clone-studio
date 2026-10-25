#!/usr/bin/env python3
"""motion_score.py CLIP DRIVING [--skip S] [--n 12] -> numbers for a motion-copy clip (CPU, no tokens).
id   : meher_validate identity sim per sampled frame (floor 0.50)
pose : |roll| deg and |yaw| error of clip face vs driving face at the same moment (copy of head motion)
track: correlation of face-centre x/y and face-size tracks clip vs driving (copy of body/camera motion; 1 = same)
mouth: correlation of mouth-width/eye-distance tracks (rough expression copy)
--skip = driving seconds skipped when the clip was made."""
import argparse, os, re, subprocess, sys, tempfile, math
import cv2, numpy as np
TOOLS = os.environ.get("MC_TOOLS") or os.path.expanduser("~/automation/shorts-factory/tools")
FM = os.path.join(TOOLS, "..", "vendor", "facemetric", "face_detection_yunet_2023mar.onnx")
ap = argparse.ArgumentParser(); ap.add_argument("clip"); ap.add_argument("driving")
ap.add_argument("--skip", type=float, default=0); ap.add_argument("--n", type=int, default=12)
a = ap.parse_args()

def frame(path, t):
    c = cv2.VideoCapture(path); c.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, im = c.read(); c.release()
    return im if ok else None

def face(im):
    h, w = im.shape[:2]; s = 1024 / max(h, w)
    if s < 1: im = cv2.resize(im, (int(w * s), int(h * s))); h, w = im.shape[:2]
    d = cv2.FaceDetectorYN.create(FM, "", (w, h), 0.6); _, f = d.detect(im)
    if f is None: return None
    f = max(f, key=lambda r: r[2] * r[3]); rx, ry, lx, ly, nx, ny, mrx, mry, mlx, mly = f[4:14]
    ed = math.hypot(lx - rx, ly - ry) or 1
    return dict(cx=(f[0] + f[2] / 2) / w, cy=(f[1] + f[3] / 2) / h, sz=f[3] / h,
                roll=math.degrees(math.atan2(ly - ry, lx - rx)), yaw=(nx - (rx + lx) / 2) / ed,
                mouth=math.hypot(mlx - mrx, mly - mry) / ed, mo=(((mry + mly) / 2) - ny) / ed)

cap = cv2.VideoCapture(a.clip); dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 1); cap.release()
ts = [dur * (i + 0.5) / a.n for i in range(a.n)]
tmp = tempfile.mkdtemp(); C, D, ids = [], [], []
for t in ts:
    ci, di = frame(a.clip, t), frame(a.driving, t + a.skip)
    p = os.path.join(tmp, f"{t:.2f}.png"); cv2.imwrite(p, ci)
    out = subprocess.run([sys.executable, os.path.join(TOOLS, "meher_validate.py"), "image", p], capture_output=True, text=True).stdout
    m = re.search(r"identity\s+sim ([\d.]+)", out); ids.append(float(m.group(1)) if m else 0.0)
    C.append(face(ci)); D.append(face(di) if di is not None else None)
ok = [i for i in range(a.n) if C[i] and D[i]]
def corr(k):
    x = np.array([C[i][k] for i in ok]); y = np.array([D[i][k] for i in ok])
    return float(np.corrcoef(x, y)[0, 1]) if len(ok) > 2 and x.std() > 1e-6 and y.std() > 1e-6 else float("nan")
roll_e = np.mean([abs(C[i]["roll"] - D[i]["roll"]) for i in ok]) if ok else float("nan")
yaw_e = np.mean([abs(C[i]["yaw"] - D[i]["yaw"]) for i in ok]) if ok else float("nan")
print("id   " + " ".join(f"{t:.1f}:{s:.2f}" for t, s in zip(ts, ids)))
print(f"id   mean {np.mean(ids):.3f} min {min(ids):.3f} below0.50 {sum(s < 0.5 for s in ids)}/{a.n}")
print(f"pose roll_err {roll_e:.1f}deg yaw_err {yaw_e:.2f}  faces {len(ok)}/{a.n}")
print(f"track corr x {corr('cx'):+.2f} y {corr('cy'):+.2f} size {corr('sz'):+.2f} | head corr roll {corr('roll'):+.2f} yaw {corr('yaw'):+.2f}")
print(f"mouth corr width {corr('mouth'):+.2f} drop {corr('mo'):+.2f}")
