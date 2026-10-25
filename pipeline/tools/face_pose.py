#!/usr/bin/env python3
"""face_pose.py IMG -> "roll R yaw Y" from YuNet eye/nose landmarks (2026-10-25). roll = head tilt in degrees (eye line),
yaw = left/right turn as nose offset from eye midpoint relative to eye distance (0 = straight; ~0.3 = clear 3/4; >0.5 = profile).
Dataset rule (owner): mostly front faces, small turns, head NOT tilted: PASS if |roll| <= 10 and |yaw| <= 0.35."""
import os, sys, math, cv2
FM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vendor", "facemetric", "face_detection_yunet_2023mar.onnx")
im = cv2.imread(sys.argv[1]); h, w = im.shape[:2]; s = 1024 / max(h, w)
if s < 1: im = cv2.resize(im, (int(w * s), int(h * s))); h, w = im.shape[:2]
d = cv2.FaceDetectorYN.create(FM, "", (w, h), 0.7); _, f = d.detect(im)
if f is None: print("FAIL pose no face"); sys.exit()
f = max(f, key=lambda r: r[2] * r[3]); rx, ry, lx, ly, nx, ny = f[4], f[5], f[6], f[7], f[8], f[9]
roll = math.degrees(math.atan2(ly - ry, lx - rx)); ed = math.hypot(lx - rx, ly - ry) or 1; yaw = (nx - (rx + lx) / 2) / ed
ok = abs(roll) <= 10 and abs(yaw) <= 0.35
print(("PASS" if ok else "FAIL") + f" pose roll {roll:+.0f} yaw {yaw:+.2f}")
