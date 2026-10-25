#!/usr/bin/env python3
"""meher_validate.py - deterministic pre-approval gates for Meher images and reels. CPU only, ZERO LLM tokens.

Every check returns PASS / WARN / FAIL. Exit code 0 = no FAIL, 1 = at least one FAIL, 2 = bad usage.
Checks were chosen from the redo causes of 2026-10-12..16 (see memory: reel-endcard-permanent,
reel-title-face-cutout-gate, project-reel-qa-gate, meher-covered-ref-always, project-meher-identity-lock).

  image  IMG [--pose auto|front|3q|profile] [--sheet SENT_SHEET]
  reel   MP4 [--brand TXT --product TXT] [--cutout PNG] [--ugc] [--title-box x0,y0,x1,y1]   (build_hf.py writes MP4.meta.json)
  selftest                       build broken fixtures from a good reel and prove each gate catches its fault

Identity uses the SAME instrument as tools/face_similarity.py (YuNet + SFace, downscale >1024, most confident
face); it is imported logic, not a second scorer. Pose gate from memory: frontal 0.80, 3/4 0.55, profile 0.35.
NOT automated (measured 2026-10-16, do not re-try): bag-vs-catalog by SIFT/colour histogram. SIFT put a Gucci quilt at 138
inliers vs the true Bottega at 12; colour-histogram rank of the true bag was 238/481 (random). Bag identity stays with
tools/reel_qa.py (gemma judge) + the eyeball-the-packshot rule.
Usage: .venv-img/bin/python tools/meher_validate.py reel out.mp4 --brand "Goyard" --product "Saint Louis"
"""
import json, argparse, hashlib, json, os, re, shutil, subprocess, sys, tempfile
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FM = os.path.join(HERE, "..", "vendor", "facemetric")
MZ = os.environ.get("MEHER_ZIP", "/home/mohan_singh/Desktop/MEHER_ZIP")
ANCHOR = os.environ.get("MEHER_ANCHOR") or MZ + "/03_references/sheet_covered_1015/ANCHOR_face.png"
BODY_OK = MZ + "/03_references/sheet_covered_1015/BODY_covered_v1.png"
BODY_BAD = MZ + "/03_references/sheet_covered_1015/FULLBODY_v2_spec.png"
POSE_FLOOR = {"front": 0.72, "3q": 0.55, "profile": 0.35}   # FAIL floors; front lowered from the 0.80 design gate to 0.72 because owner-approved stills measured 0.748/0.754
POSE_GOOD = {"front": 0.80, "3q": 0.60, "profile": 0.45}  # below this but above floor = WARN
# Gemini stills (the pinned-chat workflow) measured 0.57-0.70 on posted images, so the local-pipeline floors would fail most of them.
GEMINI_FLOOR = {"front": 0.50, "3q": 0.45, "profile": 0.35}; GEMINI_WARN = 0.57   # stranger = 0.15
SOURCE = "gemini"
DARK = 28  # mean luma under this = black frame

_det = _rec = None
def _models():
    global _det, _rec
    if _det is None:
        _det = cv2.FaceDetectorYN.create(os.path.join(FM, "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.6)
        _rec = cv2.FaceRecognizerSF.create(os.path.join(FM, "face_recognition_sface_2021dec.onnx"), "")
    return _det, _rec

def _face(im):
    """most confident face (not largest - see face_similarity.py note), after the canonical >1024 downscale.
    returns (feature, box in ORIGINAL px x0,y0,x1,y1, yaw_ratio, conf) or None"""
    det, rec = _models()
    s = min(1.0, 1024 / max(im.shape[:2]))
    sm = cv2.resize(im, None, fx=s, fy=s) if s < 1 else im
    det.setInputSize((sm.shape[1], sm.shape[0]))
    _, faces = det.detect(sm)
    if faces is None: return None
    f = max(faces, key=lambda f: f[-1])
    x, y, w, h = f[:4]
    rex, rey, lex, ley, nx = f[4], f[5], f[6], f[7], f[8]  # right eye (image left), left eye, nose
    ed = max(1.0, abs(lex - rex))
    yaw = abs(nx - (rex + lex) / 2) / ed
    return rec.feature(rec.alignCrop(sm, f)), (x / s, y / s, (x + w) / s, (y + h) / s), yaw, float(f[-1])

def _pose_of(yaw):
    return "front" if yaw < 0.12 else "3q" if yaw < 0.30 else "profile"

_anchor_feat = None
def _anchor():
    global _anchor_feat
    if _anchor_feat is None:
        r = _face(cv2.imread(ANCHOR))
        if r is None: sys.exit("no face in ANCHOR_face")
        _anchor_feat = r[0]
    return _anchor_feat

def _sim(feat):
    return float(_models()[1].match(_anchor(), feat, cv2.FaceRecognizerSF_FR_COSINE))

# ---------------------------------------------------------------- image checks
def chk_identity(im, pose="auto"):
    r = _face(im)
    if r is None: return ("identity", "FAIL", "NO FACE found")
    feat, box, yaw, conf = r
    p = _pose_of(yaw) if pose == "auto" else pose
    sc = _sim(feat); fh = box[3] - box[1]
    fl = (GEMINI_FLOOR if SOURCE == "gemini" else POSE_FLOOR)[p]; good = min(POSE_GOOD[p], GEMINI_WARN) if SOURCE == "gemini" else POSE_GOOD[p]
    d = "sim %.3f pose %s floor %.2f source %s face_h %dpx conf %.2f" % (sc, p, fl, SOURCE, fh, conf)
    if sc < fl: return ("identity", "FAIL", d)
    if sc < good: return ("identity", "WARN", d + " (above floor, below design gate %.2f)" % good)
    if fh < 90: return ("identity", "WARN", d + " (face small: identity collapses at wide framing)")
    return ("identity", "PASS", d)

def _orb_inliers(a, b):
    """SIFT + ratio test + RANSAC homography inliers of template a found inside b"""
    sift = cv2.SIFT_create(nfeatures=1500)
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY); gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ka, da = sift.detectAndCompute(ga, None); kb, db = sift.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8: return 0
    good = [m for m, n in cv2.BFMatcher().knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
    if len(good) < 8: return len(good) // 4
    H, mask = cv2.findHomography(np.float32([ka[m.queryIdx].pt for m in good]),
                                 np.float32([kb[m.trainIdx].pt for m in good]), cv2.RANSAC, 4.0)
    return int(mask.sum()) if mask is not None else 0

def chk_sheet(sheet_path):
    """covered-ref rule: the sheet sent to Gemini must contain the covered body, never the old plunging full-body ref"""
    sh = cv2.imread(sheet_path)
    if sh is None: return ("covered_ref", "FAIL", "cannot read sheet " + sheet_path)
    if "FULLBODY" in os.path.basename(sheet_path).upper(): return ("covered_ref", "FAIL", "sheet is the old FULLBODY ref")
    ok = _orb_inliers(cv2.imread(BODY_OK), sh); bad = _orb_inliers(cv2.imread(BODY_BAD), sh)
    d = "inliers covered=%d fullbody=%d" % (ok, bad)
    if bad >= 12 and bad >= ok: return ("covered_ref", "FAIL", d + " - old plunging ref present")
    if ok < 12: return ("covered_ref", "FAIL", d + " - covered body ref not found in sheet")
    return ("covered_ref", "PASS", d)


def run_image(a):
    global SOURCE; SOURCE = a.source
    im = cv2.imread(a.target)
    if im is None: return [("file", "FAIL", "unreadable image " + a.target)]
    res = [chk_identity(im, a.pose)]
    if a.sheet: res.append(chk_sheet(a.sheet))
    return res

# ---------------------------------------------------------------- reel checks
def _probe(mp4):
    o = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,r_frame_rate:format=duration",
                        "-of", "json", mp4], capture_output=True, text=True).stdout
    j = json.loads(o or "{}"); v = next((s for s in j.get("streams", []) if s["codec_type"] == "video"), None)
    au = any(s["codec_type"] == "audio" for s in j.get("streams", []))
    fr = v["r_frame_rate"].split("/") if v else ["0", "1"]
    return v, au, float(j.get("format", {}).get("duration", 0)), float(fr[0]) / max(1.0, float(fr[1]))

def _frame(mp4, t, tmp):
    p = os.path.join(tmp, "f%0.3f.png" % t)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "%.3f" % max(0, t), "-i", mp4, "-frames:v", "1", p])
    return cv2.imread(p) if os.path.exists(p) else None

def chk_format(v, au, dur, fps):
    if not v: return ("format", "FAIL", "no video stream")
    d = "%sx%s %.1ffps %.1fs audio=%s" % (v["width"], v["height"], fps, dur, au)
    if dur < 2.0: return ("format", "FAIL", d + " - shorter than 2s (1-frame or truncated file)")
    if (v["width"], v["height"]) != (1080, 1920): return ("format", "FAIL", d + " - must be 1080x1920")
    if not au: return ("format", "FAIL", d + " - NO AUDIO TRACK (music is mandatory)")
    if dur < 5 or fps < 23: return ("format", "WARN", d)
    return ("format", "PASS", d)

def chk_audio(mp4):
    """owner 2026-10-21: music must never be missing - audio must be present and audible (mean >= -35 dB)"""
    import subprocess, re
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", mp4, "-af", "volumedetect", "-vn", "-f", "null", "-"], capture_output=True, text=True).stderr
    m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", r)
    if not m: return ("audio", "FAIL", "no audio stream / cannot measure")
    v = float(m.group(1))
    return ("audio", "PASS" if v >= -35 else "FAIL", "mean volume %.1f dB (>= -35 required)" % v)

def chk_black(mp4, dur, tmp):
    f0 = _frame(mp4, 0.0, tmp); f1 = _frame(mp4, 0.12, tmp); fe = _frame(mp4, dur - 0.08, tmp); fe2 = _frame(mp4, dur - 0.5, tmp)
    L = lambda f: float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()) if f is not None else -1
    l0, l1, le, le2 = L(f0), L(f1), L(fe), L(fe2)
    d = "luma t0 %.0f t0.12 %.0f tail %.0f tail-0.5s %.0f" % (l0, l1, le, le2)
    bad = []
    if l0 < DARK and l1 < DARK: bad.append("black first frames")
    if le < DARK: bad.append("black tail")
    if bad: return ("black_frames", "FAIL", d + " - " + ", ".join(bad))
    if l0 < DARK: return ("black_frames", "WARN", d + " - frame 0 black (IG cover risk, ok if cover image passed)")
    return ("black_frames", "PASS", d)

def _ocr(im):
    """tesseract word boxes -> [(text, conf, x, y, w, h)]"""
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "o.png"); cv2.imwrite(p, im)
        o = subprocess.run(["tesseract", p, "-", "--psm", "11", "tsv"], capture_output=True, text=True).stdout
    out = []
    for ln in o.splitlines()[1:]:
        c = ln.split("\t")
        if len(c) == 12 and c[11].strip() and float(c[10]) > 0:
            out.append((c[11].strip(), float(c[10]), int(c[6]), int(c[7]), int(c[8]), int(c[9])))
    return out

def chk_endcard(mp4, dur, tmp, brand, product):
    f = _frame(mp4, dur - 0.35, tmp)
    if f is None: return [("endcard", "FAIL", "cannot read last frame")]
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); h, w = g.shape; res = []
    dark_edge = [int((g[:, :22] < 150).sum()), int((g[:, w - 22:] < 150).sum())]
    ivory = float(np.median(g[:, :40]))
    d = "dark px L/R %s, bg luma %.0f" % (dark_edge, ivory)
    if ivory < 200: res.append(("endcard_style", "WARN", d + " - end card background not ivory"))
    elif max(dark_edge) >= 20: res.append(("endcard_clip", "FAIL", d + " - element touches side edge (clipped)"))
    else: res.append(("endcard_clip", "PASS", d))
    words = _ocr(f); txt = " ".join(x[0] for x in words).upper()
    edge = [x[0] for x in words if x[1] > 60 and len(re.sub(r"[^A-Za-z]", "", x[0])) >= 3 and (x[2] < 6 or x[2] + x[4] > w - 6)]   # real words only: OCR noise ("|", "—", "J") caused false alarms 2026-10-16
    if edge: res.append(("endcard_text_clip", "FAIL", "word touches frame edge: " + ",".join(edge)))
    norm = lambda s: re.sub(r"[^A-Z0-9]", "", s.upper())
    for label, val in (("brand", brand), ("product", product)):
        if not val: continue
        parts = [p for p in re.split(r"\s+", val.upper()) if len(p) > 2]
        # script fonts (brand) OCR poorly: judge by caps product line strictly, brand loosely (WARN)
        seen = [p for p in parts if norm(p)[:5] in norm(txt)]
        lvl = "FAIL" if label == "product" else "WARN"
        if len(seen) < max(1, len(parts) // 2 + (len(parts) % 2)):
            res.append(("endcard_" + label, lvl, "OCR did not read %s '%s' (read: %s)" % (label, val, txt[:80])))
        else: res.append(("endcard_" + label, "PASS", "read %s" % seen))
    if "MEHER" not in norm(txt) and "WANDRER" not in norm(txt) and "SAVE" not in norm(txt):
        res.append(("endcard_chrome", "WARN", "no MEHER/WANDRER/SAVE text read; was the end card built by build_hf.py? OCR: " + txt[:60]))
    return res

def chk_title_face(mp4, tmp, title_box):
    """finished-reel check for t<3.5s: a face must not sit under the title block. Title box = OCR text in the upper
    55% of the frame (falls back to --title-box or the build_hf default zone)."""
    hits = []; seen = 0; used = "default"
    try: _te = float(json.load(open(mp4 + ".meta.json")).get("title_end", 99))
    except Exception: _te = 99
    for t in (0.5, 1.3, 2.1, 2.9, 3.4):
        if t > _te: continue   # title already faded out (hook scenes): nothing left to overlap the face
        f = _frame(mp4, t, tmp)
        if f is None: continue
        h, w = f.shape[:2]
        r = _face(f)
        if r is None: continue
        box = title_box
        if title_box is None:
            ws = [x for x in _ocr(f[: int(h * .60)]) if x[1] > 60 and len(re.sub(r"[^A-Za-z]", "", x[0])) > 2 and x[5] >= 30]
            if ws:  # the title is the big text: keep words within 330px of the biggest/most confident word (ignores signs, clocks, carvings)
                k = max(ws, key=lambda x: x[5] * x[1]); ws = [x for x in ws if abs(x[3] - k[3]) <= 330]
                box = (min(x[2] for x in ws), min(x[3] for x in ws), max(x[2] + x[4] for x in ws), max(x[3] + x[5] for x in ws)); used = "ocr"
            else: continue  # script-font title not readable by OCR: do not guess the zone (a moved title made the default box a false FAIL)
        seen += 1; fb = r[1]
        ov = max(0, min(fb[2], box[2]) - max(fb[0], box[0])) * max(0, min(fb[3], box[3]) - max(fb[1], box[1]))
        if ov > 0.12 * (fb[2] - fb[0]) * (fb[3] - fb[1]): hits.append(t)
    if seen == 0: return ("title_face", "WARN", "could not judge (no face, or title not located by OCR): build via build_hf.py for the .meta.json sidecar or pass --title-box")
    if hits: return ("title_face", "FAIL", "title block overlaps face at t=%s (box source: %s)" % (hits, used))
    return ("title_face", "PASS", "%d frames with face, no overlap (box: %s)" % (seen, used))

REEL_ID_FLOOR = 0.40   # video frames of approved reels measured 0.49-0.66 (stills floors 0.55-0.80 do not apply); stranger = 0.15
def chk_reel_identity(mp4, dur, tmp):
    sims = []
    for t in np.linspace(0.6, max(1.0, dur - 3.2), 6):
        f = _frame(mp4, float(t), tmp)
        r = _face(f) if f is not None else None
        if r: sims.append(_sim(r[0]))
    if not sims: return ("reel_identity", "WARN", "no face in sampled frames (product-only reel?)")
    med = float(np.median(sims)); drift = [round(s, 2) for s in sims if s < 0.88 * sims[0] or s < REEL_ID_FLOOR]
    d = "median %.2f over %d frames (floor %.2f); drifted frames: %s" % (med, len(sims), REEL_ID_FLOOR, drift)
    if med < REEL_ID_FLOOR: return ("reel_identity", "FAIL", d)
    return ("reel_identity", "WARN" if len(drift) > len(sims) // 2 else "PASS", d)

def solidity(png):
    a = np.array(cv2.imread(png, cv2.IMREAD_UNCHANGED))
    if a.ndim < 3 or a.shape[2] < 4: return 1.0
    al = a[..., 3]; m = (al > 40).astype("uint8"); n, l, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2: return 0.0
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); m = (l == k).astype("uint8")
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    f = np.zeros_like(m); cv2.drawContours(f, cs, -1, 1, -1)
    return float((al[f > 0] > 200).mean())

def chk_cutout(png, mn=0.80):
    v = solidity(png)
    return ("cutout_solid", "PASS" if v >= mn else "FAIL", "alpha solidity %.2f (>=%.2f)" % (v, mn))

def run_reel(a):
    if not os.path.exists(a.target): return [("file", "FAIL", "missing " + a.target)]
    v, au, dur, fps = _probe(a.target)
    res = [chk_format(v, au, dur, fps)]
    if not v: return [("file", "FAIL", "unreadable video (no video stream; truncated or corrupt): " + a.target)]
    if res[0][1] == "FAIL" and dur < 2.0: return res   # no point judging frames of a stub
    meta = {}
    mp = a.target + ".meta.json"   # written by build_hf.py: title box, brand, product, cutout
    if os.path.exists(mp):
        try: meta = json.load(open(mp))
        except Exception: res.append(("meta", "WARN", "unreadable " + mp))
    for k in ("brand", "product", "cutout"):
        if not getattr(a, k, None) and meta.get(k): setattr(a, k, meta[k])
    if not a.title_box and meta.get("title_box"): a.title_box = ",".join(str(int(x)) for x in meta["title_box"])
    tb = tuple(int(x) for x in a.title_box.split(",")) if a.title_box else None
    with tempfile.TemporaryDirectory() as tmp:
        res.append(chk_audio(a.target))
        res.append(chk_black(a.target, dur, tmp))
        res += chk_endcard(a.target, dur, tmp, a.brand, a.product)
        res.append(chk_title_face(a.target, tmp, tb))
        res.append(chk_reel_identity(a.target, dur, tmp))
    if a.cutout: res.append(chk_cutout(a.cutout, meta.get("solid_min", 0.80)))
    if a.ugc and not os.path.exists(a.target + ".wm"):
        res.append(("wm_marker", "FAIL", "no .wm marker: only meaningful for flow_reel/ugc_make reels (Veo sparkle cover); ignore --ugc on build_hf.py reels"))
    return res

# ---------------------------------------------------------------- selftest
def _ff(*args): subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)

def selftest(good):
    """Build known-broken fixtures from a known-good reel; every gate must flag its own fault and pass the original."""
    work = tempfile.mkdtemp(prefix="mvalid_"); rows = []
    v, au, dur, fps = _probe(good)
    cases = {}
    cases["good"] = good
    cases["black_head"] = os.path.join(work, "bh.mp4")
    _ff("-i", good, "-vf", "fade=t=in:st=0:d=0.01,drawbox=0:0:iw:ih:black:t=fill:enable='lt(t,0.4)'", "-c:a", "copy", cases["black_head"])
    cases["black_tail"] = os.path.join(work, "bt.mp4")
    _ff("-i", good, "-vf", "tpad=stop_mode=add:stop_duration=0.8:color=black", "-af", "apad=pad_dur=0.8", cases["black_tail"])
    cases["clipped_endcard"] = os.path.join(work, "ce.mp4")
    _ff("-i", good, "-vf", "pad=iw:ih:0:0:black,crop=iw:ih:0:0,drawbox=x=0:y=ih/2-300:w=26:h=600:color=black:t=fill:enable='gte(t,%.2f)'" % (dur - 2.5), "-c:a", "copy", cases["clipped_endcard"])
    cases["wrong_format"] = os.path.join(work, "wf.mp4")
    _ff("-i", good, "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:-1:-1", "-c:a", "copy", cases["wrong_format"])
    cases["no_audio"] = os.path.join(work, "na.mp4")
    _ff("-i", good, "-an", "-c:v", "copy", cases["no_audio"])
    exp = {"good": None, "black_head": "black_frames", "black_tail": "black_frames", "clipped_endcard": "endcard_clip",
           "wrong_format": "format", "no_audio": "format"}
    class A: pass
    for name, path in cases.items():
        a = A(); a.target = path; a.cutout = None; a.ugc = False
        mt = json.load(open(good + ".meta.json")); a.brand = mt["brand"]; a.product = mt["product"]; a.title_box = ",".join(str(x) for x in mt["title_box"])
        res = run_reel(a)
        fails = [r[0] for r in res if r[1] == "FAIL"]; warns = [r[0] for r in res if r[1] == "WARN"]
        want = exp[name]
        if name == "good": ok = not fails
        elif name == "no_audio": ok = "format" in warns or "format" in fails
        else: ok = want in fails
        rows.append((name, ok, fails, warns))
    # image cases
    im_cases = [("img_good_face", ANCHOR, True), ("img_no_face", None, False)]
    bag = MZ + "/05_cloud_library/products/cutout"
    nb = sorted(os.listdir(bag))[0]
    rows_img = []
    r = chk_identity(cv2.imread(ANCHOR)); rows_img.append(("anchor_identity_pass", r[1] == "PASS", r))
    r = chk_identity(cv2.imread(os.path.join(bag, nb), cv2.IMREAD_COLOR)); rows_img.append(("bag_has_no_face_fail", r[1] == "FAIL", r))
    r = chk_sheet(BODY_BAD); rows_img.append(("fullbody_sheet_fails", r[1] == "FAIL", r))
    r = chk_sheet(BODY_OK); rows_img.append(("covered_sheet_passes", r[1] == "PASS", r))
    for nm, ok, r in rows_img: rows.append((nm, ok, [r[0]] if r[1] == "FAIL" else [], [r[0]] if r[1] == "WARN" else []))
    # cutout gate on a real hollow vs solid asset from memory (pm black hollow, st_louis_gm solid) if present
    cdir = MZ + "/05_cloud_library/products/cutout"
    for pat, want in (("goyard_saint_louis_gm_black", True), ("goyard_saint_louis_pm_black", False)):
        m = [f for f in os.listdir(cdir) if pat in f] if os.path.isdir(cdir) else []
        if m:
            v_ = solidity(os.path.join(cdir, m[0])); rows.append(("cutout_" + pat, (v_ >= 0.8) == want, ["solidity %.2f" % v_], []))
    mt = json.load(open(good + ".meta.json"))
    a2 = A(); a2.target = good; a2.cutout = None; a2.ugc = True; a2.brand = mt["brand"]; a2.product = mt["product"]; a2.title_box = "70,130,1000,400"
    r2 = run_reel(a2); fl = [x[0] for x in r2 if x[1] == "FAIL"]
    rows.append(("title_over_face", "title_face" in fl, ["title_face"] if "title_face" in fl else [], []))
    rows.append(("ugc_no_wm_marker", "wm_marker" in fl, ["wm_marker"] if "wm_marker" in fl else [], []))
    hol = np.zeros((400, 400, 4), np.uint8); hol[..., :3] = 40; hol[..., 3] = 255; hol[100:300, 100:300, 3] = 0  # hollow bag body
    hp = os.path.join(work, "hollow.png"); cv2.imwrite(hp, hol)
    sol = hol.copy(); sol[..., 3] = 255; sp = os.path.join(work, "solid.png"); cv2.imwrite(sp, sol)
    rows.append(("cutout_hollow_fails", chk_cutout(hp)[1] == "FAIL", ["cutout_solid"], []))
    rows.append(("cutout_solid_passes", chk_cutout(sp)[1] == "PASS", [], []))
    bad = 0
    for n, ok, f, w in rows:
        print("%-26s %s  fails=%s warns=%s" % (n, "OK " if ok else "MISS", f, w)); bad += (not ok)
    print("SELFTEST %s (%d/%d)" % ("PASS" if not bad else "FAIL", len(rows) - bad, len(rows)))
    shutil.rmtree(work, ignore_errors=True)
    return 0 if not bad else 1

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    i = sp.add_parser("image"); i.add_argument("target"); i.add_argument("--pose", default="auto", choices=["auto", "front", "3q", "profile"])
    i.add_argument("--sheet"); i.add_argument("--source", default="gemini", choices=["gemini", "local"], help="gemini (default): floors from posted Gemini stills; local: Z-Image/Qwen pipeline floors")
    r = sp.add_parser("reel"); r.add_argument("target"); r.add_argument("--brand"); r.add_argument("--product"); r.add_argument("--cutout")
    r.add_argument("--ugc", action="store_true"); r.add_argument("--title-box")
    s = sp.add_parser("selftest"); s.add_argument("good_reel", nargs="?", default=os.path.join(HERE,"validator_fixtures","good_reel.mp4"))
    for p in (i, r):
        p.add_argument("--json", action="store_true"); p.add_argument("--strict", action="store_true", help="WARN also fails")
    a = ap.parse_args()
    if a.cmd == "selftest": sys.exit(selftest(a.good_reel))
    res = run_image(a) if a.cmd == "image" else run_reel(a)
    fail = any(x[1] == "FAIL" for x in res) or (a.strict and any(x[1] == "WARN" for x in res))
    if a.json:
        print(json.dumps({"target": a.target, "verdict": "FAIL" if fail else "PASS", "checks": [dict(zip(("check", "level", "detail"), x)) for x in res]}))
    else:
        for n, l, d in res: print("%-4s %-18s %s" % (l, n, d))
        print("VERDICT", "FAIL" if fail else "PASS")
    sys.exit(1 if fail else 0)

if __name__ == "__main__":
    main()
