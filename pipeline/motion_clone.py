#!/usr/bin/env python3
"""motion_clone.py JOB_DIR - full motion clone of driving.* onto still.* (2026-10-25).
Reads JOB_DIR/job.json {prompt, auto_improve, meher, quality, finish, seed}; writes JOB_DIR/status.json for the UI.
Steps: analyse (size/fps/audio/cuts, render size, plan) -> prompt (vision model if empty) -> per scene: chained
Wan-Animate windows (meher_animate_phased.py, 77 frames each, next window seeded by the last 5 frames) -> join ->
finish (original W x H, fps, exact duration, original audio) -> score (motion_score.py, video_qc).
One GPU job at a time: waits on gpu.lock and for any other ComfyUI (port 8188) to finish."""
import fcntl, json, math, os, re, shutil, subprocess, sys, time
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
COMFY = os.environ.get("COMFY_DIR") or os.path.dirname(HERE)          # env overrides = cloud (Kaggle) layout
PY = os.environ.get("COMFY_PY") or os.path.join(COMFY, ".venv", "bin", "python")
TOOLS = os.environ.get("MC_TOOLS") or os.path.expanduser("~/automation/shorts-factory/tools")
JARVIS = os.path.expanduser("~/.claude/mcp/jarvis-desktop")
SCORE = os.environ.get("MC_SCORE") or os.path.join(COMFY, "meher_test", "motion_score.py")
STATUS_HOOK = os.environ.get("MC_STATUS_HOOK")   # optional: called with status.json after every update (cloud live status)
YUNET = os.path.join(TOOLS, "..", "vendor", "facemetric", "face_detection_yunet_2023mar.onnx")
WIN, SEED_N, FPS_GEN = 77, 5, 16          # Animate window (4n+1), seed frames, generation fps
PIXELS = 576 * 1024                        # render budget that fits 8 GB VRAM / 15 GB RAM
MIN_SCENE = 1.0                            # scenes shorter than this (s) merge into the previous one
SEC_PER_FRAME = {"best": 1300 / WIN, "fast": 950 / WIN}   # measured r5 (6 steps) / r4 (4 steps) incl. pose+encode+decode

job = os.path.abspath(sys.argv[1]); J = lambda *p: os.path.join(job, *p)
cfg = json.load(open(J("job.json")))
ST = {"state": "queued", "steps": [], "log": [], "outputs": {}, "scores": {}, "plan": {}, "started": time.time()}
STEPS = ["Analyse", "Prompt", "Pose + generate", "Join", "Finish", "Score"]
ST["steps"] = [{"name": n, "state": "waiting", "detail": ""} for n in STEPS]

def save():
    tmp = J("status.json.tmp"); json.dump(ST, open(tmp, "w"), indent=1); os.replace(tmp, J("status.json"))
    if STATUS_HOOK: subprocess.Popen([STATUS_HOOK, J("status.json")])

def log(msg):
    ST["log"].append(time.strftime("%H:%M:%S ") + msg); ST["log"] = ST["log"][-200:]; save()

def step(i, state, detail=None):
    ST["steps"][i]["state"] = state
    if detail is not None: ST["steps"][i]["detail"] = detail
    save()

def run(cmd, logfile=None, **kw):
    with open(logfile or J("run.log"), "a") as f:
        f.write("\n$ " + " ".join(map(str, cmd)) + "\n"); f.flush()
        return subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, **kw).returncode

def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,r_frame_rate",
                          "-show_entries", "format=duration", "-of", "json", path], capture_output=True, text=True).stdout
    d = json.loads(out); v = next(s for s in d["streams"] if s["codec_type"] == "video")
    n, dd = v["r_frame_rate"].split("/")
    return dict(w=int(v["width"]), h=int(v["height"]), fps=float(n) / float(dd), dur=float(d["format"]["duration"]),
                audio=any(s["codec_type"] == "audio" for s in d["streams"]))

def ffmpeg(*args):
    if run(["ffmpeg", "-v", "error", "-y", *args]) != 0: raise RuntimeError("ffmpeg failed: " + " ".join(map(str, args))[:200])

def render_size(w, h):
    ar = w / h; rw = math.sqrt(PIXELS * ar); rh = rw / ar
    return int(round(rw / 16) * 16), int(round(rh / 16) * 16)

def crop_still(src, dst, W, H):
    """crop the still to the render aspect around the face, resize to W x H."""
    im = cv2.imread(src); h, w = im.shape[:2]; ar = W / H
    d = cv2.FaceDetectorYN.create(YUNET, "", (w, h), 0.6); _, f = d.detect(im)
    cx, cy = (w / 2, h / 3) if f is None else (lambda r: (r[0] + r[2] / 2, r[1] + r[3] / 2))(max(f, key=lambda r: r[2] * r[3]))
    if w / h > ar: cw, ch = int(h * ar), h
    else: cw, ch = w, int(w / ar)
    x0 = int(min(max(cx - cw / 2, 0), w - cw)); y0 = int(min(max(cy - ch / 3, 0), h - ch))
    cv2.imwrite(dst, cv2.resize(im[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_AREA))

def cuts(path):
    """hard cuts (s) - same detector as clip_qa.py: 15 fps grey frame diff > 6x median and > 6."""
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", "fps=15,scale=270:480,format=gray", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    V = np.frombuffer(raw, np.uint8); n = V.size // (270 * 480)
    if n < 3: return []
    V = V[:n * 270 * 480].reshape(n, 480, 270).astype(np.float32)
    df = np.array([np.abs(V[i] - V[i - 1]).mean() for i in range(1, n)]); med = np.median(df) + 1e-3
    return [round((i + 1) / 15, 2) for i in range(len(df)) if df[i] > 6 * med and df[i] > 6]

def scenes_from(cut_list, dur):
    edges = [0.0] + [c for c in cut_list if 0 < c < dur] + [dur]; sc = []
    for a, b in zip(edges, edges[1:]):
        if sc and (b - a < MIN_SCENE or sc[-1][1] - sc[-1][0] < MIN_SCENE): sc[-1][1] = b   # merge tiny pieces
        else: sc.append([a, b])
    return sc

def windows_for(n):
    """frames per window for a scene of n generated frames: [(frames, offset, pose_frames)]."""
    f1 = min(WIN, max(9, 4 * math.ceil((n - 1) / 4) + 1)); out = [(f1, 0, f1)]; covered = f1
    while covered < n:
        f = min(WIN, max(9, 4 * math.ceil((n - covered + SEED_N - 1) / 4) + 1))
        out.append((f, covered, covered - SEED_N + f)); covered += f - SEED_N
    return out

def vision(img_path, q):
    code = ("import sys,json;sys.path.insert(0,%r);import jarvis_look;"
            "print(json.dumps(jarvis_look.vision(open(%r,'rb').read(),%r)))") % (JARVIS, img_path, q)
    try:
        r = subprocess.run(["python3", "-c", code], capture_output=True, text=True, timeout=150)
        t = json.loads(r.stdout.strip().splitlines()[-1])
        return re.sub(r"\s*\[by [^\]]*\]\s*$", "", t).strip() if t and "busy" not in t else None
    except Exception:
        return None

def think(prompt):
    code = ("import sys,json;sys.path.insert(0,%r);import jarvis_think;r=jarvis_think.think(%r,max_tokens=300);"
            "print(json.dumps(r.get('text') if isinstance(r,dict) else r))") % (JARVIS, prompt)
    try:
        r = subprocess.run(["python3", "-c", code], capture_output=True, text=True, timeout=150)
        return json.loads(r.stdout.strip().splitlines()[-1]) or None
    except Exception:
        return None

def gpu_wait():
    lk = open(os.path.join(HERE, "gpu.lock"), "w")
    while True:
        try: fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB); break
        except BlockingIOError: pass
        ST["state"] = "queued"; save(); time.sleep(15)
    while subprocess.run(["bash", "-c", "ss -ltn | grep -q ':8188 ' || pgrep -f meher_animate_phased.py >/dev/null"]).returncode == 0:
        if ST["state"] != "waiting for GPU": ST["state"] = "waiting for GPU"; log("another GPU job is running - waiting")
        time.sleep(20)
    return lk

try:
    drv = next(J(f) for f in os.listdir(job) if f.startswith("driving."))
    still = next(J(f) for f in os.listdir(job) if f.startswith("still."))
    save(); lock = gpu_wait(); ST["state"] = "running"; save()

    # 1 analyse
    step(0, "running"); info = probe(drv); W, H = render_size(info["w"], info["h"])
    sc = scenes_from(cuts(drv), info["dur"])
    plan = []
    for a, b in sc:
        n = max(9, round((b - a) * FPS_GEN)); plan.append(dict(start=a, end=b, n=n, windows=windows_for(n)))
    nwin = sum(len(p["windows"]) for p in plan)
    spf = SEC_PER_FRAME[cfg.get("quality", "best")] * (W * H / PIXELS)
    est = sum(f for p in plan for f, _, _ in p["windows"]) * spf
    ST["plan"] = dict(src=info, render=[W, H], scenes=[[round(p["start"], 2), round(p["end"], 2)] for p in plan],
                      windows=nwin, estimate_min=round(est / 60))
    crop_still(still, J("still_crop.png"), W, H)
    step(0, "done", f"{info['w']}x{info['h']} {info['fps']:.0f} fps {info['dur']:.1f} s, {len(plan)} scene(s), "
                    f"{nwin} window(s), render {W}x{H}, ~{round(est / 60)} min")
    log(ST["steps"][0]["detail"])

    # 2 prompt
    step(1, "running"); user = (cfg.get("prompt") or "").strip(); age = "24 years old " if cfg.get("meher", True) else ""
    if user and cfg.get("auto_improve"):
        better = think("Rewrite this as ONE short video-generation prompt (max 60 words) for a photoreal phone video. "
                       "Keep every fact, add nothing new except natural camera-facing realism; no lists, no quotes:\n" + user)
        prompt = better.strip().strip('"') if better else user; src = "yours, improved" if better else "yours"
    elif user:
        prompt, src = user, "yours"
    else:
        mid = J("drv_mid.png"); ffmpeg("-ss", f"{info['dur'] / 2:.2f}", "-i", drv, "-frames:v", "1", mid)
        look = vision(J("still_crop.png"), "In one short sentence describe only this person's clothes, hair and the "
                      "place/background. No opinions.") or "same clothes, same hair, same background"
        act = vision(mid, "In one short sentence: what is the person doing (action, gestures, facial expression, "
                     "camera distance)? No description of the person's looks.") or "moves naturally and talks to the camera"
        prompt = (f"A {age}woman with a youthful mid-twenties face, {act.rstrip('.')}. Same look as the reference image: "
                  f"{look.rstrip('.')}. Constant exposure, natural light.") if age else \
                 f"A person {act.rstrip('.')}. Same look as the reference image: {look.rstrip('.')}. Constant exposure."
        src = "auto (vision model)"
    if age and "24 years old" not in prompt: prompt = "A 24 years old woman. " + prompt
    ST["plan"]["prompt"] = prompt; step(1, "done", src); log("prompt: " + prompt)

    # 3 generate
    step(2, "running"); steps, lora = (6, 1.0) if cfg.get("quality", "best") == "best" else (4, 1.2)
    seed = int(cfg.get("seed") or 7); parts = []; k = 0; t0 = time.time()
    for si, p in enumerate(plan):
        seg = J(f"scene{si}.mp4")
        ffmpeg("-ss", f"{p['start']:.3f}", "-i", drv, "-t", f"{p['end'] - p['start']:.3f}", "-an", "-c:v", "libx264",
               "-crf", "14", "-pix_fmt", "yuv420p", seg)
        prev = None; wins = []
        for wi, (f, off, pf) in enumerate(p["windows"]):
            k += 1; pre = f"mc_{os.path.basename(job)}_s{si}w{wi}"
            step(2, "running", f"window {k} of {nwin} (scene {si + 1}, {f} frames)"); log(f"generating window {k}/{nwin}")
            cmd = [PY, os.path.join(COMFY, "meher_animate_phased.py"), J("still_crop.png"), seg, prompt, "--prefix", pre,
                   "--width", W, "--height", H, "--frames", f, "--steps", steps, "--lora", lora, "--seed", seed]
            if prev: cmd += ["--pose-frames", pf, "--offset", off, "--continue-video", prev]
            wl = J(f"{pre}.log")
            if run(cmd, logfile=wl, cwd=COMFY) != 0: raise RuntimeError(f"window {k} failed, see {wl}")
            out = re.findall(r"(/\S*" + re.escape(pre) + r"_0\S*\.mp4) \(decode\)", open(wl).read())
            if not out: raise RuntimeError(f"window {k}: no video in log {wl}")
            prev = out[-1]; wins.append(prev)
            el = time.time() - t0; ST["plan"]["eta_min"] = round(el / k * (nwin - k) / 60); save()
        parts.append(wins)
    step(2, "done", f"{nwin} window(s) in {round((time.time() - t0) / 60)} min")

    # 4 join: drop the 5 seeded frames (10 at 32 fps) from every chained window, concat everything
    # each scene is trimmed to its exact source length so later scenes stay in sync with the original audio
    step(3, "running"); ins, fl, sc_labels = [], [], []
    for si, wins in enumerate(parts):
        labs = []
        for wi, v in enumerate(wins):
            i = len(ins) // 2; ins += ["-i", v]
            fl.append(f"[{i}:v]" + ("trim=start_frame=10,setpts=PTS-STARTPTS," if wi else "") + f"scale={W}:{H},setsar=1,fps=32[v{i}]")
            labs.append(f"[v{i}]")
        d = plan[si]["end"] - plan[si]["start"]
        fl.append("".join(labs) + f"concat=n={len(labs)}:v=1,tpad=stop_mode=clone:stop_duration=1,trim=duration={d:.4f},setpts=PTS-STARTPTS[s{si}]")
        sc_labels.append(f"[s{si}]")
    n = len(ins) // 2; joined = J("joined.mp4")
    fc = ";".join(fl) + ";" + "".join(sc_labels) + f"concat=n={len(sc_labels)}:v=1[v]"
    ffmpeg(*ins, "-filter_complex", fc, "-map", "[v]", "-c:v", "libx264", "-crf", "15", "-preset", "slow", "-bf", "0",
           "-pix_fmt", "yuv420p", joined)
    step(3, "done", f"{n} part(s)")

    # 5 finish: original size, fps, exact length, original audio (+ optional phone finish)
    step(4, "running"); vf = f"scale={info['w']}:{info['h']}:flags=lanczos,fps={info['fps']:.3f},tpad=stop_mode=clone:stop_duration=2"
    if cfg.get("finish", True): vf += ",unsharp=5:5:0.5:5:5:0.0,noise=alls=3:allf=t"
    final = J("final.mp4")
    a = ["-i", drv, "-map", "1:a:0", "-c:a", "aac", "-b:a", "160k"] if info["audio"] else []
    ffmpeg("-i", joined, *a[:2], "-filter_complex", f"[0:v]{vf}[v]", "-map", "[v]", *a[2:], "-t", f"{info['dur']:.3f}",
           "-c:v", "libx264", "-crf", "17", "-preset", "slow", "-bf", "0", "-pix_fmt", "yuv420p", "-movflags", "+faststart", final)
    ffmpeg("-i", drv, "-i", final, "-filter_complex",
           f"[0:v]fps=30,scale=-2:960[a];[1:v]fps=30,scale=-2:960[b];[a][b]hstack", "-an", "-t", f"{info['dur']:.3f}",
           "-c:v", "libx264", "-crf", "21", "-bf", "0", "-pix_fmt", "yuv420p", "-movflags", "+faststart", J("compare.mp4"))
    ST["outputs"] = {"final": "final.mp4", "compare": "compare.mp4"}
    fi = probe(final); step(4, "done", f"{fi['w']}x{fi['h']} {fi['fps']:.0f} fps {fi['dur']:.2f} s" + (" + audio" if fi["audio"] else ""))

    # 6 score (face match vs the Meher anchor, or vs the uploaded still for any other character)
    step(5, "running")
    if not cfg.get("meher", True): os.environ["MEHER_ANCHOR"] = J("still_crop.png")
    qc = subprocess.run(["python3", os.path.join(TOOLS, "video_qc.py"), final], capture_output=True, text=True).stdout.strip().splitlines()
    sc_out = subprocess.run(["python3", SCORE, final, drv, "--n", str(min(24, max(6, int(info["dur"] * 2))))],
                            capture_output=True, text=True).stdout
    g = lambda pat: (lambda m: float(m.group(1)) if m else None)(re.search(pat, sc_out))
    ST["scores"] = {"face": g(r"id\s+mean ([\d.]+)"), "face_min": g(r"min ([\d.]+)"), "motion": g(r"track corr x ([+\-\d.]+)"),
                    "head": g(r"head corr roll [+\-\d.]+ yaw ([+\-\d.]+)"), "mouth": g(r"mouth corr width ([+\-\d.nan]+)"),
                    "qc": "pass" if qc and "PASS" in qc[-1] else "fail"}
    open(J("score.txt"), "w").write(sc_out + "\n" + (qc[-1] if qc else ""))
    step(5, "done", f"face {ST['scores']['face']} motion {ST['scores']['motion']} qc {ST['scores']['qc']}")
    ST["state"] = "done"; ST["finished"] = time.time(); log("done: final.mp4")
except Exception as e:
    for s in ST["steps"]:
        if s["state"] == "running": s["state"] = "failed"
    ST["state"] = "failed"; ST["error"] = str(e)[:500]; log("FAILED: " + str(e)[:300])
