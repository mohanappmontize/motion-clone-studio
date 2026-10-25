# Motion clone studio - Kaggle GPU worker (2026-10-25).
# The relay pushes this file as a new kernel version with JOB filled in. It rebuilds the exact laptop setup
# (same ComfyUI + node commits, same model files from Hugging Face), runs pipeline/motion_clone.py on the job's
# inputs, posts live status to ntfy.sh/<topic> and leaves final.mp4 / compare.mp4 / status.json in /kaggle/working/out.
import json, os, subprocess, sys, threading, time, urllib.request

JOB = {}  # __JOB__ (relay replaces this line)

W = "/kaggle/working"; T = "/kaggle/temp" if os.path.isdir("/kaggle/temp") else "/tmp"
REPO = "https://github.com/mohanappmontize/motion-clone-studio"
COMFY_COMMIT = "bc5f880dbdd43e426077f42ccec53bb577ffee9f"
NODES = {
    "ComfyUI-GGUF": ("https://github.com/city96/ComfyUI-GGUF", "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"),
    "ComfyUI-VideoHelperSuite": ("https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite", "4d907bee61e92c2e65af3bd6383a4e4d356126d1"),
    "ComfyUI-WanAnimatePreprocess": ("https://github.com/kijai/ComfyUI-WanAnimatePreprocess", "0e0b6a2a555625acf4d4aefb780e27d06937132f"),
    "ComfyUI-Frame-Interpolation": ("https://github.com/Fannovel16/ComfyUI-Frame-Interpolation", "26545cc2dd95bc3d27f056016300673bdeee78f5"),
}
HF = "https://huggingface.co/"
MODELS = [  # (dest under models/, url, bytes) - byte-identical to the laptop files
    ("unet/Wan2.2-Animate-14B-Q4_K_S.gguf", "QuantStack/Wan2.2-Animate-14B-GGUF/resolve/main/Wan2.2-Animate-14B-Q4_K_S.gguf", 10592753472),
    ("text_encoders/umt5-xxl-encoder-Q5_K_M.gguf", "city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q5_K_M.gguf", 4145878880),
    ("clip_vision/clip_vision_h.safetensors", "Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors", 1264219396),
    ("vae/wan_2.1_vae.safetensors", "Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors", 253815318),
    ("loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", 738005744),
    ("detection/vitpose_h_wholebody_model.onnx", "Kijai/vitpose_comfy/resolve/main/onnx/vitpose_h_wholebody_model.onnx", 420252),
    ("detection/vitpose_h_wholebody_data.bin", "Kijai/vitpose_comfy/resolve/main/onnx/vitpose_h_wholebody_data.bin", 2548958740),
    ("detection/yolov10m.onnx", "Wan-AI/Wan2.2-Animate-14B/resolve/main/process_checkpoint/det/yolov10m.onnx", 61659339),
]
OUT = os.path.join(W, "out"); os.makedirs(OUT, exist_ok=True)
STAT = {"state": "setting up", "steps": [], "log": [], "plan": {}, "worker": "kaggle"}

def post():
    """live status -> ntfy topic (the page polls it); also kept in out/status.json"""
    json.dump(STAT, open(os.path.join(OUT, "status.json"), "w"))
    if not JOB.get("topic"): return
    msg = dict(STAT); msg["log"] = STAT.get("log", [])[-12:]
    try:
        urllib.request.urlopen(urllib.request.Request("https://ntfy.sh/" + JOB["topic"], data=json.dumps(msg).encode()[:3900],
                                                      method="POST"), timeout=15)
    except Exception as e:
        print("ntfy post failed", e, flush=True)

def say(m):
    print(m, flush=True); STAT["log"].append(time.strftime("%H:%M:%S ") + m); STAT["log"] = STAT["log"][-60:]; post()

def sh(c, cwd=None):
    print("+", c, flush=True); r = subprocess.run(c, shell=True, cwd=cwd)
    if r.returncode: raise RuntimeError(f"failed ({r.returncode}): {c[:200]}")

try:
    t0 = time.time(); say(f"worker started, job {JOB.get('id')}")
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; free -g | head -2")
    C = os.path.join(T, "ComfyUI")
    sh(f"git clone -q https://github.com/comfyanonymous/ComfyUI {C} && git -C {C} checkout -q {COMFY_COMMIT}")
    for n, (u, c) in NODES.items():
        sh(f"git clone -q {u} {C}/custom_nodes/{n} && git -C {C}/custom_nodes/{n} checkout -q {c}")
    P = os.path.join(T, "mcs"); sh(f"git clone -q --depth 1 {REPO} {P}")
    sh(f"cp -r {P}/pipeline/phase_io {C}/custom_nodes/phase_io && cp {P}/pipeline/meher_animate_phased.py {P}/pipeline/phase_runner.py {C}/")
    say("installing packages")
    # keep Kaggle's preinstalled CUDA torch; install the rest
    req = [l for l in open(f"{C}/requirements.txt").read().splitlines() if l.strip() and not l.startswith(("torch", "#"))]
    sh("pip -q install " + " ".join(f"'{r}'" for r in req) + " gguf onnxruntime-gpu imageio-ffmpeg opencv-python-headless")
    for n in NODES:
        rq = f"{C}/custom_nodes/{n}/requirements" + ("-no-cupy.txt" if n == "ComfyUI-Frame-Interpolation" else ".txt")
        if os.path.exists(rq): sh(f"pip -q install -r {rq} || true")
    say("downloading models (16 GB)")
    procs = []
    for dest, url, size in MODELS:
        p = f"{C}/models/{dest}"; os.makedirs(os.path.dirname(p), exist_ok=True)
        procs.append((subprocess.Popen(f"curl -sSL --retry 5 -o '{p}' '{HF + url}'", shell=True), p, size))
    for pr, p, size in procs:
        pr.wait()
        if os.path.getsize(p) != size: raise RuntimeError(f"bad download {p}: {os.path.getsize(p)} != {size}")
    say(f"setup done in {round((time.time() - t0) / 60)} min")

    jd = os.path.join(T, "job"); os.makedirs(jd, exist_ok=True)
    for key, base in (("video_url", "driving"), ("image_url", "still")):
        ext = os.path.splitext(JOB[key].split("?")[0])[1] or (".mp4" if base == "driving" else ".png")
        sh(f"curl -sSL --retry 5 -o '{jd}/{base}{ext}' '{JOB[key]}'")
    json.dump(JOB.get("cfg", {}), open(os.path.join(jd, "job.json"), "w"))

    env = dict(os.environ, COMFY_DIR=C, COMFY_PY=sys.executable, MC_TOOLS=f"{P}/pipeline/tools",
               MC_SCORE=f"{P}/pipeline/motion_score.py", MEHER_ANCHOR=f"{P}/pipeline/refs/ANCHOR_face.png", NO_SYSTEMD="1")
    pr = subprocess.Popen([sys.executable, f"{P}/pipeline/motion_clone.py", jd], env=env)
    last = None
    while pr.poll() is None:      # mirror the pipeline's status to ntfy every 20 s
        time.sleep(20)
        try: s = json.load(open(os.path.join(jd, "status.json")))
        except Exception: continue
        if s != last: last = s; s["worker"] = "kaggle"; STAT.clear(); STAT.update(s); post()
    s = json.load(open(os.path.join(jd, "status.json"))); s["worker"] = "kaggle"; STAT.clear(); STAT.update(s)
    for f in ("final.mp4", "compare.mp4", "score.txt", "status.json", "run.log"):
        if os.path.exists(os.path.join(jd, f)): sh(f"cp '{jd}/{f}' '{OUT}/'")
    sh(f"cp {jd}/*.log {OUT}/ 2>/dev/null || true")
    STAT["total_min"] = round((time.time() - t0) / 60); post()
except Exception as e:
    STAT["state"] = "failed"; STAT["error"] = str(e)[:500]; say("FAILED: " + str(e)[:300]); raise
