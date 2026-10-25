"""run_phase(): run ONE ComfyUI workflow in its own memory-capped ComfyUI process, then exit that process.

Shared by meher_i2v_phased.py and meher_animate_phased.py. Process exit is the only reliable way to give RAM
back (feedback_gpu_ram_dos_donts), so each phase's peak is its own. Over the cap = only this phase is SIGKILLed.
Peak RAM/swap come from the phase's own cgroup (systemd unit name needs the .scope suffix, or you read the
whole machine's counter: that bug reported 24.5 GB for a 10 GB-capped phase on 2026-10-25).
"""
import json, os, shutil, subprocess, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get("COMFY_PY") or os.path.join(HERE, ".venv/bin/python")
CAPPED = shutil.which("systemd-run") is not None and not os.environ.get("NO_SYSTEMD")   # cloud boxes (Kaggle) have no systemd
BASE = "http://127.0.0.1:8188"


def _cg(unit, name):
    if not CAPPED: return -1
    cg = subprocess.run(["systemctl", "--user", "show", "-p", "ControlGroup", "--value", unit + ".scope"],
                        capture_output=True, text=True).stdout.strip()
    try:
        return int(open(f"/sys/fs/cgroup{cg}/{name}").read())
    except (OSError, ValueError):
        return -1


def run_phase(name, wf, run_dir, cap, swap="6G", tag="comfy"):
    """returns the list of VHS outputs ('gifs') of the workflow; raises SystemExit with the reason on failure"""
    unit = f"{tag}-{name}-{os.getpid()}"
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    log = open(os.path.join(run_dir, f"{name}.log"), "w")
    pre = ["systemd-run", "--user", "--scope", "--quiet", f"--unit={unit}",
           "-p", f"MemoryMax={cap}", "-p", f"MemorySwapMax={swap}"] if CAPPED else []
    srv = subprocess.Popen(pre + [PY, "main.py", "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch",
                            "--cache-none"], cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
    open(os.path.join(run_dir, f"{name}.pid"), "w").write(str(srv.pid))
    t0, peak, speak = time.time(), 0, 0
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(BASE + "/system_stats"); break
            except OSError:
                if srv.poll() is not None:
                    raise SystemExit(f"{name}: server exited during startup, see {run_dir}/{name}.log")
                time.sleep(2)
        req = urllib.request.Request(BASE + "/prompt", data=json.dumps({"prompt": wf}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            pid = json.load(urllib.request.urlopen(req))["prompt_id"]
        except urllib.error.HTTPError as e:
            raise SystemExit(f"{name}: workflow rejected: {e.read().decode()[:2000]}")
        while True:
            time.sleep(5)
            peak = max(peak, _cg(unit, "memory.peak")); speak = max(speak, _cg(unit, "memory.swap.peak"))
            if srv.poll() is not None:
                raise SystemExit(f"{name}: KILLED (cap {cap} RAM / {swap} swap hit?) peak RAM {peak/1e9:.1f} GB")
            try:
                h = json.load(urllib.request.urlopen(f"{BASE}/history/{pid}"))
            except OSError:
                continue
            if pid in h:
                st = h[pid].get("status", {})
                if st.get("status_str") == "error":
                    msgs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
                    raise SystemExit(f"{name}: error {json.dumps(msgs)[:1500]}")
                return [g for o in h[pid]["outputs"].values() for g in o.get("gifs", [])]
    finally:
        print(f"phase {name:7s} {time.time()-t0:5.0f}s  peak RAM {peak/1e9:4.1f} GB  peak swap {speak/1e9:4.1f} GB "
              f"(cap {cap})", flush=True)
        srv.terminate()
        try: srv.wait(30)
        except subprocess.TimeoutExpired: srv.kill(); srv.wait()
        if CAPPED: subprocess.run(["systemctl", "--user", "stop", unit + ".scope"], capture_output=True)
