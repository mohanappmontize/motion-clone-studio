#!/usr/bin/env python3
"""Motion clone studio relay (2026-10-25): GitHub issue (label `job`) -> Kaggle GPU worker -> GitHub release.
Run every minute (systemd timer on the laptop, or the GitHub Action when KAGGLE secrets exist). One job at a time.
Needs: `gh` logged in, `kaggle` logged in. State lives in relay/state.json next to this file (not committed).
Per job: inputs re-hosted on release `job-<n>` (public, stable URLs), prompt written by a vision model when empty
(JARVIS free models on the laptop; template otherwise), worker pushed as a new kernel version, live status on ntfy,
outputs uploaded to the same release, result commented on the issue, issue closed."""
import json, os, re, shutil, subprocess, sys, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
CONF = json.load(open(os.path.join(ROOT, "docs", "config.json")))
REPO, KUSER, KSLUG, SALT = CONF["repo"], CONF["kaggle_user"], CONF["kernel_slug"], CONF["topic_salt"]
KAGGLE = shutil.which("kaggle") or os.path.expanduser("~/.local/bin/kaggle")
STATE = os.path.join(HERE, "state.json")
JARVIS = os.path.expanduser("~/.claude/mcp/jarvis-desktop")
st = json.load(open(STATE)) if os.path.exists(STATE) else {"jobs": {}}

def save(): json.dump(st, open(STATE, "w"), indent=1)
def log(m): print(time.strftime("%H:%M:%S ") + m, flush=True)
def gh(*a, ok=False):
    r = subprocess.run(["gh", *a], capture_output=True, text=True)
    if r.returncode and not ok: raise RuntimeError("gh " + " ".join(a[:3]) + ": " + r.stderr[:300])
    return r.stdout
def kg(*a):
    return subprocess.run([KAGGLE, *a], capture_output=True, text=True)
def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 5))   # after our own comments
def topic(n): return f"mcs-{SALT}-{n}"
def comment(n, body): gh("issue", "comment", str(n), "-R", REPO, "-b", body)

def field(body, name):
    m = re.search(r"###\s*" + re.escape(name) + r"\s*\n(.*?)(?=\n###\s|\Z)", body, re.S)
    v = (m.group(1).strip() if m else "")
    return "" if v in ("_No response_", "None") else v

def first_url(text):
    m = re.search(r"(https://github\.com/user-attachments/(?:assets|files)/[^\s)\"'>]+|https?://[^\s)\"'>]+)", text)
    return m.group(1) if m else None

def download(url, dst):
    tok = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True).stdout.strip()
    req = urllib.request.Request(url, headers={"Authorization": f"token {tok}"} if "github.com" in url else {})
    with urllib.request.urlopen(req, timeout=300) as r, open(dst, "wb") as f: shutil.copyfileobj(r, f)
    return r.headers.get("Content-Type", "")

def vision(img, q):
    code = ("import sys,json;sys.path.insert(0,%r);import jarvis_look;"
            "print(json.dumps(jarvis_look.vision(open(%r,'rb').read(),%r)))") % (JARVIS, img, q)
    try:
        r = subprocess.run(["python3", "-c", code], capture_output=True, text=True, timeout=150)
        t = json.loads(r.stdout.strip().splitlines()[-1])
        return re.sub(r"\s*\[by [^\]]*\]\s*$", "", t).strip() if t and "busy" not in t else None
    except Exception:
        return None

def auto_prompt(still, drv, meher):
    mid = still + "_mid.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "1.5", "-i", drv, "-frames:v", "1", mid])
    look = vision(still, "In one short sentence describe only this person's clothes, hair and the place/background. "
                  "No opinions.") or "same clothes, same hair, same background"
    act = vision(mid, "In one short sentence: what is the person doing (action, gestures, facial expression, camera "
                 "distance)? No description of the person's looks.") or "moves naturally and talks to the camera"
    who = "A 24 years old woman with a youthful mid-twenties face" if meher else "A person"
    return f"{who}, {act.rstrip('.')}. Same look as the reference image: {look.rstrip('.')}. Constant exposure, natural light."

def start(n, issue):
    body = issue["body"] or ""
    v_url, i_url = first_url(field(body, "Driving video")), first_url(field(body, "Character image"))
    if not v_url or not i_url:
        comment(n, "Couldn't find both files. Drag a video into **Driving video** and an image into **Character image**, "
                   "then edit the issue - it is picked up again automatically."); st["jobs"][str(n)] = {"state": "needs files"}; return
    opts = "\n".join(l for l in field(body, "Options").lower().splitlines() if "[x]" in l)   # checked boxes only
    cfg = {"prompt": field(body, "Prompt (optional)"), "auto_improve": False, "meher": "meher canon" in opts,
           "finish": "phone finish" in opts, "quality": "fast" if "fast" in opts else "best",
           "seed": int(re.sub(r"\D", "", field(body, "Seed")) or 7)}
    d = tempfile.mkdtemp(prefix=f"mcs{n}_")
    ct_v = download(v_url, os.path.join(d, "v")); ct_i = download(i_url, os.path.join(d, "i"))
    ext_v = ".mov" if "quicktime" in ct_v else ".webm" if "webm" in ct_v else ".mp4"
    ext_i = ".jpg" if "jpeg" in ct_i else ".webp" if "webp" in ct_i else ".png"
    vid, img = os.path.join(d, "driving" + ext_v), os.path.join(d, "still" + ext_i)
    os.rename(os.path.join(d, "v"), vid); os.rename(os.path.join(d, "i"), img)
    if not cfg["prompt"]: cfg["prompt"] = auto_prompt(img, vid, cfg["meher"])
    tag = f"job-{n}"
    gh("release", "create", tag, "-R", REPO, "--title", f"Job #{n}: {issue['title'][:60]}", "--notes",
       f"Inputs and output of motion clone job #{n}.", vid, img, ok=True)
    gh("release", "upload", tag, "-R", REPO, vid, img, "--clobber")
    base = f"https://github.com/{REPO}/releases/download/{tag}/"
    job = {"id": n, "topic": topic(n), "video_url": base + os.path.basename(vid), "image_url": base + os.path.basename(img), "cfg": cfg}
    kd = tempfile.mkdtemp(prefix="mcsk_"); src = open(os.path.join(ROOT, "kaggle", "worker.py")).read()
    src = src.replace("JOB = {}  # __JOB__ (relay replaces this line)", "JOB = " + json.dumps(job))
    open(os.path.join(kd, "worker.py"), "w").write(src)
    json.dump({"id": f"{KUSER}/{KSLUG}", "title": KSLUG, "code_file": "worker.py", "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
               "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "kernel_sources": [], "competition_sources": []},
              open(os.path.join(kd, "kernel-metadata.json"), "w"))
    r = kg("kernels", "push", "-p", kd)
    if r.returncode: raise RuntimeError("kaggle push: " + (r.stdout + r.stderr)[-400:])
    st["jobs"][str(n)] = {"state": "running", "pushed": time.time(), "cfg": cfg}; save()
    comment(n, f"Started on Kaggle GPU (T4). Live progress: https://{REPO.split('/')[0]}.github.io/{REPO.split('/')[1]}/#job-{n}\n\n"
               f"Prompt used: {cfg['prompt']}")
    log(f"job {n} pushed")

def kstatus():
    r = kg("kernels", "status", f"{KUSER}/{KSLUG}"); t = (r.stdout + r.stderr).lower()
    for s in ("complete", "error", "cancel", "running", "queued"):
        if s in t: return s
    return "unknown"

def finish(n, ks):
    d = tempfile.mkdtemp(prefix=f"mcso{n}_"); kg("kernels", "output", f"{KUSER}/{KSLUG}", "-p", d)
    out = os.path.join(d, "out") if os.path.isdir(os.path.join(d, "out")) else d
    sj = os.path.join(out, "status.json"); s = json.load(open(sj)) if os.path.exists(sj) else {}
    files = [os.path.join(out, f) for f in ("final.mp4", "compare.mp4", "status.json", "score.txt") if os.path.exists(os.path.join(out, f))]
    if files: gh("release", "upload", f"job-{n}", "-R", REPO, *files, "--clobber")
    base = f"https://github.com/{REPO}/releases/download/job-{n}/"
    if s.get("state") == "done":
        sc = s.get("scores", {})
        comment(n, f"Done in {s.get('total_min', '?')} min.\n\n- Video: {base}final.mp4\n- Side by side: {base}compare.mp4\n"
                   f"- Face match {sc.get('face')} (min {sc.get('face_min')}), motion {sc.get('motion')}, mouth {sc.get('mouth')}, QC {sc.get('qc')}")
        gh("issue", "close", str(n), "-R", REPO); st["jobs"][str(n)]["state"] = "done"
    else:
        tail = "\n".join((s.get("log") or [])[-8:])
        comment(n, f"Failed on Kaggle ({ks}). {s.get('error', '')}\n```\n{tail}\n```\nEdit the issue or comment `retry` to run it again.")
        st["jobs"][str(n)]["state"] = "failed"
    st["jobs"][str(n)]["seen"] = now(); save(); log(f"job {n} {st['jobs'][str(n)]['state']}")

running = [n for n, j in st["jobs"].items() if j.get("state") == "running"]
if running:
    n = running[0]; ks = kstatus()
    if ks in ("complete", "error", "cancel"): finish(int(n), ks)
    elif time.time() - st["jobs"][n]["pushed"] > 12.5 * 3600: finish(int(n), "timeout")
    else: log(f"job {n} {ks}"); sys.exit()
issues = json.loads(gh("issue", "list", "-R", REPO, "--label", "job", "--state", "open", "--json", "number,title,body,updatedAt,comments", "--limit", "50"))
for issue in sorted(issues, key=lambda i: i["number"]):
    n = str(issue["number"]); j = st["jobs"].get(n)
    retry = j and any(c["body"].strip().lower() == "retry" for c in issue.get("comments", [])[-1:])
    if j and j["state"] in ("needs files", "failed") and (issue["updatedAt"] > j.get("seen", "") or retry):
        j = None
    if j is None:
        st["jobs"][n] = {"state": "starting"}; save()
        try: start(int(n), issue)
        except Exception as e:
            st["jobs"][n] = {"state": "failed"}
            comment(int(n), f"Couldn't start: {str(e)[:400]}"); log(f"job {n} start failed: {e}")
        st["jobs"][n]["seen"] = now(); save()
        break   # one job at a time
