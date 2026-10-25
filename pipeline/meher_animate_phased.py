"""Motion copy with Wan 2.2 Animate 14B (Q4_K_S GGUF): a real driving video's body pose + facial expression move a
gate-passing Meher still. Animate mode only (her look and background come from the still; the driving person's
face, body and room are never used). Each phase is its own memory-capped ComfyUI process (phase_runner.py):

  pose    : YOLOv10 + ViTPose-H wholebody on the driving clip, retargeted to Meher's proportions -> pose + face crops
  encode  : umt5 Q5 text, CLIP-vision of the still, WanAnimateToVideo conditioning             -> cond, latent, trim
  sample  : Animate 14B + LightX2V 4-step (1.2, per Kijai's example), lcm, cfg 1                -> latent
  decode  : tiled VAE decode + RIFE 16->32 fps                                                   -> mp4

Usage: python meher_animate_phased.py still.png driving.mp4 "prompt" [--frames 77 --skip 0 --seed 7 --prefix x]
"""
import argparse, os, shutil, time, uuid
from phase_runner import run_phase

HERE = os.path.dirname(os.path.abspath(__file__))
NEG = ("3D render, CGI, cartoon, anime, plastic skin, airbrushed, waxy skin, doll, blurry face, melted face, "
       "smeared features, morphing face, different person, older woman, deformed hands, extra fingers, fused fingers, "
       "missing fingers, extra limbs, broken arms, twisted wrists, warped body, changing clothes, flicker, "
       "objects appearing, objects vanishing, watermark, text, subtitles")
REAL = (" Shot on iPhone, natural skin texture with visible pores, sharp detailed face, anatomically correct hands "
        "with five fingers, real fabric, natural true to life colours, soft muted contrast, iso 800 grain.")
CAPS = {"pose": "10G", "encode": "12G", "sample": "18G", "decode": "12G"}

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("driving")
ap.add_argument("prompt")
ap.add_argument("--width", type=int, default=576)
ap.add_argument("--height", type=int, default=1024)
ap.add_argument("--frames", type=int, default=77)  # one Animate window (4n+1); 16 fps -> 4.8 s
ap.add_argument("--skip", type=int, default=0, help="driving-video frames to skip (at its native rate)")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--prefix", default="meher_animate")
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--pose-frames", type=int, default=0, help="driving frames to read for pose (default = --frames); window 2 needs frames+offset")
ap.add_argument("--offset", type=int, default=0, help="video_frame_offset: start of this window in pose frames (window 2 = 77)")
ap.add_argument("--continue-video", default="", help="previous window's 32 fps RIFE mp4; its last 5 original frames seed this window")
ap.add_argument("--lora", type=float, default=1.2, help="LightX2V strength")
a = ap.parse_args()

run = os.path.join(HERE, "phase_runs", f"{a.prefix}_{time.strftime('%m%d_%H%M%S')}")
os.makedirs(run)
P = lambda n: os.path.join(run, n)
img = f"{a.prefix}_{uuid.uuid4().hex[:6]}{os.path.splitext(a.image)[1]}"
shutil.copy(a.image, os.path.join(HERE, "input", img))
W, H = a.width, a.height
load = lambda key, path: {key: {"class_type": "PhaseLoadAny", "inputs": {"path": P(path)}}}

phases = {
    "pose": {
        "vid": {"class_type": "VHS_LoadVideoPath", "inputs": {"video": os.path.abspath(a.driving), "force_rate": 16,
                "custom_width": 0, "custom_height": 0, "frame_load_cap": a.pose_frames or a.frames, "skip_first_frames": a.skip,
                "select_every_nth": 1}},
        "ref": {"class_type": "LoadImage", "inputs": {"image": img}},
        "det": {"class_type": "OnnxDetectionModelLoader", "inputs": {"vitpose_model": "vitpose_h_wholebody_model.onnx",
                "yolo_model": "yolov10m.onnx", "onnx_device": "CUDAExecutionProvider"}},
        "pf": {"class_type": "PoseAndFaceDetection", "inputs": {"model": ["det", 0], "images": ["vid", 0],
               "retarget_image": ["ref", 0], "width": W, "height": H}},
        "draw": {"class_type": "DrawViTPose", "inputs": {"pose_data": ["pf", 0], "width": W, "height": H,
                 "retarget_padding": 16, "body_stick_width": -1, "hand_stick_width": -1, "draw_head": True}},
        "s1": {"class_type": "PhaseSaveAny", "inputs": {"value": ["draw", 0], "path": P("pose.pt")}},
        "s2": {"class_type": "PhaseSaveAny", "inputs": {"value": ["pf", 1], "path": P("face.pt")}},
        "pv": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["draw", 0], "frame_rate": 16, "loop_count": 0,
               "filename_prefix": f"{a.prefix}_pose", "format": "video/h264-mp4", "pingpong": False, "save_output": True}}},
    "encode": {
        "t": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": "umt5-xxl-encoder-Q5_K_M.gguf", "type": "wan"}},
        "v": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "cv": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}},
        "ref": {"class_type": "LoadImage", "inputs": {"image": img}},
        "cve": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["cv", 0], "image": ["ref", 0], "crop": "none"}},
        "p": {"class_type": "CLIPTextEncode", "inputs": {"text": a.prompt + REAL, "clip": ["t", 0]}},
        "n": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["t", 0]}},
        **load("pose", "pose.pt"), **load("face", "face.pt"),
        "wa": {"class_type": "WanAnimateToVideo", "inputs": {"positive": ["p", 0], "negative": ["n", 0], "vae": ["v", 0],
               "clip_vision_output": ["cve", 0], "reference_image": ["ref", 0], "face_video": ["face", 0],
               "pose_video": ["pose", 0], "width": W, "height": H, "length": a.frames, "batch_size": 1,
               "continue_motion_max_frames": 5, "video_frame_offset": a.offset,
               **({"continue_motion": ["cm", 0]} if a.continue_video else {})}},
        **({"cm": {"class_type": "VHS_LoadVideoPath", "inputs": {"video": os.path.abspath(a.continue_video), "force_rate": 0,
               "custom_width": 0, "custom_height": 0, "frame_load_cap": 0, "skip_first_frames": 0,
               "select_every_nth": 2}}} if a.continue_video else {}),
        "sc": {"class_type": "PhaseSaveCond", "inputs": {"positive": ["wa", 0], "negative": ["wa", 1], "path": P("cond.pt")}},
        "sl": {"class_type": "PhaseSaveLatent", "inputs": {"samples": ["wa", 2], "path": P("latent0.pt")}},
        "st": {"class_type": "PhaseSaveAny", "inputs": {"value": ["wa", 3], "path": P("trim_latent.pt")}}},
    "sample": {
        "u": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "Wan2.2-Animate-14B-Q4_K_S.gguf"}},
        "l": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["u", 0], "strength_model": a.lora,
              "lora_name": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"}},
        "c": {"class_type": "PhaseLoadCond", "inputs": {"path": P("cond.pt")}},
        "z": {"class_type": "PhaseLoadLatent", "inputs": {"path": P("latent0.pt")}},
        **load("trim", "trim_latent.pt"),
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": a.seed}},
        "g": {"class_type": "CFGGuider", "inputs": {"model": ["l", 0], "positive": ["c", 0], "negative": ["c", 1], "cfg": 1.0}},
        "ks": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "lcm"}},
        "sig": {"class_type": "BasicScheduler", "inputs": {"model": ["l", 0], "scheduler": "simple", "steps": a.steps, "denoise": 1.0}},
        "smp": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["g", 0],
                "sampler": ["ks", 0], "sigmas": ["sig", 0], "latent_image": ["z", 0]}},
        "tr": {"class_type": "TrimVideoLatent", "inputs": {"samples": ["smp", 0], "trim_amount": ["trim", 0]}},
        "s": {"class_type": "PhaseSaveLatent", "inputs": {"samples": ["tr", 0], "path": P("latent1.pt")}}},
    "decode": {
        "v": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "z": {"class_type": "PhaseLoadLatent", "inputs": {"path": P("latent1.pt")}},
        "d": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["z", 0], "vae": ["v", 0], "tile_size": 512,
              "overlap": 64, "temporal_size": 32, "temporal_overlap": 8}},
        "f": {"class_type": "RIFE VFI", "inputs": {"frames": ["d", 0], "ckpt_name": "rife47.pth", "multiplier": 2,
              "clear_cache_after_n_frames": 10, "fast_mode": True, "ensemble": True, "scale_factor": 1.0,
              "dtype": "float16", "torch_compile": False, "batch_size": 1}},
        "o": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["f", 0], "frame_rate": 32, "loop_count": 0,
              "filename_prefix": a.prefix, "format": "video/h264-mp4", "pingpong": False, "save_output": True}}},
}

t_all = time.time()
for name in ["pose", "encode", "sample", "decode"]:
    outs = run_phase(name, phases[name], run, CAPS[name], tag=f"comfy-{a.prefix}")
    for g in outs:
        print(os.path.join(HERE, "output", g.get("subfolder", ""), g["filename"]), f"({name}) total {time.time()-t_all:.0f}s",
              flush=True)
