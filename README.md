# Motion clone studio

Copy the camera move, body motion and facial expressions of a driving video onto a character image, keeping the
character's face and figure. Wan2.2-Animate-14B (Q4 GGUF) in ComfyUI, run on a free Kaggle T4 GPU.

- Page: https://mohanappmontize.github.io/motion-clone-studio/
- Start a run: page -> "Add video and image" -> GitHub form -> drag files -> Create.
- Output keeps the driving video's full length, size, fps and audio. Long videos are made in 4.8 s windows,
  each seeded by the previous window's last frames; each scene (hard cut) is its own chain.

Layout: `docs/` page (GitHub Pages) · `kaggle/worker.py` GPU worker · `relay/relay.py` issue -> Kaggle -> release
· `pipeline/` the same scripts the laptop studio runs.
Uploads and results are public (release `job-<n>`).
