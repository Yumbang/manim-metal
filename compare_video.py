# ruff: noqa
"""Render an animated scene with BOTH Cairo and Metal, then diff frame-by-frame.

Answers: does Cairo-vs-Metal parity hold for VIDEO (animation), not just static
frames?  Renders the same scene with each backend to separate media dirs,
extracts every frame with ffmpeg, aligns them, and reports per-frame and
worst-frame divergence.  Saves a montage of the worst frame for visual review.

Usage:
    uv run python compare_video.py VTransform
    uv run python compare_video.py VThreeDRot --quality l
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
SCENE_FILE = os.path.join(ROOT, "video_scenes.py")
OUT = os.path.join(ROOT, "media_vcmp")


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def _render(scene: str, renderer: str, media_dir: str, quality: str, fps: int) -> str:
    """Render scene with the given backend; return path to the produced mp4."""
    if os.path.isdir(media_dir):
        shutil.rmtree(media_dir)
    cmd = [
        "uv", "run", "manim",
        f"-q{quality}", "--fps", str(fps), "--disable_caching",
        "--renderer", renderer, "--media_dir", media_dir,
        SCENE_FILE, scene,
    ]
    r = _run(cmd)
    mp4s = glob.glob(os.path.join(media_dir, "videos", "**", "*.mp4"), recursive=True)
    if not mp4s:
        sys.stderr.write(r.stdout[-3000:] + "\n" + r.stderr[-3000:] + "\n")
        raise RuntimeError(f"{renderer}: no mp4 produced for {scene}")
    return max(mp4s, key=os.path.getmtime)


def _extract(mp4: str, frame_dir: str) -> list[str]:
    if os.path.isdir(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)
    _run(["ffmpeg", "-y", "-i", mp4, os.path.join(frame_dir, "%05d.png")])
    return sorted(glob.glob(os.path.join(frame_dir, "*.png")))


def compare_scene(scene: str, quality: str, fps: int) -> dict:
    os.makedirs(OUT, exist_ok=True)
    metal_mp4 = _render(scene, "metal", os.path.join(OUT, scene, "metal_media"), quality, fps)
    cairo_mp4 = _render(scene, "cairo", os.path.join(OUT, scene, "cairo_media"), quality, fps)

    mf = _extract(metal_mp4, os.path.join(OUT, scene, "metal_frames"))
    cf = _extract(cairo_mp4, os.path.join(OUT, scene, "cairo_frames"))

    n = min(len(mf), len(cf))
    per_frame = []
    worst = {"mae": -1.0, "i": -1}
    for i in range(n):
        m = np.array(Image.open(mf[i]).convert("RGB")).astype(np.int16)
        c = np.array(Image.open(cf[i]).convert("RGB")).astype(np.int16)
        if m.shape != c.shape:
            h = min(m.shape[0], c.shape[0]); w = min(m.shape[1], c.shape[1])
            m, c = m[:h, :w], c[:h, :w]
        d = np.abs(m - c)
        mae = float(d.mean())
        bad = float((d.max(2) > 12).mean() * 100)
        per_frame.append({"i": i, "mae": round(mae, 3), "maxdiff": int(d.max()), "bad%": round(bad, 3)})
        if mae > worst["mae"]:
            worst = {"mae": mae, "i": i, "maxdiff": int(d.max()), "bad%": bad}

    # Save worst-frame montage: cairo | metal | diff(x4 red)
    if worst["i"] >= 0:
        m = np.array(Image.open(mf[worst["i"]]).convert("RGB")).astype(np.int16)
        c = np.array(Image.open(cf[worst["i"]]).convert("RGB")).astype(np.int16)
        h = min(m.shape[0], c.shape[0]); w = min(m.shape[1], c.shape[1])
        m, c = m[:h, :w], c[:h, :w]
        diff = np.abs(m - c).max(2)
        heat = np.zeros((h, w, 3), np.uint8); heat[:, :, 0] = np.clip(diff * 4, 0, 255)
        strip = np.concatenate([c.astype(np.uint8), m.astype(np.uint8), heat], axis=1)
        Image.fromarray(strip).save(os.path.join(OUT, f"{scene}_worst.png"))

    maes = [f["mae"] for f in per_frame]
    bads = [f["bad%"] for f in per_frame]
    summary = {
        "scene": scene,
        "frames_metal": len(mf),
        "frames_cairo": len(cf),
        "frames_compared": n,
        "mean_mae": round(float(np.mean(maes)), 3) if maes else None,
        "max_mae": round(float(np.max(maes)), 3) if maes else None,
        "mean_bad%": round(float(np.mean(bads)), 3) if bads else None,
        "max_bad%": round(float(np.max(bads)), 3) if bads else None,
        "worst_frame": worst["i"],
        "per_frame": per_frame,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--quality", default="l")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()
    s = compare_scene(args.scene, args.quality, args.fps)
    # compact human line + full json
    print(
        f"{s['scene']}: frames m/c={s['frames_metal']}/{s['frames_cairo']} "
        f"mean_mae={s['mean_mae']} max_mae={s['max_mae']} "
        f"mean_bad%={s['mean_bad%']} max_bad%={s['max_bad%']} worst@{s['worst_frame']}"
    )
    print("JSON " + json.dumps(s))


if __name__ == "__main__":
    main()
