#!/usr/bin/env python3
"""
Rebuild the two §9.9 sample strips from RAW generator outputs.

- sdxl_posters_sample.png: 2x3 grid of six SDXL-LoRA narrative keyframes,
  selected for visual diversity (rooftop / lab / chase / operatives /
  portal / silhouette). Pulled from _local_assembly/narrative_keyframes/.
- stylegan3_frames_sample.png: 2x3 grid of six StyleGAN3-ADA raw samples
  from the trailer corpus-fit generator. Pulled from
  _local_assembly/stylegan3_upscaled/ (1920x1080 upscales of the 32 raw
  W-space samples).

These are RAW generator outputs. No title cards, no teaser overlays.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.image import imread

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "plots"
ART.mkdir(parents=True, exist_ok=True)

KF = ROOT / "_local_assembly" / "narrative_keyframes"
SG = ROOT / "_local_assembly" / "stylegan3_upscaled"


def _grid(paths, titles, out_path, suptitle, dpi=150):
    assert len(paths) == 6 and len(titles) == 6, "expect 6 images"
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2))
    for ax, p, t in zip(axes.ravel(), paths, titles):
        ax.imshow(imread(p))
        ax.set_title(t, fontsize=10.5, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(suptitle, fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# SDXL: six scenes with distinct prompts / visual vocabulary.
sdxl_paths = [
    KF / "scene_01.png",   # rooftop neon cityscape
    KF / "scene_03.png",   # underground lab
    KF / "scene_07.png",   # aerial hover chase
    KF / "scene_10.png",   # masked operatives
    KF / "scene_13.png",   # quantum portal
    KF / "scene_15.png",   # samurai silhouette sunset
]
sdxl_titles = [
    "(a) rooftop cityscape opener",
    "(b) underground laboratory",
    "(c) aerial hover chase",
    "(d) masked operatives",
    "(e) quantum-portal close-up",
    "(f) silhouette at sunset",
]
_grid(
    sdxl_paths,
    sdxl_titles,
    ART / "sdxl_posters_sample.png",
    "SDXL + LoRA raw keyframes (1024x576, 40-step DDIM, guidance 8.5)",
)

# StyleGAN3: six W-space samples spanning palette and shape clusters.
sg_paths = [
    SG / "trailer_scene_09_1920x1080.jpg",   # blue-grey mountainous cloud
    SG / "trailer_scene_11_1920x1080.jpg",   # yellow/orange heat silhouette
    SG / "trailer_scene_20_1920x1080.jpg",   # dark alley with reflection
    SG / "trailer_scene_26_1920x1080.jpg",   # industrial sunset
    SG / "trailer_scene_27_1920x1080.jpg",   # blue-green explosion streak
    SG / "trailer_scene_31_1920x1080.jpg",   # monochrome street reflection
]
sg_titles = [
    "(a) blue-grey cloud / mountain",
    "(b) orange-sky heat haze",
    "(c) dark alley reflection",
    "(d) industrial-sunset silhouette",
    "(e) blue/green streak",
    "(f) monochrome reflection",
]
_grid(
    sg_paths,
    sg_titles,
    ART / "stylegan3_frames_sample.png",
    "StyleGAN3-ADA raw samples (1920x1080 upscale of W-space draws, 20 kimg)",
)
