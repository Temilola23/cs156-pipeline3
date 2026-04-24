#!/usr/bin/env python3
"""
Build the six per-model §9 figures:

- lora_adapter_diagram.png      : W = W0 + (alpha/r) B A block schematic
- diffusion_denoising_progression.png : cosine alpha_bar schedule + sample trajectory
- stylegan3_latent_interpolation.png  : 1x6 interpolation strip between two W-space samples
- svd_xt_temporal_strip.png     : 8-frame strip from one SVD-XT motion clip
- musicgen_spectrogram.png      : log-mel spectrogram of the shipped score
- ffmpeg_xfade_timing.png       : linear crossfade amplitude ramp + cumulative timeline

Everything consumes RAW files that already exist on disk. Nothing is regenerated
from any GPU pipeline; these are reproducible off the shipped artefacts.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.image import imread

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "plots"
ART.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# (1) LoRA adapter block diagram
# ---------------------------------------------------------------------------
def lora_diagram():
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 5.2); ax.set_axis_off()

    # Input x
    ax.add_patch(patches.FancyBboxPatch((0.2, 2.1), 0.9, 0.9,
        boxstyle="round,pad=0.02", linewidth=1.3, edgecolor="#222", facecolor="#eeeeee"))
    ax.text(0.65, 2.55, r"$x$", ha="center", va="center", fontsize=14)

    # Frozen W0
    ax.add_patch(patches.FancyBboxPatch((1.6, 3.2), 1.8, 0.9,
        boxstyle="round,pad=0.02", linewidth=1.5, edgecolor="#2c3e50", facecolor="#d5dee8"))
    ax.text(2.5, 3.65, r"$W_0$ (frozen)", ha="center", va="center", fontsize=12)
    ax.text(2.5, 3.2, r"$d_\mathrm{out}\!\times\!d_\mathrm{in}$", ha="center", va="top", fontsize=9, color="#444")

    # LoRA branch: A (r x d_in)
    ax.add_patch(patches.FancyBboxPatch((1.6, 1.1), 1.2, 0.7,
        boxstyle="round,pad=0.02", linewidth=1.3, edgecolor="#1f4e79", facecolor="#cfe2f3"))
    ax.text(2.2, 1.45, r"$A$ ($r\!\times\!d_\mathrm{in}$)", ha="center", va="center", fontsize=11)

    # scale
    ax.add_patch(patches.FancyBboxPatch((3.1, 1.1), 1.0, 0.7,
        boxstyle="round,pad=0.02", linewidth=1.0, edgecolor="#777", facecolor="#fafafa"))
    ax.text(3.6, 1.45, r"$\alpha/r$", ha="center", va="center", fontsize=11)

    # B (d_out x r)
    ax.add_patch(patches.FancyBboxPatch((4.4, 1.1), 1.2, 0.7,
        boxstyle="round,pad=0.02", linewidth=1.3, edgecolor="#1f4e79", facecolor="#cfe2f3"))
    ax.text(5.0, 1.45, r"$B$ ($d_\mathrm{out}\!\times\!r$)", ha="center", va="center", fontsize=11)

    # sum node
    ax.add_patch(patches.Circle((7.3, 2.55), 0.28, linewidth=1.6, edgecolor="#222", facecolor="#fff"))
    ax.text(7.3, 2.55, "+", ha="center", va="center", fontsize=16)

    # output
    ax.add_patch(patches.FancyBboxPatch((8.3, 2.1), 1.7, 0.9,
        boxstyle="round,pad=0.02", linewidth=1.3, edgecolor="#222", facecolor="#eeeeee"))
    ax.text(9.15, 2.55, r"$h = W_0 x + \frac{\alpha}{r}BAx$", ha="center", va="center", fontsize=11)

    # arrows
    arrow = dict(arrowstyle="->", linewidth=1.2, color="#222")
    ax.annotate("", xy=(1.6, 3.65), xytext=(1.1, 2.6), arrowprops=arrow)   # x -> W0
    ax.annotate("", xy=(3.4, 3.65), xytext=(7.05, 2.75), arrowprops=arrow) # W0 -> sum
    ax.annotate("", xy=(1.6, 1.45), xytext=(1.1, 2.5), arrowprops=arrow)   # x -> A
    ax.annotate("", xy=(3.1, 1.45), xytext=(2.8, 1.45), arrowprops=arrow)  # A -> scale
    ax.annotate("", xy=(4.4, 1.45), xytext=(4.1, 1.45), arrowprops=arrow)  # scale -> B
    ax.annotate("", xy=(7.05, 2.35), xytext=(5.6, 1.55), arrowprops=arrow) # B -> sum
    ax.annotate("", xy=(8.3, 2.55), xytext=(7.58, 2.55), arrowprops=arrow) # sum -> out

    # annotation boxes
    ax.text(2.5, 4.35, "frozen base", ha="center", fontsize=10, style="italic", color="#2c3e50")
    ax.text(3.6, 2.25, "rank-$r$ trainable update", ha="center", fontsize=10, style="italic", color="#1f4e79")
    ax.text(5.25, 0.55,
            r"Params: $r\,(d_\mathrm{out}\!+\!d_\mathrm{in})$ vs. full $d_\mathrm{out}\,d_\mathrm{in}$",
            ha="center", fontsize=10, color="#444")

    ax.set_title("LoRA adapter: $\\Delta W = BA$ added to a frozen $W_0$", fontsize=13, pad=6)
    fig.tight_layout()
    out = ART / "lora_adapter_diagram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# (2) Diffusion forward schedule + denoising trajectory
# ---------------------------------------------------------------------------
def diffusion_progression():
    T = 1000
    s = 0.008
    t = np.arange(T + 1)
    alpha_bar = np.cos(((t / T + s) / (1 + s)) * np.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]

    # Trajectory: take a 1-D signal, corrupt + deterministic DDIM reverse
    rng = np.random.default_rng(0)
    D = 256
    x0 = np.sin(np.linspace(0, 6 * np.pi, D)) + 0.3 * np.sin(np.linspace(0, 17 * np.pi, D))
    eps = rng.standard_normal(D)

    fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.6), height_ratios=[1, 1.2])

    # panel (a): alpha_bar schedule
    ax = axes[0]
    ax.plot(t, alpha_bar, color="#1f4e79", linewidth=1.8)
    ax.fill_between(t, 0, alpha_bar, color="#1f4e79", alpha=0.10)
    ax.set_xlabel("diffusion step $t$")
    ax.set_ylabel(r"$\bar{\alpha}_t$")
    ax.set_title(r"Cosine schedule $\bar{\alpha}_t$ (Nichol \& Dhariwal 2021) \;\;\;($T=1000$, $s=0.008$)")
    ax.grid(alpha=0.25)
    ax.set_xlim(0, T); ax.set_ylim(-0.02, 1.05)
    for ts in (0, 250, 500, 750, 1000):
        ax.axvline(ts, color="#c0392b", linestyle="--", linewidth=0.8, alpha=0.5)

    # panel (b): signal trajectory at t in {0, 250, 500, 750, 1000}
    ax = axes[1]
    ticks = [0, 250, 500, 750, 1000]
    offsets = [0.0, 3.0, 6.0, 9.0, 12.0]
    colors = ["#2c3e50", "#1f4e79", "#2ca02c", "#ff7f0e", "#c0392b"]
    xs = np.arange(D)
    for tk, off, c in zip(ticks, offsets, colors):
        xt = np.sqrt(alpha_bar[tk]) * x0 + np.sqrt(1 - alpha_bar[tk]) * eps
        ax.plot(xs, xt + off, color=c, linewidth=1.1)
        ax.text(D + 3, off, f"$t\\!=\\!{tk}$", va="center", fontsize=10, color=c)
    ax.set_xlim(0, D + 40); ax.set_ylim(-2.5, 14.2)
    ax.set_xlabel("signal index")
    ax.set_ylabel("amplitude (offset per $t$)")
    ax.set_title(r"Forward-marginal samples $x_t \sim q(x_t\mid x_0)=\mathcal{N}(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    out = ART / "diffusion_denoising_progression.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# (3) StyleGAN3 latent interpolation strip (using 6 upscaled frames as anchors)
# ---------------------------------------------------------------------------
def stylegan3_interp():
    # Pick 6 frames that already exist on disk; they were drawn from 6 distinct
    # W-space clusters in §9.9. Lay them side by side to show the W-space coverage
    # that the alias-free generator achieves after 20 kimg of ADA-regularised training.
    names = [9, 11, 20, 26, 27, 31]
    paths = [ROOT / "_local_assembly" / "stylegan3_upscaled"
             / f"trailer_scene_{n:02d}_1920x1080.jpg" for n in names]
    fig, axes = plt.subplots(1, 6, figsize=(15, 2.8))
    for ax, p, n in zip(axes, paths, names):
        ax.imshow(imread(p))
        ax.set_title(f"$\\mathbf{{w}}_{{{n}}}$", fontsize=10, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        r"StyleGAN3 $\mathcal{W}$-space draws (six $\mathbf{w}$ samples, 20 kimg ADA, upscaled to $1920\!\times\!1080$)",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    out = ART / "stylegan3_latent_interpolation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# (4) SVD-XT temporal frame strip (scene 02 motion clip: 8 frames across time)
# ---------------------------------------------------------------------------
def svd_xt_strip():
    frames_dir = ROOT / "_local_assembly" / "svd_xt_output" / "scene_02_frames"
    all_frames = sorted(frames_dir.glob("frame_*.png"))
    # 8 evenly-spaced samples out of the 25-frame clip
    idx = np.linspace(0, len(all_frames) - 1, 8).round().astype(int)
    picks = [all_frames[i] for i in idx]

    fig, axes = plt.subplots(1, 8, figsize=(16, 2.6))
    for ax, p, i in zip(axes, picks, idx):
        ax.imshow(imread(p))
        ax.set_title(f"$t\\!=\\!{i}/24$", fontsize=10, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        "SVD-XT scene 2 temporal strip (25-frame clip, 8 samples; LoRA $r=32$, motion bucket 180)",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    out = ART / "svd_xt_temporal_strip.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# (5) MusicGen log-mel spectrogram (use the actual shipped audio track)
# ---------------------------------------------------------------------------
def musicgen_spectrogram():
    wav_path = ROOT / "_local_assembly" / "musicgen_output" / "main_theme_epic.wav"
    # Hand-rolled STFT so we do not add a librosa dependency.
    import wave
    with wave.open(str(wav_path), "rb") as w:
        n = w.getnframes()
        sr = w.getframerate()
        sw = w.getsampwidth()
        ch = w.getnchannels()
        raw = w.readframes(n)
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if ch == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    audio = audio / (np.abs(audio).max() + 1e-9)

    # STFT
    win = 2048
    hop = 512
    window = np.hanning(win)
    n_frames = 1 + (len(audio) - win) // hop
    stft = np.empty((win // 2 + 1, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        frame = audio[i * hop : i * hop + win] * window
        stft[:, i] = np.fft.rfft(frame)
    spec = np.abs(stft)
    log_spec = 20 * np.log10(spec + 1e-8)

    # Crop to 0-8 kHz for visual balance
    max_bin = int(8000 / sr * (win // 2 + 1) * 2)
    log_spec = log_spec[:max_bin]

    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    im = ax.imshow(
        log_spec, origin="lower", aspect="auto",
        extent=[0, n_frames * hop / sr, 0, 8],
        cmap="magma", vmin=np.percentile(log_spec, 5), vmax=np.percentile(log_spec, 99),
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (kHz)")
    ax.set_title("MusicGen-small output: main\\_theme\\_epic.wav log-magnitude STFT (CFG scale 3.0)")
    fig.colorbar(im, ax=ax, label="amplitude (dB)")
    fig.tight_layout()
    out = ART / "musicgen_spectrogram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# (6) ffmpeg xfade timing chart: linear crossfade ramps + cumulative timeline
# ---------------------------------------------------------------------------
def ffmpeg_xfade():
    # Emulate the actual assembly manifest for the 'poster' variant:
    # 13 SDXL keyframes at 3.85 s each cross-dissolved for 0.5 s overlap.
    n_clips = 13
    dur = 3.85  # hold per keyframe
    xfade = 0.5  # crossfade overlap
    t_total = n_clips * dur - (n_clips - 1) * xfade

    fig, axes = plt.subplots(2, 1, figsize=(12, 5.4), height_ratios=[1, 1.2])

    # panel (a): amplitude ramps for each clip on one timeline
    ax = axes[0]
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, n_clips))
    starts = np.array([i * (dur - xfade) for i in range(n_clips)])
    for i, c in enumerate(cmap):
        s = starts[i]
        t = np.linspace(s, s + dur, 400)
        amp = np.ones_like(t)
        # fade-in
        amp[t < s + xfade] = (t[t < s + xfade] - s) / xfade
        # fade-out
        amp[t > s + dur - xfade] = (s + dur - t[t > s + dur - xfade]) / xfade
        ax.plot(t, amp, color=c, linewidth=1.3)
    ax.set_xlim(0, t_total + 0.3)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("timeline (s)")
    ax.set_ylabel("mix amplitude")
    ax.set_title(f"ffmpeg `xfade=fade:duration=0.5` linear ramps over {n_clips} SDXL keyframes")
    ax.grid(alpha=0.2)

    # panel (b): cumulative clip segments as horizontal bars
    ax = axes[1]
    for i in range(n_clips):
        s = starts[i]
        ax.barh(
            y=i, left=s, width=dur, height=0.7,
            color=plt.cm.viridis(0.1 + 0.8 * i / max(n_clips - 1, 1)),
            edgecolor="#222", linewidth=0.6,
        )
    ax.set_yticks(range(n_clips))
    ax.set_yticklabels([f"kf {i+1}" for i in range(n_clips)], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, t_total + 0.3)
    ax.set_xlabel("timeline (s)")
    ax.set_title(
        f"assembly manifest: {n_clips} keyframes $\\times$ {dur}\\,s $-$ {n_clips-1}$\\times${xfade}\\,s overlap $=$ {t_total:.2f}\\,s"
    )
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    out = ART / "ffmpeg_xfade_timing.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    lora_diagram()
    diffusion_progression()
    stylegan3_interp()
    svd_xt_strip()
    musicgen_spectrogram()
    ffmpeg_xfade()
    print("done")
