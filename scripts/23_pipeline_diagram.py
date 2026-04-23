#!/usr/bin/env python3
"""
Task 6.1: Pipeline 3 Diagram Generator

Generates a Graphviz-based SVG diagram of the full Pipeline 3 flow:
  ELICIT (324 ratings)
  → AUGMENT (MovieLens, GenMatch, TVAE, CVAE)
  → ACT II (14 methods: state-space, uncertainty, Bayesian, representation, bandits, generative)
  → ACT III (SDXL, Llama, MotionLoRA, MusicGen)
  → OUTPUT (ffmpeg stitch → final MP4)

Outputs:
  - artifacts/plots/pipeline_diagram.svg
  - artifacts/plots/pipeline_diagram.png
"""

import os
import sys
import subprocess
from pathlib import Path

try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    print("Warning: graphviz Python package not found. Install with: pip install graphviz")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found")


def check_dot_cli():
    """Check if graphviz dot CLI is available."""
    try:
        result = subprocess.run(["which", "dot"], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


def build_graphviz_diagram(output_dir):
    """Build pipeline diagram using graphviz Digraph API."""
    dot = Digraph(
        name="Pipeline3Flow",
        comment="CS156 Pipeline 3 Full Flow Diagram",
        format="svg",
        engine="dot",
    )
    dot.attr(
        rankdir="LR",
        bgcolor="white",
        splines="ortho",
        nodesep="0.5",
        ranksep="1.2",
    )
    dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica")
    dot.attr("graph", fontname="Helvetica")

    # ========== ELICIT (Data) ==========
    with dot.subgraph(name="cluster_elicit") as c:
        c.attr(label="ELICIT (Data)", style="filled", color="lightgrey", bgcolor="#f0f0f0")
        c.node("elicit_324", "324 Ratings\n(from Minerva user)", color="#4a90e2", fontcolor="white")

    # ========== AUGMENT ==========
    with dot.subgraph(name="cluster_augment") as c:
        c.attr(label="AUGMENT", style="filled", color="lightgrey", bgcolor="#f0f0f0")
        c.node("aug_ml", "MovieLens Twin\n(51K)", color="#7ed321")
        c.node("aug_gm", "GenMatch (GA)\n(5K)", color="#7ed321")
        c.node("aug_tvae", "TVAE\n(5K)", color="#7ed321")
        c.node("aug_cvae", "CVAE\n(2.5K)", color="#7ed321")

    # ========== ACT II (14 Methods) ==========
    # Organized in semantic groups
    with dot.subgraph(name="cluster_act2") as c:
        c.attr(label="ACT II — 14 Methods (from-scratch)", style="filled", color="lightgrey", bgcolor="#f0f0f0")

        # State-space methods
        c.node("m_gp", "GP", color="#ff9800")
        c.node("m_kalman", "Kalman", color="#ff9800")
        c.node("m_rts", "RTS", color="#ff9800")
        c.node("m_ekf", "EKF", color="#ff9800")
        c.node("m_ukf", "UKF", color="#ff9800")
        c.node("m_pfbs", "Bootstrap PF", color="#ff9800")
        c.node("m_ffbsi", "FFBSi", color="#ff9800")
        c.node("m_hmm", "HMM", color="#ff9800")

        # Uncertainty / Causal
        c.node("m_conformal", "Conformal", color="#e91e63")
        c.node("m_bnn", "MC Dropout\nBNN", color="#e91e63")
        c.node("m_anova", "ANOVA\n(PyMC)", color="#e91e63")
        c.node("m_ipw", "IPW", color="#e91e63")
        c.node("m_aipw", "AIPW", color="#e91e63")
        c.node("m_genmatch2", "GenMatch\n(Causal)", color="#e91e63")

    # ========== ACT III (Generators) ==========
    with dot.subgraph(name="cluster_act3") as c:
        c.attr(label="ACT III — Generation", style="filled", color="lightgrey", bgcolor="#f0f0f0")
        c.node("gen_sdxl", "SDXL\nDreamBooth\n(scenes)", color="#9c27b0", fontcolor="white")
        c.node("gen_llama", "Llama 3.1\nQLoRA\n(voiceover)", color="#9c27b0", fontcolor="white")
        c.node("gen_motionA", "Motion LoRA A\n(generic)", color="#9c27b0", fontcolor="white")
        c.node("gen_motionB", "Motion LoRA B\n(ablations)", color="#9c27b0", fontcolor="white")
        c.node("gen_musicgen", "MusicGen\n(audio bed)", color="#9c27b0", fontcolor="white")

    # ========== OUTPUT ==========
    with dot.subgraph(name="cluster_output") as c:
        c.attr(label="OUTPUT", style="filled", color="lightgrey", bgcolor="#f0f0f0")
        c.node("output_ffmpeg", "ffmpeg\nstitch", color="#4caf50")
        c.node("output_mp4", "ideal_temilola_movie.mp4", color="#ffd700", fontcolor="black", shape="box3d")

    # ========== EDGES ==========
    # ELICIT → AUGMENT
    dot.edge("elicit_324", "aug_ml")
    dot.edge("elicit_324", "aug_gm")
    dot.edge("elicit_324", "aug_tvae")
    dot.edge("elicit_324", "aug_cvae")

    # AUGMENT → ACT II (fan-out to key methods)
    dot.edge("aug_ml", "m_gp")
    dot.edge("aug_gm", "m_genmatch2")
    dot.edge("aug_tvae", "m_kalman")
    dot.edge("aug_cvae", "m_hmm")
    dot.edge("aug_tvae", "m_bnn")
    dot.edge("aug_ml", "m_conformal")

    # ACT II → ACT III (selective routing)
    # State-space → scene generation
    dot.edge("m_kalman", "gen_sdxl")
    dot.edge("m_hmm", "gen_sdxl")

    # HMM regime → scene selection
    dot.edge("m_hmm", "gen_motionA")

    # CVAE taste → conditioning
    dot.edge("m_bnn", "gen_llama")

    # Causal → voiceover conditioning
    dot.edge("m_genmatch2", "gen_motionB")
    dot.edge("m_aipw", "gen_musicgen")

    # ACT III → OUTPUT
    dot.edge("gen_sdxl", "output_ffmpeg")
    dot.edge("gen_llama", "output_ffmpeg")
    dot.edge("gen_motionA", "output_ffmpeg")
    dot.edge("gen_motionB", "output_ffmpeg")
    dot.edge("gen_musicgen", "output_ffmpeg")

    # OUTPUT → FINAL
    dot.edge("output_ffmpeg", "output_mp4")

    # Render
    svg_path = os.path.join(output_dir, "pipeline_diagram")
    dot.render(svg_path, cleanup=True, format="svg")

    return os.path.join(output_dir, "pipeline_diagram.svg")


def build_matplotlib_diagram(output_dir):
    """Fallback: matplotlib-based diagram with rectangles and arrows."""
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Helper to draw rounded rectangle boxes
    def draw_box(ax, xy, width, height, text, color, fontsize=10):
        box = FancyBboxPatch(
            xy, width, height,
            boxstyle="round,pad=0.1",
            edgecolor="black", facecolor=color, linewidth=2, alpha=0.8
        )
        ax.add_patch(box)
        ax.text(
            xy[0] + width / 2, xy[1] + height / 2,
            text,
            ha="center", va="center", fontsize=fontsize, weight="bold", color="white"
        )

    # Helper to draw arrows
    def draw_arrow(ax, start, end, color="black"):
        arrow = FancyArrowPatch(
            start, end,
            arrowstyle="->,head_width=0.4,head_length=0.4",
            color=color, linewidth=2, mutation_scale=20
        )
        ax.add_patch(arrow)

    # ELICIT
    draw_box(ax, (0.5, 9), 2, 1.5, "324 Ratings\n(ELICIT)", "#4a90e2", 9)

    # AUGMENT cluster
    draw_box(ax, (4, 10), 1.8, 1, "MovieLens\n(51K)", "#7ed321", 8)
    draw_box(ax, (6.2, 10), 1.8, 1, "GenMatch\n(5K)", "#7ed321", 8)
    draw_box(ax, (8.4, 10), 1.8, 1, "TVAE\n(5K)", "#7ed321", 8)
    draw_box(ax, (10.6, 10), 1.8, 1, "CVAE\n(2.5K)", "#7ed321", 8)

    # Edges: ELICIT → AUGMENT
    draw_arrow(ax, (2.5, 9.75), (4, 10.5), "gray")
    draw_arrow(ax, (2.5, 9.75), (6.2, 10.5), "gray")
    draw_arrow(ax, (2.5, 9.75), (8.4, 10.5), "gray")
    draw_arrow(ax, (2.5, 9.75), (10.6, 10.5), "gray")

    # ACT II methods (arranged in two rows)
    methods_row1 = [
        ("GP", 4), ("Kalman", 5.5), ("RTS", 7), ("EKF", 8.5), ("UKF", 10), ("Bootstrap", 11.5)
    ]
    methods_row2 = [
        ("HMM", 4.5), ("Conformal", 6), ("BNN", 7.5), ("ANOVA", 9), ("IPW", 10.5), ("AIPW", 12)
    ]

    for method, x in methods_row1:
        draw_box(ax, (x - 0.6, 7.5), 1.2, 0.8, method, "#ff9800", 7)
        draw_arrow(ax, (5, 10), (x, 7.5), "gray")

    for method, x in methods_row2:
        draw_box(ax, (x - 0.6, 6.2), 1.2, 0.8, method, "#e91e63", 7)
        draw_arrow(ax, (8.4, 10), (x, 6.2), "gray")

    # ACT III generators
    draw_box(ax, (4, 4), 1.5, 1, "SDXL\nDreamBooth", "#9c27b0", 8)
    draw_box(ax, (6, 4), 1.5, 1, "Llama 3.1\nQLoRA", "#9c27b0", 8)
    draw_box(ax, (8, 4), 1.5, 1, "Motion\nLoRA A", "#9c27b0", 8)
    draw_box(ax, (10, 4), 1.5, 1, "Motion\nLoRA B", "#9c27b0", 8)
    draw_box(ax, (12, 4), 1.5, 1, "MusicGen", "#9c27b0", 8)

    # ACT II → ACT III
    for method, x in methods_row1 + methods_row2:
        draw_arrow(ax, (x, 6.2), (6.5, 5), "gray")

    # OUTPUT: ffmpeg
    draw_box(ax, (6.5, 2), 2.5, 1, "ffmpeg stitch", "#4caf50", 9)
    for x_gen in [4.75, 6.75, 8.75, 10.75, 12.75]:
        draw_arrow(ax, (x_gen, 4), (7.75, 3), "gray")

    # Final output
    draw_box(ax, (6.5, 0.3), 2.5, 1, "ideal_temilola_movie.mp4", "#ffd700", 9)
    draw_arrow(ax, (7.75, 2), (7.75, 1.3), "darkgreen")

    # Title
    ax.text(10, 11.8, "CS156 Pipeline 3: Full Flow", fontsize=16, weight="bold", ha="center")

    # Save
    png_path = os.path.join(output_dir, "pipeline_diagram.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Matplotlib diagram saved to: {png_path}")
    plt.close()

    return png_path


def main():
    script_dir = Path(__file__).parent.absolute()
    repo_root = script_dir.parent
    output_dir = repo_root / "artifacts" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pipeline 3 Diagram Generator")
    print(f"Output directory: {output_dir}")
    print()

    has_dot = check_dot_cli()

    if has_dot and HAS_GRAPHVIZ:
        print("✓ graphviz CLI (dot) available")
        print("✓ graphviz Python package available")
        print("→ Using graphviz backend (professional vector diagram)")
        print()

        try:
            svg_file = build_graphviz_diagram(str(output_dir))
            print(f"✓ SVG generated: {svg_file}")

            # Also generate PNG from SVG using graphviz
            png_file = os.path.join(output_dir, "pipeline_diagram.png")
            try:
                subprocess.run(
                    ["dot", "-Tpng", "-Gdpi=150", svg_file, f"-o{png_file}"],
                    check=True,
                    capture_output=True
                )
                print(f"✓ PNG generated: {png_file}")
            except subprocess.CalledProcessError as e:
                print(f"Warning: PNG generation via dot failed: {e}")
                if HAS_MATPLOTLIB:
                    print("  Falling back to matplotlib for PNG...")
                    build_matplotlib_diagram(str(output_dir))

        except Exception as e:
            print(f"Graphviz generation failed: {e}")
            if HAS_MATPLOTLIB:
                print("Falling back to matplotlib...")
                build_matplotlib_diagram(str(output_dir))
    elif HAS_MATPLOTLIB:
        print("Graphviz CLI or Python package not available.")
        print("→ Using matplotlib fallback (raster diagram)")
        print()
        build_matplotlib_diagram(str(output_dir))
    else:
        print("ERROR: Neither graphviz nor matplotlib available!")
        sys.exit(1)

    # Verify outputs
    print()
    print("Output files:")
    for ext in ["svg", "png"]:
        path = output_dir / f"pipeline_diagram.{ext}"
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  ✓ {path.name}: {size_kb:.1f} KB")
        else:
            print(f"  ✗ {path.name}: NOT FOUND")


if __name__ == "__main__":
    main()
