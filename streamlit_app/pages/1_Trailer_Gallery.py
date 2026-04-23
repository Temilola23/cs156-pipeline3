"""
Page 1: Trailer Gallery
=======================
Placeholder gallery of ideal-movie posters and keyframes.
Displays 5-pathway spec table if trailer assets are not yet available.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Trailer Gallery", layout="wide")

st.title("🎞️ Ideal Movie Trailer")

st.markdown("""
This page displays the generative output: AI-crafted poster, keyframes, and the final trailer clip.

**ACT III Status:** If the 60-second trailer is not yet rendered, this page shows the **5-pathway synthesis spec**.
""")

st.divider()

# Check for trailer video
trailer_path = Path("video/ideal_temilola_movie.mp4")

if trailer_path.exists():
    st.subheader("✓ Trailer Ready")
    st.video(str(trailer_path))
    st.success("60-second AI-generated trailer (SDXL DreamBooth + AnimateDiff Motion LoRA + CogVideoX + MusicGen)")
else:
    st.info("📹 ACT III trailer pending GPU render (A100 SDXL, AnimateDiff, CogVideoX, audio synthesis).")

st.divider()

# 5-pathway synthesis spec
st.subheader("🛠️ Generation Pipeline: 5-Pathway Spec")

spec_data = {
    "Pathway": [
        "Scene Synthesis",
        "Frame Animation",
        "Video Extension",
        "Voiceover Script",
        "Audio Composition",
    ],
    "Input (ACT II Artifact)": [
        "CVAE taste centroids + genre clusters",
        "Motion LoRA (A: generic, B: expanded corpus) + scene stills",
        "AnimateDiff clips → CogVideoX upsampling",
        "Synth synopsis (Llama QLoRA) + emotional arc",
        "Script (Llama) → TTS + MusicGen composition",
    ],
    "Method": [
        "SDXL DreamBooth (10 LoRA adapters per movie taste profile)",
        "AnimateDiff cross-frame attention (Temporal consistency)",
        "CogVideoX-2B (video diffusion model, 60-120 frames)",
        "Llama 3.1 8B (QLoRA, 8 demos, in-context learning)",
        "MusicGen (tone matching: sad/epic/upbeat from genre)",
    ],
    "Output": [
        "512×512 PNG stills (one per scene, 5 scenes total)",
        "256×256 MP4 clips (2–4 sec each, 5 clips)",
        "512×512 MP4 video (60 sec duration, 24 fps)",
        "Markdown transcript (emotional beats, quotes)",
        "48kHz WAV (voiceover + instrumental bed)",
    ],
}

spec_df = pd.DataFrame(spec_data)
st.dataframe(spec_df, use_container_width=True, hide_index=True)

st.divider()

st.markdown("""
### 📚 Taste Synthesis Rationale

Each pathway ingests a specific **ACT II artifact**:

1. **CVAE Centroids** → discovered taste "prototypes" (sweet, dark, introspective genres)
2. **GenMatch neighbors** → expand corpus to 500 candidate trailers
3. **Kalman smoother output** → temporal arc (early enthusiasm → middle-period stability → late phase evolution)
4. **HMM regimes** → identify which "taste mode" each scene should target
5. **LightGCN embeddings** → select semantically similar movies (network neighbors)

The **generative model** then assembles these into a coherent 60-second experience:
**Cold open → taste-specific scene → conflict → resolution → iconic frame**.

""")

st.info("""
**Why 5 pathways?**
- Modular (each can run in parallel on A100)
- Interpretable (trace synthesis back to ACT II method)
- Robust (if one delays, others can proceed)
- Extensible (ablations by pathway: no motion, no music, etc.)
""")
