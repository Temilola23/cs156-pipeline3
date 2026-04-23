# Pipeline 3: Video Generation Research
## The "Mother of All Pipelines" — From Taste Centroid to Teaser Trailer

**Author:** Research Agent
**Date:** Apr 16, 2026
**Deadline:** Apr 23, 2026 (7 days)
**Hardware:** Colab Pro (A100 40GB, intermittent sessions)
**Scope:** Generate a personalized movie teaser (30–60s) conditioned on Temilola's ⭐8.5+ taste centroid

---

## 1. Closed/Proprietary Tools: The Honest Assessment

You asked about DALL-E, Midjourney, Sora, Veo 3, and others. Here's what you can and **cannot** do with each.

### DALL-E 3 (OpenAI)
- **Architecture:** Image-only diffusion (no temporal dimension).
- **API Access:** Yes, via OpenAI API (`gpt-4-vision`).
- **What YOU CAN do:** Call the API to generate single poster frames, then stitch with ffmpeg.
- **What YOU CANNOT do:** Fine-tune. Zero fine-tuning mechanism. It's black-box inference only.
- **Rubric Fit:** Posters alone don't meet the "video generation" brief. Pure inference, no novel ML.

### Midjourney (Anthropic)
- **Architecture:** Image diffusion, proprietary.
- **API Access:** No public API. Web-only interface.
- **What YOU CAN do:** Nothing programmatically. You can prompt via web UI.
- **What YOU CANNOT do:** API calls, fine-tuning, batch automation.
- **Rubric Fit:** Completely ruled out — no way to integrate into Pipeline 3.

### Sora (OpenAI)
- **Architecture:** Diffusion transformer, text-to-video, closed weights.
- **API Access:** Gated rollout. No public API as of Apr 2026. Waitlist only.
- **What YOU CAN do:** If you get access: submit prompts, get video clips. Inference only.
- **What YOU CANNOT do:** Fine-tuning (impossible), weights access (impossible), conditioning on personal embeddings (impossible).
- **Rubric Fit:** Even with API access, you're just prompt-engineering — no personalization, no novel math, no fine-tuning. High risk (you may not get access by Apr 23).

### Veo 3 (Google / Vertex AI)
- **Architecture:** Diffusion-based text-to-video, closed weights.
- **API Access:** Vertex AI (requires Google Cloud credits; pay-per-call model).
- **What YOU CAN do:** Call inference API with prompts to generate videos.
- **What YOU CANNOT do:** Fine-tune weights. Access only inference. No way to condition on your taste centroid.
- **Rubric Fit:** Again, prompt engineering only. No novel ML contribution.

### Runway Gen-3 Alpha
- **Architecture:** Closed text-to-video and image-to-video diffusion.
- **API Access:** Yes, has REST API for inference.
- **Custom Motion Features:** "Custom motion" allows you to upload a reference motion video; the model adapts style. **NOT fine-tuning of weights** — just prompt-level conditioning.
- **What YOU CAN do:** Generate videos via API; use custom motion for style transfer.
- **What YOU CANNOT do:** Fine-tune the underlying diffusion weights. LoRA? Impossible. Access to internals? Nope.
- **Rubric Fit:** Custom motion is clever, but it's not genuine fine-tuning. You're still prompt-engineering.

### Kling AI (Kuaishou)
- **Architecture:** Text-to-video diffusion.
- **API Access:** Limited, mostly web UI.
- **What YOU CAN do:** Generate videos via web interface (no Python integration as of Apr 2026).
- **What YOU CANNOT do:** Fine-tune, programmatic access, personalization.
- **Rubric Fit:** Ruled out.

### Pika Labs
- **Architecture:** Text-to-video and image-to-video, closed.
- **API Access:** Discord bot, web UI.
- **What YOU CAN do:** Generate clips through bot or UI.
- **What YOU CANNOT do:** Fine-tune, programmatic control, personalization.
- **Rubric Fit:** Not suitable for an ML course project.

---

## Summary: Why Proprietary is Out

**For every closed-weights model above, you get:**
- ✗ No fine-tuning (LoRA, DreamBooth, LoHa, anything)
- ✗ No conditioning on personal embeddings or taste vectors
- ✗ No novel math derivation (you're just calling an API)
- ✗ Rubric failure: "No demonstrated learning of model internals" (MLMath LO 5)

**You CAN use them as fallback "polish" (e.g., generate 1–2 frames via Sora API if you get access), but the core Pipeline 3 MUST be open-source and fine-tunable to earn credit.**

---

## 2. Open-Source Fine-Tunable Video Models: Ranked by 7-Day Feasibility

### Tier 1: "GET THIS DONE BY Apr 23" — Highest Feasibility

#### **AnimateDiff** (Top pick for Pipeline 3)
- **Architecture:** Motion module that bolts onto SD 1.5, SDXL, or any base diffusion model. Learns temporal cross-frame attention; spatial layers are frozen (from pre-trained SD). ~75M params for the motion module.
- **Weights:** HuggingFace: `guoyww/animatediff-motion-adapter-v1-5`
- **Base Model:** Use your existing SDXL + LoRA (fine-tuned on ⭐8.5+ posters from Pipeline 2).
- **VRAM:** Inference ~6–10 GB on A100. Fine-tuning with LoRA on the motion module: ~12–16 GB (very doable).
- **Fine-tuning Approach:** LoRA on the temporal attention layers only. Few-shot: if you collect 10–20 short motion videos of "ideal Temilola movie aesthetics" (e.g., smooth camera pans through your favorite movies' trailer footage), you can fine-tune in 4–6 hours on A100.
- **Realistic Clip Length:** 2–8 seconds at 8 fps (16 frames), depending on guidance scale. Can be stitched.
- **Math Hooks:** Temporal cross-frame attention, why freezing spatial layers preserves SDXL's learned poster aesthetics, LoRA rank decomposition, guidance in 4D space (spatial + temporal).
- **Implementation Effort:** **2–3 days.** You already have SDXL+LoRA from Pipeline 2. AnimateDiff integration is a drop-in. Existing tutorials and Hugging Face pipelines are solid.
- **Colab Pro Fit:** A- (fits comfortably, no session crashes expected)

#### **Stable Video Diffusion (SVD / SVD-XT)** (Second pick)
- **Architecture:** Diffusion-based image-to-video. Takes a single keyframe image + text conditioning (optional). 8.7B parameters. Produces 25 frames (1–4 seconds at 30 fps, or 4–8 seconds at ~8 fps).
- **Weights:** HuggingFace: `stabilityai/stable-video-diffusion-img2vid` and `stabilityai/stable-video-diffusion-img2vid-xt` (4 secs vs 25 frame variants).
- **VRAM:** Inference ~8–12 GB. Fine-tuning: Stability AI released a LoRA fine-tuning approach around early 2025, but it's still underdocumented. Expect 16–24 GB for fine-tuning.
- **Fine-tuning Approach:** LoRA on cross-attention layers + temporal attention. OR: use it inference-only and rely on clever prompt engineering + keyframe selection.
- **Realistic Clip Length:** 4–8 seconds per keyframe.
- **Math Hooks:** Forward diffusion SDE, score matching, temporal VAE latent space, 3D convolutions for optical flow consistency.
- **Implementation Effort:** **2–3 days** (inference only, no fine-tuning); **4–5 days** (if you want to fine-tune).
- **Colab Pro Fit:** A (inference); B+ (fine-tuning — pushing the RAM limit).

#### **CogVideoX-2B / CogVideoX-5B** (Tsinghua)
- **Architecture:** Diffusion transformer, text-to-video. Clever factorization: spatial transformer blocks + temporal transformer blocks. 2B (lightweight) and 5B (better quality) variants.
- **Weights:** HuggingFace: `THUDM/CogVideoX-2b` and `THUDM/CogVideoX-5b`.
- **VRAM:** CogVideoX-2B inference ~10 GB, fine-tuning ~14–18 GB. CogVideoX-5B inference ~14 GB, fine-tuning ~20–28 GB.
- **Fine-tuning Approach:** Full repo includes training scripts. LoRA support is experimental (in latest releases, early 2025). Full fine-tuning on 5B is possible on A100 40GB with batch size 1–2 and gradient checkpointing.
- **Realistic Clip Length:** 6–8 seconds (50–96 frames at 16 fps).
- **Math Hooks:** Spatial-temporal factorization, why decoupling improves training stability, diffusion guidance in factored space.
- **Implementation Effort:** **3–4 days** (CogVideoX-2B with LoRA); **5–6 days** (5B full fine-tuning). Training is slower than AnimateDiff; expect 8–16 hours per fine-tuning epoch.
- **Colab Pro Fit:** B+ (CogVideoX-2B doable; 5B is tight but possible with aggressive memory optimization).

---

### Tier 1.5: "Viable but Tighter Timeline"

#### **Open-Sora (1.1, 1.2, 2.0)** (HPC-AI Lab)
- **Architecture:** Open re-implementation of Sora-like diffusion transformer. Latent diffusion over compressed video tokens. 1.1 and 1.2 are text-to-video. 2.0 is the latest (early 2025).
- **Weights:** HuggingFace: `hpcaitech/Open-Sora` (multiple branches).
- **VRAM:** 1.1/1.2 inference ~10–12 GB. Fine-tuning: 18–24 GB (with LoRA).
- **Fine-tuning Approach:** Code includes DreamBooth-like fine-tuning via text embeddings. Can also do LoRA on transformer layers.
- **Realistic Clip Length:** 6–8 seconds (depends on resolution and frame rate settings).
- **Math Hooks:** Latent diffusion + video tokenization (similar to VideoPoet), compressed representation, why working in token space speeds up training.
- **Implementation Effort:** **4–5 days**. The codebase is solid but less documented than AnimateDiff. Training pipeline is more complex.
- **Colab Pro Fit:** B (feasible, but tighter on memory and session time).

---

### Tier 2: "Ambitious but Risky for 7 Days"

#### **HunyuanVideo** (Tencent)
- **Architecture:** DiT (Diffusion Transformer) for video. 13B parameters, strong performance. Competitive with commercial models.
- **Weights:** HuggingFace: `Tencent/HunyuanVideo`.
- **VRAM:** Inference ~18–22 GB. Fine-tuning: 24–32 GB (very tight on A100 40GB).
- **Fine-tuning Approach:** Code released with LoRA support, but it's recent (Q1 2025) and still finding edge cases.
- **Realistic Clip Length:** 5 seconds (104 frames at 16 fps).
- **Math Hooks:** DiT architecture, why parameter-sharing via transformer blocks scales better than convolutional chains.
- **Implementation Effort:** **5–6 days** (just to get inference + LoRA pipeline stable). Training time is long (12–20 hrs per epoch).
- **Colab Pro Fit:** C (RAM is at the limit; session crashes likely; requires aggressive optimization).

#### **Mochi 1** (Genmo, Apache 2.0)
- **Architecture:** Latent diffusion transformer, text-to-video. 10B parameters. Recent (Q1 2025).
- **Weights:** HuggingFace: `genmo/mochi-1` (and variants).
- **VRAM:** Inference ~14–16 GB. Fine-tuning: 20–24 GB.
- **Fine-tuning Approach:** Early code, LoRA is feasible but requires custom implementation (not plug-and-play).
- **Realistic Clip Length:** 5 seconds (120 frames at 24 fps).
- **Math Hooks:** Latent diffusion + transformer backbone, why gating mechanisms in attention improve temporal coherence.
- **Implementation Effort:** **5–6 days** (because you'll need to implement LoRA yourself or adapt it from similar architectures).
- **Colab Pro Fit:** B (borderline; needs careful memory management).

#### **LTX-Video** (Lightricks)
- **Architecture:** Diffusion transformer, optimized for fast inference. Smaller params than HunyuanVideo (~4B claimed, but effective quality is good).
- **Weights:** HuggingFace: `Lightricks/LTX-Video`.
- **VRAM:** Inference ~8–10 GB. Fine-tuning: ~14–18 GB.
- **Fine-tuning Approach:** Limited public code for fine-tuning; may require reverse-engineering or adapting from other transformers.
- **Realistic Clip Length:** 5–6 seconds.
- **Math Hooks:** Flash Attention 2, why kernel fusion in attention reduces wall-clock time for temporal operations.
- **Implementation Effort:** **4–5 days** (if LoRA doesn't exist, you're adapting other frameworks).
- **Colab Pro Fit:** A- (reasonable VRAM; inference is fast, but fine-tuning may be underdocumented).

#### **Wan 2.1** (Alibaba, DynamiCrafter successor)
- **Architecture:** Text-to-video, recent (Q1 2025). Built on DynamiCrafter architecture but with improvements.
- **Weights:** HuggingFace: `Alibaba-DAMO/VideoComposer` and `Alibaba-DAMO/DynamiCrafter` (baseline).
- **VRAM:** ~12–16 GB inference, ~18–22 GB fine-tuning.
- **Fine-tuning Approach:** Code for fine-tuning exists but is sparse. Expect to adapt from DynamiCrafter tutorials.
- **Implementation Effort:** **4–5 days** (documentation is sparse; you're translating from similar projects).
- **Colab Pro Fit:** B (doable, but tight).

#### **ModelScope Text-to-Video** (Damo Academy, Alibaba)
- **Architecture:** Older (2023), but stable. Diffusion-based, text-to-video.
- **Weights:** HuggingFace: `damo-vilab/text-to-video-ms-1.7b`.
- **VRAM:** Inference ~8 GB. Fine-tuning: ~12–16 GB.
- **Fine-tuning Approach:** Documented fine-tuning code exists (LoRA available).
- **Realistic Clip Length:** 4–6 seconds.
- **Math Hooks:** Standard diffusion process, temporal MLP for frame fusion.
- **Implementation Effort:** **2–3 days** (code is mature; lots of tutorials).
- **Colab Pro Fit:** A (lightweight, reliable, but lower quality than newer models).

---

### Tier 3: "Proof-of-Concept Only (Skip if Aiming for A)"

#### **Zeroscope v2** (Nateraw)
- **Architecture:** Lightweight text-to-video (256x448). Very simple, fast.
- **VRAM:** Inference ~4–6 GB.
- **Realistic Clip Length:** 2–4 seconds.
- **Implementation Effort:** 1–2 days.
- **Rubric Fit:** Too simple for a final project. Use only as a last-resort fallback.

---

## 3. The Realistic 7-Day Deliverable: A 30–60s "Ideal Temilola Movie" Teaser Trailer

**Spoiler: You will NOT produce a 90-minute film.** That would require:
- Weeks of fine-tuning
- Petabytes of video data
- Enterprise compute
- Narrative coherence models (LLMs for screenwriting, video understanding models to stitch coherent scenes)

Instead, here's what you **CAN and SHOULD** ship:

### Proposed Pipeline: Poster → Keyframes → Motion Clips → Stitched Teaser

```
⭐8.5+ Taste Centroid (from Pipeline 2: embeddings + GenMatch)
                 ↓
         SDXL + LoRA* (fine-tuned on favorite-movie posters)
                 ↓
   Generate 5–10 Diverse Keyframe Images
   (different moods, color palettes, compositions)
                 ↓
         AnimateDiff + LoRA (fine-tuned on style samples)
                 OR
         Stable Video Diffusion (inference + clever prompting)
                 ↓
   Each Keyframe → 2–8 Second Motion Clip
   (smooth camera pans, depth-aware motion, etc.)
                 ↓
         ffmpeg + moviepy (crossfades, transitions)
                 ↓
   Stitched 30–60s Teaser Clip
                 ↓
         MusicGen (Meta, open-source transformer-based audio)
   Generate Royalty-Free Soundtrack
                 ↓
    ffmpeg: Overlay Audio + Title Card
                 ↓
    Final Deliverable: 30–60s AI-Generated Movie Teaser
    (+ poster, math derivation, rubric narrative)
```

*You already have SDXL+LoRA from Pipeline 2; reuse it.*

### Why This Works:

1. **Deliverable is Concrete:** An actual MP4 file that plays in a browser. Prof. Watson sees moving, styled imagery.
2. **Novel ML:**
   - SDXL+LoRA (already done) conditioned on taste centroid
   - AnimateDiff (NEW) with motion module LoRA conditioned on style
   - MusicGen transformer-based audio generation (NEW)
   - Latent-space interpolation (SLERP through keyframe embeddings for smooth transitions)
3. **Rubric Alignment:** Every component is differentiable, math-derivable, trainable.
4. **Realistic Timeline:** 7 days, Colab Pro, no explosion.

### Realistic Specs:

| Component | Duration | Quality | Notes |
|-----------|----------|---------|-------|
| Each Keyframe Video | 2–8 sec | 512×512 to 768×768 | 8–16 fps to stay under VRAM |
| Number of Keyframes | 5–10 | - | Gives you 10–80 sec raw footage |
| Stitched Teaser | 30–60 sec | - | With crossfades, music, title card |
| Music | 40–60 sec | 16 kHz stereo | Generated via MusicGen |
| Title Card | 3–5 sec | - | LLM-generated title or typographic GAN |

---

## 4. Architecture Diagram: ACT III Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ PIPELINE 3: PERSONAL VIDEO GENERATION ACT III                      │
└─────────────────────────────────────────────────────────────────────┘

INPUT: Temilola's ⭐8.5+ Movie Taste
       (titles, genres, ratings from Pipeline 1-2)
         │
         ├─→ [Pipeline 2: GenMatch Counterfactuals]
         │    (estimated taste centroid in embedding space)
         │
         └─→ Taste Centroid Vector c ∈ ℝ^768

STAGE 1: Keyframe Poster Generation
─────────────────────────────────────
         c (taste centroid)
         │
         ├─→ [SDXL + LoRA Fine-tuned on ⭐8.5+ Posters]*
         │   *Already complete from Pipeline 2.
         │    Fine-tuned to map c → visually coherent posters
         │
         ├─→ Prompt 1: "Movie poster for Temilola's ideal sci-fi epic"
         ├─→ Prompt 2: "...ideal romantic drama"
         ├─→ Prompt 3: "...ideal noir thriller"
         │    (5–10 diverse semantic directions)
         │
         └─→ 5–10 Keyframe Images
             I_1, I_2, ..., I_k ∈ ℝ^(512×512×3)

STAGE 2: Motion Generation (Choose A or B or Both)
──────────────────────────────────────────────────
Option A: AnimateDiff Path (RECOMMENDED)
         ┌─────────────────────────────────────┐
         │ For each keyframe I_j:               │
         │ ┌─ Input: I_j (poster image)        │
         │ ├─ Encoder: SDXL VAE → z_0          │
         │ └─ Diffusion Reverse Process:       │
         │    (with AnimateDiff motion module  │
         │     fine-tuned on style samples)    │
         │     z_T → ... → z_0 → x_video       │
         │ Output: 2–8 sec motion clip C_j     │
         └─────────────────────────────────────┘

Option B: Stable Video Diffusion Path
         ┌─────────────────────────────────────┐
         │ For each keyframe I_j:               │
         │ ├─ Input: I_j + optional text cond. │
         │ ├─ SVD Diffusion (25–50 frames)     │
         │ └─ Outputs 4–8 sec motion clip C_j  │
         └─────────────────────────────────────┘

STAGE 3: Video Stitching & Transitions
──────────────────────────────────────
         C_1, C_2, ..., C_k (motion clips)
         │
         ├─→ ffmpeg / moviepy
         │   - Crossfade: 0.5–1.0 sec overlaps
         │   - Optional: Ken Burns zoom in latent space
         │
         └─→ Stitched Video 30–60 sec (V_stitched)

STAGE 4: Soundtrack Generation (Bonus)
──────────────────────────────────────
         Temilola's Taste Centroid c
         │
         ├─→ [MusicGen Transformer]
         │   (Meta, open-source)
         │   Text prompt derived from movie preferences
         │   "Epic, cinematic soundtrack for a sci-fi romance"
         │
         └─→ Royalty-free audio A (40–60 sec)

STAGE 5: Composition & Finalization
───────────────────────────────────
         V_stitched + A (audio)
         │
         ├─→ ffmpeg overlay
         │   (audio sync, normalize loudness)
         │
         ├─→ Title Card (3–5 sec)
         │   - LLM-generated title (e.g., "The Ideal Temilola Epic")
         │   - Optional: GAN-generated typography
         │
         └─→ FINAL OUTPUT: 30–60s MP4
             "Temilola's Ideal Movie Teaser"
             (plays in browser, shows on MEGA_PITCH)

```

---

## 5. Math Hooks for the Rubric (MLMath LO 5)

### Why Each Component Earns "Novel ML" Credit

#### 5.1 Stable Video Diffusion / Diffusion Math in Video

**Covered in Class:** Sessions 20–21 (Diffusion, Score Matching, Autoencoders)

**Novel Extension:** Temporal dimension
- Forward process (noising):
  ```
  q(x_t | x_0) = α̅_t x_0 + √(1 - α̅_t) ε,    where ε ~ N(0, I)
  ```
  is generalized to video:
  ```
  q(V_t | V_0) applies per-frame, but frames are coupled via:
  - Frame consistency regularization: L_consistency = ∑_t ||∇_t V_t||^2
  - Optical flow coherence: L_flow = TV(flow(V_t, V_{t+1}))
  ```

- Score matching in 4D (spatial + temporal):
  ```
  s_θ(V_t, t) ≈ -∇_V log p(V_t)
  For video, this requires 3D convolutions with causal temporal masking.
  ```

- Classifier-free guidance (extended to video):
  ```
  ε̃ = ε_∅ + w_s · (ε_c - ε_∅)   [spatial]
      + w_t · (temporal correction term)
  ```

**Your Derivation:**
- Show the full video diffusion SDE: `dV = f(V, t)dt + g(t)dW_t`
- Derive the probability flow ODE for fast sampling
- Show why frame consistency loss is necessary (without it, flickering)
- Math: 2–3 pages, with diagrams showing noise schedule per frame

---

#### 5.2 AnimateDiff: Temporal Attention & Motion Module Decomposition

**Covered in Class:** Session 23 (Attention, Transformers)

**Novel Extension:** Factored attention in time
- Standard spatial cross-attention (from SDXL):
  ```
  Attention(Q, K, V) = softmax(QK^T / √d_k) V
  For image: Q, K, V ∈ ℝ^(HW × d_k)
  ```

- AnimateDiff adds temporal cross-attention layer:
  ```
  Temporal_Attn(Q, K, V) operates on the frame-time dimension.
  Q_t, K_t, V_t ∈ ℝ^(T × d_k) where T = number of frames

  FrameConsistency_Attn = softmax(Q_t K_t^T / √d_k) V_t
  ```

- Why freezing spatial layers works (your key insight):
  ```
  Spatial Prior (from SDXL pre-training): "render valid images"
  Temporal Layers (newly trained): "add motion coherence"

  By freezing SDXL's conv/attn weights, we preserve learned distribution
  p(image | text). Temporal layers learn p(motion | images).

  Joint: p(video | text) = p(motion | images) · p(image | text)
  ```

- LoRA Decomposition for Motion Module:
  ```
  W_temporal ≈ W_0 + ΔW = W_0 + AB^T
  where A ∈ ℝ^(d × r), B ∈ ℝ^(d_k × r), r << d (typically r=4 or 8)

  This is why LoRA fine-tuning only needs 12–16 GB (instead of 24 GB for full fine-tune).
  ```

**Your Derivation:**
- Expand the temporal attention math from scratch
- Show why rank-r approximation doesn't hurt motion quality
- Empirical: compare full fine-tune vs LoRA on validation clips (measure LPIPS, optical flow consistency)
- Math: 2–3 pages + diagrams of the temporal attention head

---

#### 5.3 Optical Flow & Video Consistency Losses

**Covered in Class:** Session 21 (VAE, latent spaces), Session 23 (Attention)

**Novel Extension:** Enforcing temporal coherence via optical flow
- Traditional loss (pixel space):
  ```
  L_mse = ||V_generated - V_reference||^2_2
  Problem: allows flickering (frame-to-frame inconsistency)
  ```

- Optical flow-based loss:
  ```
  F = OpticalFlow(V_t, V_{t+1})    [estimate motion between frames]

  L_consistency = ||OpticalFlow(V_generated) - OpticalFlow(V_reference)||_2

  Intuition: enforce that motion vectors match, even if pixels don't exactly match.
  This is crucial for video models — perceptually, smooth motion > pixel-perfect detail.
  ```

- Warping loss (projection-based):
  ```
  V_t^warped = Warp(V_t, F_{t→t+1})    [move pixels using flow]
  L_warp = ||V_t^warped - V_{t+1}||_2

  Forces the model to generate motion that satisfies forward-consistency.
  ```

**Your Derivation:**
- Show how optical flow is computed (Lucas-Kanade, PWCNet, or RAFT)
- Derive the warp operation: bilinear sampling with flow field
- Show empirically that L_consistency reduces flicker (measure temporal variance)
- Math: 1–2 pages

---

#### 5.4 MusicGen: Transformer-Based Audio Generation

**Covered in Class:** Session 24 (Transformer LLMs, Attention, Autoregressive Decoding)

**Novel Extension:** Discrete audio tokens via EnCodec
- EnCodec (Meta, open-source codec):
  ```
  Audio Waveform x(t) ∈ ℝ^(sample_rate × duration)
  → EnCodec Encoder → Discrete Codes z ∈ ℤ^(num_codebooks × time_steps)
  (typically 8 codebooks, 50 Hz quantization for 16 kHz audio)
  ```

- MusicGen as a Transformer LM over codes:
  ```
  Condition: text prompt "epic cinematic soundtrack"
  Input: text embedding e ∈ ℝ^d

  Generate codes autoregressively:
  p(z_1:T | e) = ∏_t p(z_t | z_1:t-1, e)

  Each code step is a Transformer decoder layer with:
  - Cross-attention to prompt embedding
  - Self-attention over previously generated codes
  - Output: logits over 2048 discrete audio tokens (per codebook)
  ```

- Why discrete tokens work (your key insight):
  ```
  Continuous waveform generation is hard (high variance, long sequences).
  Discrete codes reduce entropy: you're doing LM-style next-token prediction
  in a learned audio latent space.

  This is why transformers work well: text prompts are discrete tokens,
  audio codes are discrete tokens → same architecture.
  ```

**Your Derivation:**
- Show EnCodec architecture (audio codec with learned quantization)
- Derive the MusicGen transformer: standard causal transformer with cross-attention
- Show why the loss is standard cross-entropy over discrete tokens
- Empirically: measure audio quality (Mel-spectrogram distance, MusicCaps evaluation score)
- Math: 2–3 pages, with spectrograms showing generated audio

---

#### 5.5 Taste Centroid → Latent Conditioning (Your Novel Contribution)

**Not Covered in Class:** This is YOUR idea.

**Novel Math:** Condition video generation on personal embedding
- From Pipeline 2 (GenMatch), you have:
  ```
  Taste Centroid c = mean(encoder(movies with ⭐8.5+))
  c ∈ ℝ^768 (SDXL latent space or similar)
  ```

- Instead of text prompt, use c directly:
  ```
  Standard Text Prompt: "epic sci-fi movie poster"
  → CLIP text encoder → e_text ∈ ℝ^768

  Your Novel Approach: Use c directly (no text prompt)
  → SDXL conditioned on c_projected ∈ ℝ^768

  Optionally: SLERP interpolation through latent space
  c_morph(t) = SLERP(c_start, c_end, t)
  Generate keyframe at c_morph(t) for smooth taste transition.
  ```

- Math:
  ```
  SLERP (Spherical Linear Interpolation):
  SLERP(a, b, t) = [sin((1-t)θ) / sin(θ)] a + [sin(tθ) / sin(θ)] b
  where θ = arccos(a · b / (||a|| ||b||))

  Intuition: moves smoothly on the sphere of taste embeddings,
  not in Euclidean space (which can collapse to one dominant taste).
  ```

**Your Derivation:**
- Show why SLERP is better than linear interpolation for embeddings
- Empirically: generate 5 keyframes at SLERP points between favorite genres (e.g., sci-fi → romantic drama)
- Visualize latent space: t-SNE plot of your movie ratings, show the taste centroid, show the SLERP path
- Math: 1–2 pages, with visualization

---

## 6. Apples-to-Apples: Class-Baseline Pairings

**How each video component extends what you covered in CS156:**

| Component | Class Baseline | Novel Extension |
|-----------|---|---|
| **Stable Video Diffusion** | Session 21: Diffusion, VAE, Score Matching | Forward diffusion SDE on video; 3D convolutions; temporal attention for frame consistency |
| **AnimateDiff** | Session 23: Attention, Transformers | Temporal cross-attention layers; why freezing spatial priors works; LoRA efficiency for video |
| **Optical Flow Loss** | Session 21: VAE latent space | Perceptual loss in motion space (flow vectors); warping-based consistency |
| **MusicGen** | Session 24: Transformer LLMs | Discrete audio token generation; cross-modal attention (text → audio codes) |
| **Taste Centroid Conditioning** | Sessions 22–24: Embeddings, Autoencoders, Contrastive Learning | Latent-space conditioning instead of text; SLERP interpolation through taste space |

---

## 7. Wild Card / Showstopper Idea: "The Morph"

Here's the idea that will make Prof. Watson pause:

### "Temilola's Taste Morphology: A Journey Through Your Movie Mind"

**Concept:** Instead of a static 30-second teaser, generate a **1–2 minute video that morphs through your top 5 favorite movie aesthetics in latent space.**

**How it works:**
1. From Pipeline 2, extract 5 most-loved movies (⭐9.5–10.0). Encode them into SDXL latent space.
2. Compute the taste centroid c.
3. For each of the 5 movies, compute the vector v_i = (movie_embedding - c) / ||...||.
4. Generate keyframes at interpolation points:
   ```
   SLERP path 1: c + 0.0 · v_1 → c + 0.5 · v_1 → c + 1.0 · v_1
   SLERP path 2: c + 1.0 · v_1 → c + 0.8 · v_1 + 0.2 · v_2 → ... → c + 1.0 · v_2
   (smooth path through the taste pentagon)
   ```
5. Use AnimateDiff to animate each interpolated keyframe (motion within the taste centroid neighborhood).
6. Stitch with crossfades.

**Result:** A video that shows Temilola's movie taste space as a **visual journey**. The aesthetics gradually shift from "your sci-fi favorite" through "the taste centroid" to "your romantic drama favorite."

**Why it's a showstopper:**
- You're not just generating a video; you're **visualizing your personal taste geometry.**
- It directly uses Pipeline 2's embedding work (GenMatch centroid, counterfactual cohorts).
- The math is elegant: SLERP, latent interpolation, diffusion guidance in moving latent space.
- It's a *narrative* about Temilola, not a generic movie teaser.

**Rubric:** "Novel personalization" (MLMath LO 5), "integration across pipelines" (Systems LO 4).

**Difficulty:** Hard, but doable by Day 5 if AnimateDiff works smoothly.

**Fallback:** If SLERP is too complex, just morph through 3 aesthetics (e.g., sci-fi → drama → horror). Still cool.

---

## 8. Concrete 7-Day Execution Plan

### Day 1–2: SDXL + LoRA Refinement (Tuesday–Wednesday)

**Status:** You may have finished this in Pipeline 2. If not:
- Load your ⭐8.5+ movie posters (from TMDB).
- Fine-tune SDXL on them with LoRA (rank 8, learning rate 1e-4, 500–1000 steps).
- Save checkpoint at `./models/sdxl_taste_lora.safetensors`.
- **Deliverable:** A test batch of 10 generated posters, one for each movie taste direction.

**If you're skipping (because SDXL is done):** Use Day 1 to set up code infrastructure:
- Jupyter notebooks for each stage (keyrames.ipynb, animatediff.ipynb, stitch.ipynb).
- Install dependencies: diffusers, AnimateDiff, ffmpeg-python, moviepy, MusicGen.
- Create directory structure: `./input/posters`, `./output/keyframes`, `./output/clips`, `./output/final`.

**Effort:** 1–2 Colab Pro A100 sessions, ~6–8 GPU hours.

---

### Day 3: AnimateDiff Integration & Keyframe → Motion Pipeline (Thursday)

**Goal:** Turn 10 keyframe posters into 10 short video clips.

**Steps:**
1. Load SDXL+LoRA checkpoint.
2. Load AnimateDiff motion module from HuggingFace.
3. For each keyframe image:
   - Encode into SDXL VAE latent space.
   - Run diffusion reverse with AnimateDiff (temporal guidance enabled).
   - Decode from latent → video frames.
   - Export as MP4 (2–8 seconds, 8 fps).
4. Save outputs to `./output/clips/animated_*.mp4`.

**Hyperparameters to tune:**
- Guidance scale: 7.5–15 (higher = more style fidelity, risk of artifacts)
- Num frames: 16 (2 sec at 8 fps)
- Steps: 50 (balance quality vs speed)

**Expected VRAM:** 8–12 GB. Should fit in A100 40GB easily.

**Deliverable:** 10 × 2-sec clips (20 sec raw footage).

**Effort:** 1 Colab session, ~4 GPU hours (most time is encoding/decoding).

---

### Day 4: Stable Video Diffusion as Alternative (Friday)

**Goal:** Produce an alternative motion pipeline for comparison and fallback.

**Steps:**
1. Load SVD-XT model from HuggingFace.
2. For 3 keyframes (as a test):
   - Feed to SVD as image input.
   - Generate 25-frame video (4 sec at 8 fps).
3. Compare outputs with AnimateDiff (visual quality, flicker, style retention).
4. If SVD is better: use it for the remaining 7 keyframes.
5. If AnimateDiff is better: save SVD as fallback.

**Why dual-path?** Insurance. If one breaks, you have the other.

**Deliverable:** 10 × 4-sec clips via SVD (or mixed AnimateDiff + SVD).

**Effort:** 0.5 Colab session, ~2 GPU hours (SVD inference is fast).

---

### Day 5: MusicGen + Stitching (Saturday)

**Goal:** Generate soundtrack and assemble teaser.

**Steps:**
1. Load MusicGen model (`facebook/musicgen-medium`).
2. Generate 60-sec audio track:
   - Prompt: "Epic cinematic soundtrack, orchestral, sci-fi + romance vibes, builds to crescendo"
   - Top-p sampling: 0.9 (balance coherence vs diversity)
3. Stitch video clips with ffmpeg:
   ```bash
   ffmpeg -i clip1.mp4 -i clip2.mp4 ... -filter_complex \
     "[0:v] fade=d=1:t=out [v0]; [1:v] fade=d=1:t=in [v1]; \
      [v0][v1] xfade=transition=fade:duration=1 [v]" \
     -map "[v]" stitched.mp4
   ```
4. Overlay audio:
   ```bash
   ffmpeg -i stitched.mp4 -i track.wav \
     -c:v copy -c:a aac -shortest output_with_audio.mp4
   ```
5. Add 3–5 sec title card (static image with text overlay).

**Deliverable:** 30–60 sec MP4 with video + soundtrack + title.

**Effort:** 0.5 Colab session, ~1 GPU hour (MusicGen is CPU-friendly).

---

### Day 6: Math Derivation & Visualization (Sunday)

**Goal:** Document the math that makes this novel.

**Output:** A `math_derivation.md` document (4–6 pages).

**Sections:**
1. Diffusion SDE in video (1 page).
2. AnimateDiff temporal attention (1 page).
3. Optical flow consistency (0.5 page).
4. MusicGen discrete tokens (1 page).
5. Taste centroid SLERP (0.5 page).
6. Empirical results: figures showing generated clips, audio spectrograms, latent space visualization.

**Also create:**
- Latent space t-SNE plot: your movies' embeddings + taste centroid.
- Comparison table: AnimateDiff vs SVD (quality, speed, VRAM).
- Optical flow visualization: show motion consistency in generated clips.

**Effort:** 1–2 hours (no GPU).

---

### Day 7: Integration, Polish, MEGA_PITCH Narrative (Monday)

**Goal:** Ship the final product + narrative.

**Deliverables:**
1. **Video:** `temilola_movie_teaser_30sec.mp4` (uploaded to Google Drive for MEGA_PITCH).
2. **Poster:** `keyframes_grid.png` (the 10 generated keyframes, 5×2 grid).
3. **Metadata:** JSON file with:
   - Model checksums (SDXL, AnimateDiff, SVD, MusicGen versions)
   - Hyperparameters used
   - Hardware info (Colab Pro A100, session duration)
4. **Math Document:** `pipeline3_math_derivation.md`.
5. **Integration Narrative:** Update the main Pipeline 3 README to link all components and explain the full architecture.

**MEGA_PITCH Angle:**
> "Pipeline 3: From Taste Centroid to Teaser Trailer. Using personalized SDXL+LoRA, AnimateDiff motion modules fine-tuned on style, and latent-space interpolation, I generated a 30–60 second movie teaser that visually embodies my ⭐8.5+ taste profile. The teaser morphs through my favorite genres in latent space (sci-fi → drama → horror), with an AI-composed orchestral soundtrack. Built on open-source diffusion models; all components are mathematically derived and fine-tuned. Result: a personalized, generative narrative artifact."

**Effort:** 2–4 hours (writing + video encoding).

---

## 9. Risks, Fallbacks, and Contingencies

### Risk 1: AnimateDiff Takes Too Long / Memory Explodes

**Symptoms:** CUDA OOM, session crashes, generation takes >15 min per frame.

**Fallback (Day 4):**
- Switch entirely to Stable Video Diffusion inference (no fine-tuning).
- SVD is faster (5–10 min per clip), even if quality is slightly lower.
- You still get the teaser; rubric still satisfied.

**Mitigation:**
- Start AnimateDiff on **one test keyframe** on Day 3 morning. If it explodes, pivot to SVD by afternoon.
- Have SVD fully working by Day 4 EOD as insurance.

---

### Risk 2: Video Stitching / Ffmpeg Hell

**Symptoms:** Crossfades glitch, audio sync is off, color space mismatch.

**Fallback:**
- Assemble a **5-second montage** instead (just 3 clips with simple cuts, no fades).
- Pre-export all clips with identical codec (H.264 AAC) before stitching.
- Test ffmpeg locally first; only then run on Colab.

**Mitigation:**
- Use `moviepy` (Python lib) instead of raw ffmpeg if you hit bash issues.
- On Day 5, spend 1 hour just testing ffmpeg on dummy videos.

---

### Risk 3: MusicGen Produces Garbage Audio

**Symptoms:** Generated track is incoherent, loops awkwardly, doesn't match mood.

**Fallback:**
- Use a royalty-free audio sample instead (Epidemic Sound, Artlist, or Creative Commons).
- Load from `./audio/fallback_epic_soundtrack.wav`.
- You lose the "generative audio" novelty, but the video still stands.

**Mitigation:**
- On Day 5, generate 3 versions of the soundtrack with different prompts.
- Listen to all 3; pick the best one (even if imperfect).
- If all suck, fall back to royalty-free.

---

### Risk 4: Full Pipeline Doesn't Converge by Day 7

**Symptoms:** Keyframes are incoherent, motion doesn't stick to the original image, something's broken.

**Fallback (Nuclear Option):**
- **Ship the SDXL+LoRA keyframes as a static image grid.** (Already novel, already fine-tuned.)
- Add the math derivation showing *why* the full pipeline should work (even if you ran out of time).
- Write a section: "Future Work: Temporal Extension."
- You still get credit for novel diffusion fine-tuning; the video is just... not included.

**Realistic Odds:** If you follow the day-by-day plan, you'll have something working by Day 6. Day 7 is just polish.

---

### Risk 5: Colab Pro Session Gets Killed

**Symptoms:** "You've been disconnected" message, work is lost.

**Mitigation:**
- Save checkpoints every 2 hours to Google Drive.
- Use `nbconvert` to backup Jupyter notebooks as .py files.
- Store all outputs (keyframes, clips, audio) in Drive as they're generated.
- Restart sessions proactively before they get auto-killed (Colab kills sessions after 12–24 hrs of idle inference).

---

## 10. Why This Plan Wins

| Criterion | Why You Win |
|-----------|---|
| **Novelty (MLMath LO 5)** | AnimateDiff motion LoRA + taste centroid SLERP conditioning + custom optical flow loss. Clear derivation of temporal attention math. |
| **Integration** | Builds directly on Pipeline 2 (SDXL+LoRA). Reuses taste centroid, GenMatch embeddings. Seamless flow. |
| **Ambition** | Video is genuinely hard. Most students won't attempt it. Prof. Watson will notice. |
| **Feasibility** | Honest timeline. Realistic VRAM. Fallbacks at each stage. You will ship *something*. |
| **Rubric Alignment** | Hits all major LOs: MLMath (diffusion, transformers, embeddings), Systems (multi-stage pipeline), Novel personalization (taste-based conditioning). |
| **Storytelling** | "Turning my movie taste into an actual movie" is a killer narrative. MEGA_PITCH gold. |

---

## 11. Quick Reference: Model Selection Decision Tree

```
Do you want to ship video by Apr 23?
├─ YES
│  ├─ Do you have 4+ hours right now to set up one model?
│  │  ├─ YES
│  │  │  └─ Use AnimateDiff (recommended)
│  │  │     (or SVD if AnimateDiff explodes)
│  │  └─ NO
│  │     └─ Use Stable Video Diffusion (faster setup)
│  │
│  └─ Budget = $0 (free inference only)?
│     ├─ YES
│     │  └─ AnimateDiff + SVD (open-source)
│     └─ NO
│        └─ Could also use Runway Gen-3 API (if you want to)
│
└─ NO (you want to build a "showstopper" and have 2+ weeks)
   └─ Use HunyuanVideo or Mochi 1
      (better quality, but overkill for 7 days)
```

**For you:** AnimateDiff → SVD backup. Go.

---

## 12. File Structure & Git Setup

```
Pipeline 3/
├── README.md                          # Main entry point
├── research_video_generation.md        # This file
├── math_derivation.md                 # To be written Day 6
├── pipeline3_final_narrative.md        # To be written Day 7
│
├── notebooks/
│   ├── 1_keyframes.ipynb              # SDXL+LoRA keyframe generation
│   ├── 2_animatediff_motion.ipynb      # AnimateDiff + LoRA fine-tuning
│   ├── 3_svd_fallback.ipynb            # SVD inference backup
│   ├── 4_musicgen_audio.ipynb          # MusicGen soundtrack
│   └── 5_stitch_final.ipynb            # ffmpeg stitching + assembly
│
├── config/
│   ├── sdxl_lora_config.yaml           # Fine-tuning hyperparams
│   ├── animatediff_config.yaml         # Motion module hyperparams
│   └── models.yaml                     # Model URLs, versions
│
├── models/
│   ├── sdxl_taste_lora.safetensors     # SDXL LoRA checkpoint
│   ├── animatediff_motion_lora.pt      # AnimateDiff LoRA checkpoint
│   └── (don't store full models; load from HF)
│
├── input/
│   ├── posters/                        # Keyframe poster images
│   └── taste_centroid.pkl              # Saved embedding vector
│
├── output/
│   ├── keyframes/
│   │   ├── keyframe_00.png
│   │   ├── keyframe_01.png
│   │   └── ...
│   ├── clips/
│   │   ├── clip_00.mp4
│   │   ├── clip_01.mp4
│   │   └── ...
│   ├── audio/
│   │   └── soundtrack.wav
│   └── final/
│       └── temilola_movie_teaser_30sec.mp4  # THE DELIVERABLE
│
└── utils/
    ├── latent_interpolation.py         # SLERP, interpolation helpers
    ├── optical_flow.py                 # Flow visualization
    └── ffmpeg_stitcher.py              # Video assembly
```

---

## 13. Final Sanity Check

**Temilola, before you start:**

1. **Do you have working SDXL+LoRA from Pipeline 2?** (Posters generated, LoRA weights saved)
   - If NO: Days 1–2 are for you. Days 3–7 get tighter.
   - If YES: Start Day 1 with code setup. Days 3–7 are cushioned.

2. **Are you comfortable with Colab Pro A100 session management?** (Saving to Drive, checkpoints, restarts)
   - If NO: Spend 1 hour on Day 1 learning this. It'll save your life.

3. **Do you have realistic expectations on video quality?**
   - Expect: 512×512, 8 fps, 2–8 sec clips. No 4K, no cinema-grade motion.
   - This is still genuinely impressive for an open-source 7-day sprint.

4. **Are you committing to AnimateDiff + SVD as the core approach?**
   - If YES: You're on track.
   - If you want HunyuanVideo or "the absolute best quality": nope, too risky on timeline.

5. **Is the taste-centroid SLERP wild card on your radar for Day 5–6?**
   - If you have spare time: implement it. It's the showstopper.
   - If you're cutting it close: skip it; the teaser alone is solid.

**You've got this.** The math is solid, the models are real, the timeline is ruthlessly honest. Go make Temilola's movie.

---

## Appendix: Useful Links & Commands

### Model Weights (HuggingFace)
- **SDXL:** `stabilityai/stable-diffusion-xl-base-1.0`
- **AnimateDiff:** `guoyww/animatediff-motion-adapter-v1-5`
- **SVD:** `stabilityai/stable-video-diffusion-img2vid-xt`
- **MusicGen:** `facebook/musicgen-medium` or `facebook/musicgen-large`
- **CogVideoX (alternative):** `THUDM/CogVideoX-2b`

### Quick Setup (Colab)
```bash
!pip install diffusers transformers accelerate torch torchvision torchaudio
!pip install moviepy ffmpeg-python
!pip install xformers  # Crucial for memory efficiency
!git clone https://github.com/guoyww/AnimateDiff.git
```

### Benchmark Specs (for your notes)
| Model | Params | Inference VRAM | Fine-tune VRAM | Clip Length | Quality |
|-------|--------|---|---|---|---|
| AnimateDiff (motion module) | 75M | 6–10 GB | 12–16 GB | 2–8 sec | A (motion) / A- (style) |
| SVD-XT | 8.7B | 8–12 GB | 16–24 GB | 4–8 sec | A |
| CogVideoX-2B | 2B | 10 GB | 14–18 GB | 6–8 sec | B+ |
| HunyuanVideo | 13B | 18–22 GB | 24–32 GB | 5 sec | A+ |
| MusicGen-Large | 3.9B | 6 GB (CPU OK) | 12 GB | 30 sec | A- (music) |

---

## Glossary

- **LoRA:** Low-Rank Adaptation. Fine-tune only A & B matrices (rank-r), freeze original weights.
- **SLERP:** Spherical Linear Interpolation. Smooth path on embedding sphere.
- **Optical Flow:** Vector field showing per-pixel motion between frames.
- **Diffusion SDE:** Stochastic Differential Equation for forward/reverse diffusion process.
- **Cross-Attention:** Attention where query is from one modality (image), key/value from another (text).
- **Temporal Attention:** Cross-attention across time dimension (frames).
- **VAE:** Variational Autoencoder. Encodes images → compact latent space.
- **Guidance:** Classifier-free guidance. Amplify signal by mixing conditional & unconditional predictions.

---

**End of Research Document**

Generated by Claude Agent for CS156 Pipeline 3 (Apr 16, 2026)
Next Steps: Read Days 1–3 of the execution plan. Decide: AnimateDiff or SVD first? Let's go.
