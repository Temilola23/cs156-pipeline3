# Pipeline 3: The Generative Mega-Stack — Comprehensive Research Report

**Prepared for:** Temilola's CS156 Pipeline 3 (Prof. Watson, due Apr 23 2026)
**Objective:** Build a unified narrative for generating the "ideal movie" from latent taste embeddings — posters, plots, titles, soundtracks, even trailers.
**Focus:** All methods chosen for LO 5 potential (first-principles math + from-scratch feasibility + rubric impact).

---

## Executive Summary

This report details a coherent "Generative Mega-Stack" for Pipeline 3:

1. **Poster generation** via Stable Diffusion SDXL / Flux.1 + LoRA fine-tuning on Temilola's liked posters
2. **Plot/title generation** via small LLMs (Llama 3.2 1B/3B, Phi-3, Qwen2.5) + QLoRA fine-tuning on movie plots
3. **Soundtrack snippets** via MusicGen or AudioLDM conditioned on taste embeddings
4. **Video trailers** via LTX-Video or CogVideoX (7-day feasibility marginal; AnimateDiff safer)
5. **Centroid-to-movie generation** using Conditional VAEs or classifier-free guided diffusion
6. **Conditional architectures** spanning CVAE, conditional diffusion, ControlNet-guided layouts
7. **Wild card:** IP-Adapter style transfer + latent interpolation to "evolve" a generated poster along Temilola's taste manifold

All methods are paired with apples-to-apples class baselines (PCA centroid + nearest neighbor content recommendation; simple LLM zero-shot generation; etc.). The implementation arc is:

- **Phase A (Colab):** Embed all 162 movies + compute centroid. Fine-tune LoRA on 20–30 liked posters. Fine-tune small LLM on plot + title pairs.
- **Phase B (Colab):** Condition diffusion/LLM on centroid embedding + latent walks. Generate 5 posters, 1 plot, 1 title per centroid. Visualize latent traversal.
- **Phase C (optional, time permitting):** MusicGen snippet; AnimateDiff short loop; IP-Adapter style transfer.

---

## 1. Poster Generation: Stable Diffusion SDXL + LoRA + ControlNet

### 1.1 Why This Impresses the Rubric

- **MLCode:** Library (diffusers, kohya_ss) + from-scratch forward pass via latent diffusion equation
- **MLExplanation:** Score-matching diffusion loss; LoRA low-rank parameterization; zero-convolution initialization
- **MLFlexibility:** Transfer learning (frozen SDXL backbone); parameter-efficient fine-tuning (LoRA ~0.5% trainable params)
- **MLMath:** First-principles derivation of diffusion loss via score-matching; ELBO for VAE encoder; LoRA rank-constrained optimization

### 1.2 Method Description

**Stable Diffusion SDXL 2.0** is a 6.6B-param latent diffusion model:
- **Text encoder:** CLIP ViT-L/14 (frozen for LoRA)
- **U-Net denoiser:** 2.6B params (LoRA injects rank-4 adapters into attention layers)
- **VAE:** Encodes images to 8× downsampled latent space (reduces VRAM; ~384D per patch)

**LoRA fine-tuning** on Temilola's liked posters (assume ~25 images):
- Train low-rank matrices (rank r=4–8) injected into cross-attention layers of U-Net
- Typical training time: 30–60 min on A100 or 80 GB Colab GPU
- LoRA file size: ~50–100 MB (vs. 6.6 GB base model)
- **Cost:** ~400–600 A100-hours per image with careful tuning; feasible in 1–2 Colab sessions

**ControlNet** (optional, for poster layout control):
- Canny-edge or depth maps of liked posters → enforce composition consistency
- Zero-conv layers ensure stable training even with small datasets (< 50 images)

### 1.3 Mathematical Foundation

**Diffusion forward process** (time-conditioned noise addition):
$$q(x_t | x_0) = \sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

where $\{\alpha_t\}_{t=1}^{T}$ is a noise schedule.

**Score-matching objective** (training the reverse denoiser):
$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon_\theta(x_t, t, c) - \epsilon \|^2_2 \right]$$

where $c$ is the text/image conditioning (CLIP embedding) and $\epsilon_\theta$ is the learned noise predictor.

**LoRA parameterization** (low-rank injection):
$$W_{\text{attention}} = W_0 + \Delta W = W_0 + B A$$

where $A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times d}$ (r ≪ d), and only A, B are trained:
$$\text{Param count} \approx 2rd \ll d^2$$

### 1.4 Library Options & From-Scratch Feasibility

| Method | Library | From-Scratch? | Effort |
|--------|---------|---------------|--------|
| **SDXL inference** | `diffusers.StableDiffusionXLPipeline` | Easy (load + forward pass) | 15 min |
| **LoRA training** | `kohya_ss` or `peft.LoraConfig` | Moderate (custom training loop) | 4–6 hours |
| **ControlNet** | `diffusers.ControlNetModel` | Hard (U-Net skip connections) | 8+ hours |
| **VAE inversion** | `diffusers.AutoencoderKL.encode()` | Easy (1 forward + backward) | 30 min |

**Recommended for 7-day constraint:** Use `diffusers` library for inference + `peft.LoraConfig` for fine-tuning (simple, tested, low setup friction).

### 1.5 7-Day Implementation Estimate

| Task | Time |
|------|------|
| Environment setup + Colab GPU | 0.5 h |
| Collect/annotate 20–30 liked posters | 1 h |
| LoRA training (2–3 runs, tuning) | 4 h |
| Inference pipeline + prompt engineering | 2 h |
| Visualization (latent interpolation, comparison) | 1.5 h |
| **Total** | **9 h** (LOW-MEDIUM effort) |

### 1.6 Apples-to-Apples Baseline

**PCA centroid + nearest neighbor image retrieval:**
- Compute PCA on Temilola's 162 movie posters (ResNet-50 features, 2048D)
- Extract cluster centroid from K-means (e.g., k=5)
- Return 5 nearest-neighbor posters from dataset
- **Metrics:** Visual similarity (LPIPS), novelty (distance to training set)
- **Why inferior:** No generative capability; limited to existing images; no latent space exploration

---

## 2. Plot & Title Generation: Small LLMs + QLoRA Fine-Tuning

### 2.1 Why This Impresses the Rubric

- **MLCode:** HF `transformers` + `peft` (QLoRA) + from-scratch generation loop + decoding strategies (beam search, nucleus sampling)
- **MLExplanation:** Autoregressive cross-entropy loss; low-rank quantized adaptation; conditional text generation
- **MLFlexibility:** Transfer learning (pre-trained LLM); task-specific fine-tuning; interpretable token-level generation
- **MLMath:** Language model softmax likelihood; KL-divergence regularization in QLoRA; conditional sequence probability

### 2.2 Method Description

**Choice of model:** Llama 3.2 1B or 3B (open-source, permissive license, Colab-friendly)

- **1B variant:** ~1B params, ~4 GB loaded (8-bit), fits comfortably in Colab + other components
- **3B variant:** ~3B params, ~8 GB loaded; marginal quality increase; still feasible with memory optimization

**QLoRA fine-tuning** on movie plots:
- **Data source:** Temilola's 162 movies + public TMDB/MovieLens plot summaries (~50–100 word synopses per movie)
- **Task format:**
  ```
  Input:  "Movie rating: 4.5/5, genres: [Sci-Fi, Drama], tags: [introspective, philosophical]"
  Output: "[Generated plot synopsis 50–100 words]"
  ```
- **QLoRA:** Quantize base model to 4-bit, inject LoRA adapters (rank=8) into attention layers
- **Training:** ~500–1000 examples, 3–5 epochs, batch size 4–8 (gradient accumulation), ~2–4 hours on Colab

### 2.3 Mathematical Foundation

**Autoregressive language model likelihood** (token-by-token factorization):
$$p(y_1 \ldots y_n | x) = \prod_{i=1}^n p(y_i | y_{<i}, x)$$

where $x$ is the conditioning context (rating + genres + embedding) and $y$ is the plot text.

**Fine-tuning objective** (cross-entropy + L2 penalty on LoRA weights):
$$\mathcal{L} = -\sum_i \log p_\theta(y_i | y_{<i}, x) + \lambda \| A B^T \|^2_F$$

**QLoRA quantization trick** (4-bit NormalFloat for efficiency):
- Store weights in NF4 (normalized 4-bit), requantize to float32 during forward pass
- LoRA adapters remain in float32 for gradient computation
- **Memory savings:** ~75% reduction vs. LoRA; enables fitting 3B–7B models on 16–24 GB VRAM

### 2.4 Library Options & From-Scratch Feasibility

| Method | Library | From-Scratch? | Effort |
|--------|---------|---------------|--------|
| **Model loading** | `transformers.AutoModelForCausalLM` | Easy | 10 min |
| **QLoRA setup** | `peft.LoraConfig` + `bitsandbytes` | Easy | 30 min |
| **Fine-tuning loop** | Custom + HF `Trainer` | Moderate | 2–3 hours |
| **Inference** | `transformers.GenerationMixin` | Moderate (sampling, beam search) | 1 hour |
| **Decoding strategies** | From scratch (multinomial, top-k, nucleus) | Hard | 4+ hours |

**Recommended for 7-day constraint:** Use HF `Trainer` + `bitsandbytes` (battle-tested, stable, minimal debugging).

### 2.5 7-Day Implementation Estimate

| Task | Time |
|------|------|
| Environment + Llama 3.2 3B download | 1 h |
| Prepare plot dataset (Temilola + TMDB) | 1.5 h |
| QLoRA config + training loop | 3 h |
| Inference + prompt engineering | 1.5 h |
| Evaluation (ROUGE, manual inspection) | 1.5 h |
| **Total** | **9 h** (LOW-MEDIUM effort) |

### 2.6 Apples-to-Apples Baseline

**Zero-shot LLM generation (no fine-tuning):**
- Use Llama 3.2 1B in-context learning: provide 2–3 example (rating, genres) → plot pairs in prompt
- Generate plots for Temilola's taste centroid without any fine-tuning
- **Metrics:** ROUGE-L (n-gram overlap with human plots), perplexity, human rating
- **Why inferior:** No task-specific adaptation; generic template-like outputs; ignores Temilola's stylistic preferences

**Baseline code sketch:**
```python
prompt = """Example:
Rating: 4.0, Genres: [Sci-Fi, Drama] → A lone astronaut discovers an alien intelligence...

Rating: 4.5, Genres: [Drama, Mystery] → """
output = llm.generate(prompt, max_new_tokens=100)
```

---

## 3. Soundtrack Generation: MusicGen or AudioLDM

### 3.1 Why This Impresses the Rubric

- **MLCode:** Diffusers or Meta MusicGen API; latent audio diffusion; mel-spectrogram conditioning
- **MLExplanation:** Conditional audio diffusion; temporal consistency; acoustic feature embeddings
- **MLFlexibility:** Multi-modal conditioning (text + image embeddings); genre/mood control
- **MLMath:** Mel-scale filterbank; score-matching on latent audio; diffusion in spectrogram domain

### 3.2 Method Description

**MusicGen** (Meta, 350M–1.3B params):
- Autoregressive transformer decoder over compressed audio tokens (EnCodec, 4-level quantization)
- Inputs: text descriptions ("epic sci-fi orchestral theme") OR conditioning on Temilola's taste vector
- Outputs: 30-second stereo audio clips (16 kHz)
- **Advantage:** Lightweight, faster inference, well-tested

**AudioLDM 2** (alternative):
- Latent diffusion in learned audio feature space
- Conditioning: CLAP text encoder (unifies text + audio embeddings)
- **Advantage:** Richer, more diverse outputs; better at fine-grained mood control

### 3.3 Mathematical Foundation

**MusicGen autoregressive decoding:**
$$p(z_1 \ldots z_n | c) = \prod_{i=1}^n p(z_i | z_{<i}, c)$$

where $z_i$ ∈ {1, ..., K}^4 (4 quantization levels, K=1024 vocab per level) and $c$ is CLAP text embedding.

**AudioLDM latent diffusion:**
$$q(a_t | a_0) = \sqrt{\bar{\alpha}_t} \cdot \text{CLAP-encode}(a_0) + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

Denoising in latent space, then decoding to waveform via learned mel-vocoder.

### 3.4 Library Options & From-Scratch Feasibility

| Method | Library | From-Scratch? | Effort |
|--------|---------|---------------|--------|
| **MusicGen inference** | `audiocraft.models.MusicGen` | Easy | 15 min |
| **AudioLDM 2 inference** | `diffusers` + CLAP encoder | Easy | 30 min |
| **Conditioning on embeddings** | Custom prompt engineering | Medium | 1–2 hours |
| **Multi-modal fusion** | From scratch (CLAP + mel fusion) | Hard | 6+ hours |

### 3.5 7-Day Implementation Estimate

| Task | Time |
|------|------|
| Setup + model loading | 0.5 h |
| Craft 5–10 taste-descriptive prompts | 0.5 h |
| Generate 5 audio snippets (MusicGen) | 1 h |
| (Optional) Fine-tune via embeddings | 3–4 h |
| Audio visualization + subjective evaluation | 1 h |
| **Total** | **6–7 h** (LOW-MEDIUM effort) |

### 3.6 Apples-to-Apples Baseline

**Genre-conditional random selection:**
- Extract dominant genre(s) from Temilola's centroid cluster
- Randomly select a free-licensed background music track from Freesound or YouTube Audio Library matching genre
- **Metrics:** Genre match (manual), mood alignment (manual + audio feature similarity), novelty (distance to training soundtrack embeddings)
- **Why inferior:** No generative capability; no personalization; no synthesis of novel compositions

---

## 4. Trailers / Video Generation: LTX-Video, CogVideoX, Mochi vs. AnimateDiff

### 4.1 Why This Impresses the Rubric

- **MLCode:** Diffusers video generation pipelines; asymmetric diffusion transformers; temporal consistency losses
- **MLExplanation:** Frame-to-frame denoising; 3D VAE convolutions; temporal attention across frames
- **MLFlexibility:** Image-to-video + text-to-video conditioning; LoRA-based style control
- **MLMath:** 3D diffusion over space + time; temporal positional embeddings; optical flow regularization

### 4.2 Method Landscape (April 2026)

#### **LTX-Video** (Lightricks)
- **Strengths:** 1216×704 @ 30 fps in real-time on 12 GB VRAM; optimized for speed
- **Weakness:** Shorter clips (~8–12 sec); less mature ecosystem
- **Feasibility:** MEDIUM (requires custom inference pipeline; community support growing)

#### **CogVideoX** (Tsinghua/Zhipu)
- **Strengths:** High-quality 6-second clips (720×480); open-source; 3D VAE + expert transformer
- **Weakness:** Longer inference time; smaller community
- **Feasibility:** MEDIUM-HIGH (HF integration exists; good documentation)

#### **Mochi 1** (Genmo)
- **Strengths:** 10B params; balanced realism + flexibility; excellent text consistency
- **Weakness:** Commercial licensing concerns; largest model (memory-heavy)
- **Feasibility:** MEDIUM (API-based inference available; from-scratch fine-tuning hard)

#### **AnimateDiff** (2023, but still useful)
- **Strengths:** Stable Diffusion ecosystem; loops seamlessly; extensive community tooling (ComfyUI)
- **Weakness:** No longer SOTA; older motion modules; limited to short (~4 sec) clips
- **Feasibility:** HIGH (most tooling, largest community, easiest debugging)

### 4.3 7-Day Trade-Off Analysis

| Approach | Effort | Output Quality | 7-Day Feasibility |
|----------|--------|-----------------|-------------------|
| **AnimateDiff** | LOW | Good (stylized) | ✅ HIGH |
| **LTX-Video** | MEDIUM | Excellent | ⚠️ MEDIUM |
| **CogVideoX** | MEDIUM-HIGH | Very good | ⚠️ MEDIUM |
| **Mochi (API)** | LOW | Excellent | ⚠️ MEDIUM (quota limits) |

### 4.4 Recommendation for Pipeline 3

**Primary choice: AnimateDiff**
- Pair SDXL LoRA poster with motion module
- Generate 4–6 second looping "teaser" with background music
- **Implementation:** ComfyUI workflow or `diffusers.AnimateDiffPipeline`

**Fallback: LTX-Video API**
- Generate 30fps snippet from generated poster image
- Less customization, but cleaner output

**Time budget:** 6–8 hours (setup + generation + composition with audio)

### 4.5 Mathematical Foundation (AnimateDiff)

**Motion module injection** (temporal attention layers):

For each spatial attention block, add a temporal attention mechanism:
$$\text{Attn}_{\text{temporal}}(Q_t, K_t, V_t) = \text{softmax} \left( \frac{Q_t K_t^T}{\sqrt{d}} \right) V_t$$

where $Q_t, K_t, V_t$ are computed across frames (temporal dimension) while keeping spatial queries/keys frozen (from SDXL U-Net).

**Frame consistency loss** (during motion module training, not needed for inference-only):
$$\mathcal{L}_{\text{flow}} = \mathbb{E} \left[ \| \text{OpticalFlow}(f_i, f_{i+1}) - \text{motion\_pred}(f_i) \|_2^2 \right]$$

---

## 5. Centroid-to-Movie Generation: Conditional VAE vs. Classifier-Free Guided Diffusion

### 5.1 Architecture Comparison

#### **Approach A: Conditional VAE (CVAE)**

**What it does:**
- Encodes each movie (poster + text features) to a latent $z$ conditioned on Temilola's rating
- Posterior: $q_\phi(z | x, y)$ where $x = $ poster features, $y = $ rating
- Decoder: $p_\theta(x | z, y)$ reconstructs poster
- At inference: encode centroid embedding → sample $z$ → decode to novel poster

**Architecture:**
```
Encoder(rating=4.5, genres, sentiment_features)
  → Gaussian posterior (μ, σ)
  → sample z ~ N(μ, σ)
  → Decoder(z, rating=4.5)
  → reconstructed/novel poster
```

**Pros:**
- Probabilistic latent space; explicit uncertainty quantification
- Can marginalize out rating to explore space: $\int p(x | z, y) p(y) dy$
- Amenable to manifold traversal, interpolation

**Cons:**
- Posterior collapse (KL → 0, ignoring $z$) is common → requires careful annealing
- Requires paired (movie, rating) data → limited by N=162

#### **Approach B: Classifier-Free Guided Diffusion**

**What it does:**
- Train a single diffusion model to denoise both conditioned ($c = $ rating/embedding) and unconditional ($c = \emptyset$)
- At inference, blend predictions: $\hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + w (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$
- Guidance scale $w$ controls how much the model "listens" to conditioning

**Pros:**
- Simpler training (single model); no posterior collapse
- Flexible guidance strength (w ∈ [0, ∞]) trades off diversity vs. fidelity
- Scales to large unconditional datasets

**Cons:**
- Less interpretable latent space than CVAE
- Requires careful calibration of $w$

### 5.2 Mathematical Foundation

#### **CVAE Training Objective (ELBO):**

$$\mathcal{L}_{\text{CVAE}} = -\mathbb{E}_{q_\phi(z|x,y)} \left[ \log p_\theta(x | z, y) \right] + \beta \cdot \text{KL}(q_\phi(z|x,y) \| p(z|y))$$

where:
- First term: reconstruction (poster fidelity)
- Second term: KL regularization (latent distribution match)
- $\beta$ annealing schedule prevents posterior collapse

#### **Classifier-Free Guidance Blending:**

$$\epsilon_t^{\text{guided}} = (1 - w) \epsilon_t(x_t, \emptyset) + w \epsilon_t(x_t, c)$$

or equivalently:
$$\epsilon_t^{\text{guided}} = \epsilon_t(x_t, \emptyset) + w \left( \epsilon_t(x_t, c) - \epsilon_t(x_t, \emptyset) \right)$$

Guidance scale $w > 1$ amplifies condition signal; $w < 1$ adds stochasticity.

### 5.3 7-Day Implementation Estimate

**CVAE from-scratch (RECOMMENDED for math depth):**

| Task | Time |
|------|------|
| Implement encoder (MLP + 2D conv) | 2 h |
| Implement decoder (deconv + MLP) | 2 h |
| ELBO loss + beta-annealing schedule | 1 h |
| Training loop + checkpointing | 1.5 h |
| Latent traversal visualization | 1 h |
| Baseline (PCA centroid + nearest neighbor) | 1 h |
| **Total** | **8.5 h** (MEDIUM effort) |

**Classifier-Free Guided Diffusion (library-heavy):**

| Task | Time |
|------|------|
| Setup diffusers conditional pipeline | 1 h |
| Condition engineering (rating → embedding) | 1 h |
| Unconditional sampling in training | 1.5 h |
| Guidance scale sweeps | 1 h |
| Comparison vs. CVAE | 1.5 h |
| **Total** | **6 h** (LOW-MEDIUM effort) |

### 5.4 Apples-to-Apples Baseline: PCA + Nearest Neighbor

**Method:**
1. PCA on all 162 movie embeddings (latent sentiment features, genres, ratings)
2. Compute centroid of top k favorite movies (k=5)
3. Find 5 nearest neighbors in original space
4. Return their posters + titles + plots

**Evaluation:**
- **Similarity to centroid:** L2 distance in PCA space
- **Coverage:** Do the 5 nearest neighbors span different genres/styles?
- **Novelty:** Are they genuinely new recommendations, or just repeats?

---

## 6. Conditional Generation Architectures: Deep Dive

### 6.1 Architectural Landscape

| Architecture | Conditioning | Use Case | Math Complexity | 7-Day Feasibility |
|--------------|--------------|----------|-----------------|-------------------|
| **CVAE** | Latent + condition in encoder/decoder | Interpolation, manifold traversal | HIGH (ELBO + KL) | MEDIUM |
| **Conditional GAN** | Concatenate condition to input | Adversarial generation | HIGH (minimax) | MEDIUM-HIGH |
| **Classifier-Free Diffusion** | Dropout condition during training | Flexible guidance strength | MEDIUM (blending) | LOW-MEDIUM |
| **ControlNet** | Spatial constraints (depth, pose, edges) | Layout + style control | MEDIUM (U-Net skip fusion) | LOW (library exists) |
| **IP-Adapter** | Image feature embeddings | Style transfer + multi-modal | MEDIUM (cross-attention) | LOW (library exists) |

### 6.2 Conditional GAN (Quick Reference)

**Minimax objective:**
$$\min_G \max_D \mathbb{E}_{x \sim p_{\text{data}}} \left[ \log D(x, c) \right] + \mathbb{E}_{z \sim p_z} \left[ \log(1 - D(G(z, c), c)) \right]$$

**Why it's excellent for rubric:**
- Adversarial training (novel, deep theory)
- Clear first-principles derivation (game theory, Wasserstein distance, gradient penalties)
- Strong visual results

**Why it's risky for 7-day deadline:**
- Notoriously unstable (mode collapse, gradient issues)
- Requires careful tuning (spectral normalization, gradient penalty, discriminator architecture)
- Debugging is slow

**Verdict:** Skip for Pipeline 3 unless you have confidence in stability. CVAE or classifier-free diffusion are safer bets.

### 6.3 ControlNet Deep Dive

**Purpose:** Add spatial conditioning to diffusion models (depth maps, pose, edges, segmentation)

**Architecture (Zhang et al., ICCV 2023):**

For each U-Net block, create a "control block":
- Input: spatial condition (depth, pose, etc.) + copy of U-Net params (zero-conv initialized)
- Output: scaled addition to U-Net intermediate activations

**Zero-convolution trick:**
$$\text{ZeroConv}(x) = \text{Conv2D}(x, w=0, b=0) \to \text{output} = 0$$

During training, gradients flow through the zero-initialized convolution, so no harmful noise affects the backbone.

**Training:** Robust to small (< 50k) and large (> 1M) datasets; no catastrophic forgetting.

**For posters:** Use Canny edge maps of liked posters to enforce composition consistency.

---

## 7. Wild Card: IP-Adapter + Latent Interpolation for "Taste Manifold Evolution"

### 7.1 The Idea

Instead of generating one "ideal movie poster" per centroid, **interpolate along Temilola's taste manifold** using:

1. **Latent inversion:** Encode each of Temilola's favorite posters into SDXL latent space using VAE encoder
2. **SLERP (Spherical Linear Interpolation):** Smoothly interpolate between favorite posters in latent space
3. **IP-Adapter style transfer:** At each interpolation step, apply IP-Adapter to lock the visual "mood" while allowing content variation
4. **Generation loop:** Generate 20–30 posters along the interpolation path, visualizing the evolution of her taste

### 7.2 Why This Is Wild

- **Unexpected:** Combines encoder inversion + latent interpolation + style transfer in a single pipeline
- **Math-dense:** Involves VAE reconstruction, spherical interpolation (avoiding linear blending artifacts), and IP-Adapter's cross-attention mechanics
- **Rubric goldmine:** Novel method (not in class), deep math (SLERP derivation, IP-Adapter formalism), impressive visualization
- **Narrative power:** "Watch how your taste evolves along this latent curve" — deeply personal and humorous

### 7.3 Mathematical Foundation

**SLERP (Spherical Linear Interpolation) in latent space:**

Given two latent vectors $z_a, z_b \in \mathbb{R}^d$:

$$z_t = \text{SLERP}(z_a, z_b, t) = \frac{\sin((1-t) \theta)}{\sin(\theta)} z_a + \frac{\sin(t \theta)}{\sin(\theta)} z_b$$

where $\theta = \arccos \left( \frac{z_a \cdot z_b}{\| z_a \| \| z_b \|} \right)$ (angle between vectors).

**Why SLERP vs. linear interpolation?**
- Linear: $z_t = (1-t) z_a + t z_b$ → speeds up in middle, slows at edges (non-constant velocity on manifold)
- SLERP: constant velocity along hypersphere; preserves manifold structure

**IP-Adapter style locking (cross-attention injection):**

$$\text{Attn}_{\text{out}} = \text{Attn}(Q_{\text{text}}, K_{\text{text}}, V_{\text{text}}) + w \cdot \text{Attn}(Q_{\text{image}}, K_{\text{image}}, V_{\text{image}})$$

where weight $w$ balances text control vs. image style. By freezing the style image and only varying text prompts + latent $z_t$, we get smooth stylistic evolution.

### 7.4 Implementation Sketch

```python
# 1. Encode favorite posters
encoder = StableDiffusionXLVAEEncoder()
z_favs = [encoder(poster) for poster in temilola_favs]

# 2. Compute pairwise SLERP paths
paths = []
for i in range(len(z_favs) - 1):
    path = [slerp(z_favs[i], z_favs[i+1], t)
            for t in linspace(0, 1, 10)]
    paths.append(path)

# 3. Generate + apply IP-Adapter
pipe = StableDiffusionXLPipeline.from_pretrained("...")
ip_adapter = IPAdapter(pipe, "style_image_path", scale=0.5)

for latent_z in flatten(paths):
    # Decode to image space
    poster = pipe(
        prompt="[centroid-derived prompt]",
        latents=latent_z,
        cross_attention_kwargs=ip_adapter.kwargs
    )
    save(poster, f"evolution_{idx}.png")
```

### 7.5 7-Day Implementation Estimate

| Task | Time |
|------|------|
| Latent inversion (VAE encode) | 0.5 h |
| SLERP implementation | 0.5 h |
| IP-Adapter integration | 1.5 h |
| Generation loop | 1 h |
| Visualization (video, grid, animation) | 1.5 h |
| **Total** | **5 h** (LOW-MEDIUM effort) |

**Effort is LOW because:**
- All libraries exist (diffusers + IP-Adapter)
- SLERP is ~10 lines of NumPy
- No training required (inference-only)

---

## 8. Data Augmentation & Scaling Strategy

### 8.1 Breaking the N=162 Ceiling

**Problem:** Temilola's 162 movies limit fine-tuning capacity; overfitting risk is high.

**Solutions:**

1. **TMDB/IMDB enrichment:**
   - Pull TMDB poster URLs for all 162 movies (already done in Pipeline 1/2)
   - Retrieve 1000+ movies with similar genres + ratings as soft targets
   - Augment training set to N ≈ 500–1000 via genre/rating nearest neighbors

2. **Plot-only fine-tuning:**
   - Use public plot summaries (MovieLens, IMDB, TMDB) for LLM fine-tuning
   - Temilola's 162 rating labels provide "taste filter" without requiring plots for every one

3. **Synthetic augmentation:**
   - CVAE latent interpolation can generate novel examples in the latent space
   - Bootstrap training data: generate synthetic (z, rating) pairs; use as pseudo-labels

### 8.2 Recommended Data Splits

| Set | Size | Source | Purpose |
|-----|------|--------|---------|
| **Personal favorite** | 20–30 | Temilola's 5-star ratings | LoRA fine-tuning (posters) |
| **Personal rated** | 162 | Temilola (existing) | Centroid computation, CVAE training |
| **Augmented (soft targets)** | 500–1000 | TMDB NN + public datasets | LLM fine-tuning, data augmentation |

---

## 9. Apples-to-Apples Comparison Table

Here's a unified table comparing all methods across Pipelines 1, 2, and 3:

| Method | Pipeline | Novelty | Math Depth | VRAM (GB) | 7-Day Effort | Rubric Impact |
|--------|----------|---------|-----------|----------|--------------|---------------|
| **K-means** | 1 | No (class method) | Low | 1 | 0.5 h | Baseline |
| **Ridge regression** | 1 | No | Low | 0.5 | 0.5 h | Baseline |
| **Sentence-transformers** | 2 | No | Medium | 4 | 1 h | Transfer learning |
| **Kernel SVM (from scratch)** | 2 | No | Medium | 2 | 3 h | Advanced SVM |
| **Autoencoder** | 2 | Partial (but class-adjacent) | Medium | 2 | 3 h | Representation learning |
| **SDXL + LoRA (posters)** | 3 | **YES** | High | 12–16 | 9 h | **LO 5 candidate** |
| **Llama 3.2 + QLoRA (plots)** | 3 | **YES** | High | 8–12 | 9 h | **LO 5 candidate** |
| **CVAE (centroid generation)** | 3 | **YES** | **Very High** | 6–8 | 8.5 h | **LO 5 candidate** |
| **Classifier-free diffusion** | 3 | **YES** | High | 10–14 | 6 h | LO 4–5 |
| **MusicGen** | 3 | **YES** | Medium | 6–8 | 6 h | LO 4–5 |
| **AnimateDiff** | 3 | **YES** | Medium | 8–12 | 8 h | LO 4–5 |
| **IP-Adapter + SLERP (wild card)** | 3 | **YES** (unexpected combo) | High | 10–14 | 5 h | **LO 5 candidate** |

---

## 10. The Grand Narrative Arc for Pipeline 3

### **Act 1: Taste Embedding (3–4 days)**
- Combine Pipeline 1/2 embeddings (sentence-transformers + ResNet-50 features)
- Compute Temilola's "taste centroid" via weighted K-means (weights = her ratings)
- PCA/UMAP visualization of the latent space + centroid location

### **Act 2: Conditional Generation (4–5 days)**
- Fine-tune SDXL LoRA on 25 favorite posters → conditional image generation
- Fine-tune Llama 3.2 3B via QLoRA on plots → conditional text generation
- Train CVAE end-to-end on (embedding, rating) pairs → probabilistic centroid sampling

### **Act 3: Synthesis & Evolution (2–3 days)**
- **Phase B:** Generate 5 posters + 1 plot + 1 title from centroid
- **Wild card:** IP-Adapter + SLERP interpolation → 30-frame "taste evolution" video showing smooth transition from favorite A to favorite B
- **(Optional)** MusicGen or AnimateDiff for audio/video

### **Act 4: Evaluation & Comparison (1–2 days)**
- Apples-to-apples: generated posters vs. PCA-NN baseline (human + automatic metrics)
- Generated plots vs. zero-shot LLM (ROUGE, human preference)
- CVAE manifold coverage vs. nearest-neighbor diversity
- Rubric checklist: all required sections, first-principles math, visualization polish

### **Act 5: Narrative Polish (final day)**
- One coherent story: "We built a machine to dream Temilola's ideal movie"
- Formatting: consistent fonts, no bullet-heavy sections, clear diagram captions
- References + reproducibility notes + Git repo link

---

## 11. Libraries & Tools to Grab (April 2026)

### **Core Libraries**

| Library | Purpose | Install |
|---------|---------|---------|
| `diffusers` | Diffusion model inference + LoRA training | `pip install diffusers` |
| `transformers` | LLM loading + generation | `pip install transformers` |
| `peft` | LoRA + QLoRA adapters | `pip install peft` |
| `bitsandbytes` | 4-bit quantization | `pip install bitsandbytes` |
| `audiocraft` | MusicGen | `pip install audiocraft` |
| `sentence-transformers` | Text embeddings (reuse from P2) | `pip install sentence-transformers` |
| `torch` + `torchvision` | Deep learning backbone | Already in Colab |

### **ControlNet + IP-Adapter**

- **ControlNet:** Bundled in `diffusers >= 0.19.0`
- **IP-Adapter:** [GitHub: tencent-ailab/IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) (manual integration or via `diffusers`)

### **Datasets & Checkpoints**

| Asset | Source |
|-------|--------|
| **SDXL 2.0 checkpoint** | [stabilityai/stable-diffusion-xl-base-1.0 on HF](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |
| **Llama 3.2 3B** | [meta-llama/Llama-3.2-3B on HF](https://huggingface.co/meta-llama/Llama-3.2-3B) |
| **MusicGen** | [facebook/musicgen-medium on HF](https://huggingface.co/facebook/musicgen-medium) |
| **Movie plots** | TMDB API (already used in P1), IMDB datasets |
| **Poster images** | TMDB (via existing pipeline) |

### **Optional: Community Tools**

- **ComfyUI:** Node-based diffusion interface (AnimateDiff workflows)
- **kohya_ss:** Standalone LoRA training UI (easier than code for poster fine-tuning)
- **Unsloth:** Accelerated LLM fine-tuning (drop-in for `transformers`)

---

## 12. Practical Colab Checkpoints (18 GB RAM Constraint)

### **Phase A: Setup** (Cells 0–10)
```
Imports + environment setup
├─ Load sentence-transformers model
├─ Load all 162 movie embeddings (cached)
├─ Compute centroid + visualize
└─ Save embeddings to .npz
```

### **Phase B: Poster Generation** (Cells 11–20)
```
SDXL LoRA fine-tuning
├─ Download SDXL base model (6.6 GB) → GPU
├─ Crop/resize 25 favorite posters
├─ LoRA config (rank=4, alpha=16)
├─ Train for 30 min (batch_size=1, gradient_accumulation_steps=4)
├─ Save LoRA weights (~50 MB)
└─ Memory checkpoint (del model, empty_cache)
```

### **Phase C: Text Generation** (Cells 21–30)
```
Llama 3.2 3B QLoRA fine-tuning
├─ Download Llama 3.2 3B (quantized, 8 GB)
├─ Load QLoRA config (rank=8, lora_alpha=32)
├─ Fine-tune on 200 plot examples (1 epoch, 2 hours)
├─ Save LoRA weights (~20 MB)
└─ Memory checkpoint
```

### **Phase D: Centroid Generation** (Cells 31–40)
```
CVAE training
├─ Initialize encoder (MLP) + decoder (deconv)
├─ ELBO loss + beta annealing schedule
├─ Train on 162 examples (5 epochs, 30 min)
├─ Evaluation: latent traversal plots
└─ Generate 5 posters + plots from centroid
```

### **Phase E: Wild Card** (Cells 41–50, if time permits)
```
IP-Adapter + SLERP interpolation
├─ Encode 2 favorite posters to latent space
├─ Compute SLERP path (10 interpolation steps)
├─ Generate poster at each step with IP-Adapter
└─ Create animation (MP4) showing evolution
```

---

## 13. Checklist: Rubric Alignment

### **MLCode (Working library + from-scratch)**
- ✅ Diffusers inference (library)
- ✅ LoRA training loop (from-scratch variant available)
- ✅ CVAE encoder/decoder (from-scratch required)
- ✅ QLoRA fine-tuning (library + custom loop)
- ✅ Latent inversion + SLERP (from-scratch)

### **MLExplanation (Intuitive + analytical)**
- ✅ Score-matching diffusion derivation
- ✅ LoRA low-rank factorization + parameter efficiency
- ✅ CVAE ELBO breakdown
- ✅ Classifier-free guidance blending formula
- ✅ SLERP interpolation on manifold

### **MLFlexibility (Beyond class + transfer)**
- ✅ Transfer learning (frozen SDXL backbone)
- ✅ Parameter-efficient fine-tuning (LoRA, QLoRA)
- ✅ Multi-modal conditioning (text + image)
- ✅ Conditional generation (CVAE, diffusion)

### **MLMath (First-principles derivations)**
- ✅ Diffusion forward/reverse process (probability, noise schedules)
- ✅ ELBO for VAEs (KL divergence + reconstruction)
- ✅ Score matching (gradient of log-density)
- ✅ SLERP geometry (spherical interpolation)
- ✅ Adversarial minimax (optional: GAN formulation)

### **Data Visualization**
- ✅ 2D latent space projection (t-SNE / UMAP)
- ✅ Centroid + favorite cluster highlights
- ✅ Generated poster grids + comparisons
- ✅ Latent traversal heatmap
- ✅ Evolution video (IP-Adapter + SLERP)

### **References + Reproducibility**
- ✅ All paper links (diffusion, LoRA, CVAE, IP-Adapter)
- ✅ Model checkpoint names (HF Hub links)
- ✅ Training hyperparameters
- ✅ Seed for reproducibility
- ✅ Colab notebook link (GitHub)

---

## 14. Risk Mitigation & Fallback Plan

### **Risk: Poster LoRA overfitting (only 25 images)**
- **Mitigation:** Use aggressive augmentation (crop, rotate, color jitter); regularize with batch normalization
- **Fallback:** Use pre-trained style LoRAs from CivitAI + blend them

### **Risk: CVAE posterior collapse**
- **Mitigation:** Start with $\beta = 0.01$, anneal to $\beta = 1.0$ over epochs
- **Fallback:** Use β-TCVAE (total correlation penalty) or switch to a VAE-GAN hybrid

### **Risk: QLoRA training instability (NaN loss)**
- **Mitigation:** Use smaller rank (r=4), lower learning rate (5e-4), gradient clipping
- **Fallback:** Switch to full LoRA (12 GB VRAM feasible) or use pre-trained adapter

### **Risk: AnimateDiff output is jittery/blurry**
- **Mitigation:** Use motion module v3 (latest); add optical flow regularization in inference
- **Fallback:** Generate static 5-frame "key frame" set instead of looping video

### **Risk: SLERP interpolation looks linear/boring**
- **Mitigation:** Add semantic mixing in latent space (e.g., interpolate prompt embeddings separately)
- **Fallback:** Use straight nearest-neighbor jumps + transition effects in post-processing

---

## 15. Key Research & References

### **Diffusion Models**

1. [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239) — Ho, Jain, Abbeel (2020)
2. [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — Ho, Salimans (2022)
3. [Stable Diffusion 1.5 Release](https://huggingface.co/docs/diffusers/using-diffusers/stable_diffusion) — Stability AI
4. [Fine-Tuning Stable Diffusion XL with LoRA](https://www.flex.ai/blueprints/fine-tuning-a-stable-diffusion-xl-with-lora) — FlexAI (2025)
5. [FLUX.1 [dev] ControlNet + LoRA](https://fal.ai/models/fal-ai/flux-general) — FAL (2026)

### **LoRA & Parameter-Efficient Fine-Tuning**

6. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al. (2021)
7. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al. (2023)
8. [Enhancing Diffusion-Based Music Generation Performance with LoRA](https://www.mdpi.com/2076-3417/15/15/8646) — MDPI (2025)

### **ControlNet & IP-Adapter**

9. [Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)](https://arxiv.org/abs/2302.05543) — Zhang et al. (ICCV 2023)
10. [IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models](https://github.com/tencent-ailab/IP-Adapter) — Tencent AI Lab (2023)

### **Conditional VAEs**

11. [Learning Structured Output Representation using Deep Conditional Generative Models (CVAE)](https://arxiv.org/abs/1506.02216) — Sohn, Lee, Yan (2015)
12. [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://arxiv.org/abs/1804.03599) — Higgins et al. (2017)

### **Music & Audio Generation**

13. [AudioLDM: Text-to-Audio Generation with Latent Diffusion Models](https://audioldm.github.io/audioldm2/) — GitHub (2023)
14. [Simple and Controllable Music Generation with Composable Tokens (MusicGen)](https://ai.meta.com/research/publications/simple-and-controllable-music-generation-with-composable-tokens/) — Meta (2023)

### **Video Generation**

15. [Stable Video Diffusion: Image-to-Video Generation with Latent Diffusion](https://stability.ai/stable-video) — Stability AI (2024)
16. [AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2307.04725) — Guo et al. (2023)
17. [LTX-Video: High-Fidelity Text-to-Video Generation with Latent Interpolation](https://github.com/Lightricks/LTX-Video) — Lightricks (2024)
18. [CogVideoX: Text-to-Video Diffusion Models with Expert Transformer](https://github.com/THUDM/CogVideo) — Tsinghua (2024)

### **Latent Inversion & Interpolation**

19. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)](https://arxiv.org/abs/2010.11929) — Dosovitskiy et al. (2020) [for SLERP/latent geometry]
20. [Smooth Diffusion: Crafting Smooth Latent Spaces in Diffusion Models](https://openaccess.thecvf.com/content/CVPR2024/papers/Guo_Smooth_Diffusion_Crafting_Smooth_Latent_Spaces_in_Diffusion_Models_CVPR2024_paper.pdf) — Guo et al. (CVPR 2024)

### **Hugging Face & Implementation Guides**

21. [🤗 Diffusers Documentation](https://huggingface.co/docs/diffusers/index) — Hugging Face (2026)
22. [Diffusion Models Course](https://huggingface.co/learn/diffusion-course/en/unit1/2) — HF Learning Hub (2026)
23. [Transformers: State-of-the-art Natural Language Processing](https://huggingface.co/docs/transformers/index) — HF (2026)

---

## 16. Conclusion: Why This Pipeline Will Shine

**The narrative is coherent and personal:**
- Start with Temilola's taste (centroid in embedding space)
- Train generative models to "dream" movies matching her preferences
- Generate novel posters, plots, and soundtracks
- Interpolate smoothly along her taste manifold using IP-Adapter
- Reflect on what this says about her preferences + implicit biases

**Math is deep and first-principles:**
- CVAE ELBO derivation (KL + reconstruction trade-off)
- Diffusion score-matching (gradient of log-density)
- LoRA factorization (low-rank parameterization efficiency)
- SLERP interpolation (manifold geometry)
- Classifier-free guidance (blended denoising)

**Implementation is realistic for 7 days:**
- ~40 hours of work (intensive, but doable)
- Relies on proven libraries (diffusers, transformers, peft)
- Fallback plans for each risky component
- Clear phase breakdown with checkpoints

**Rubric alignment is airtight:**
- ✅ Novel method (not in class; goes beyond P1/P2)
- ✅ From-scratch math + library implementations coexist
- ✅ Apples-to-apples baselines (PCA-NN, zero-shot LLM, etc.)
- ✅ Visuals are compelling (latent walks, interpolation videos, grid comparisons)
- ✅ Professional presentation (one coherent narrative, no bullet-heavy sections)

---

**End of Research Report**

Generated: April 16, 2026
Prepared for: Temilola Olowolayemo, CS156 Pipeline 3, Prof. Watson
Total Research Time: 3 hours
Sources: 23 papers + library docs (see Section 15)
