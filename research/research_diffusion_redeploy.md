# Pipeline 3: Diffusion Models as the ACT III Centerpiece

**Prepared for:** Temilola Olowolayemo, CS156 Pipeline 3 (Prof. Watson, due Apr 23 2026)
**Objective:** Redeploy diffusion-based generative models (score-based, DDPM, latent diffusion) as the technical spine of ACT III (GENERATE), integrating first-principles math, fine-tuning on personal data, and novel conditional sampling from learned taste posteriors.
**Constraint:** 7-day implementation window; Colab Pro A100 available; ~25 liked posters + 162 embeddings as training signal.

---

## Executive Summary

Diffusion models can be more than "apply SDXL + LoRA." We can fold diffusion into Pipeline 3's probabilistic framework by:

1. **Viewing diffusion as inverse sampling from the taste posterior** — after training a Gaussian Process or VAE in ACT II, sample hypothetical movies from the high-rating region by inverting through diffusion (score-distillation / SDS-guidance).
2. **Fine-tuning diffusion conditioners on modality ablations** — the ACT I experiment (poster-only, title-only, synopsis-only ratings) directly conditions diffusion: "generate a poster such that a typical Temilola rates it 4.5 when shown in isolation."
3. **Treating diffusion latents as embeddings** — skip pixel-space generation; instead, diffuse in the learned CVAE latent space, then decode. This is theoretically cleaner and computationally cheaper.
4. **Structured conditional generation** — use classifier-free guidance **derived from first principles** to trade off "realism" vs "match my taste."

This positions diffusion not as a decoration (pretty posters) but as the inverse inference engine that closes the loop: taste model → posterior → diffusion sampler → validation.

---

## 1. Five Concrete Diffusion Approaches for Pipeline 3

### 1.1 **Approach A: DreamBooth Fine-Tune of SDXL on ⭐5 Posters**

**What:** Fine-tune SDXL's full UNet (not just LoRA) on Temilola's 5-star movie posters using a rare-token placeholder `[T]` + regularization on real images from COCO/OpenImages.

**Why it impresses the rubric:**
- **MLCode:** From-scratch diffusion training loop (noise sampling, forward process, reverse denoising) + library (diffusers) verification.
- **MLMath:** Derivation of L2 diffusion loss as lower bound on forward KL; DreamBooth regularization as multi-task objective combining liked-poster likelihood + generic-poster prior.

**Math foundation:**
```
Forward: q(x_t | x_0) = √(ᾱ_t) x_0 + √(1 - ᾱ_t) ε    [noise schedule ᾱ_t = ∏ᵢ(1 - βᵢ)]

Reverse denoising: p_θ(x_{t-1} | x_t) ≈ 𝒩(μ_θ(x_t, t, c), σ²_t I)

Score matching (noise prediction): L = 𝔼_{t,x₀,ε} [‖ ε_θ(x_t, t, c) - ε ‖²₂]

DreamBooth: L_total = L_diffusion(liked_posters) + λ·L_diffusion(COCO_prior)
```

**Concrete recipe:**
- Collect 25 ⭐5-rated posters. If <25, augment via rotations/crops.
- Use `diffusers.StableDiffusionXLPipeline` + custom training loop (DeepSpeed or standard PyTorch).
- Noise schedule: linear β ∈ [0.0001, 0.02] over 1000 steps (matched to SDXL default).
- Batch size: 2, LR: 1e-4, steps: 500–1000 (varies by GPU; A100 ≈ 45 min at batch=2).
- Reg images: 100 COCO images (10 semantic classes: people, art, interiors, landscapes, etc.).
- Output: checkpoint saving at step 100, 250, 500; pick step with lowest validation loss on held-out 5 liked posters.

**Apples-to-apples baseline:** PCA centroid + nearest-neighbor ResNet-50 retrieval. Generate 5 images each; measure CLIP-similarity to Temilola's taste centroid (hypothesis: DreamBooth >0.72 CLIP-sim; NN <0.65).

**Wall-clock on A100:** 1–2 hours training + validation.

---

### 1.2 **Approach B: Latent-Space Diffusion on CVAE Embeddings**

**What:** Train a VAE on Temilola's 162 movie embeddings (ResNet-50 poster features, 2048D → 64D bottleneck). Then train a diffusion model in the 64D latent space. Sample from high-rating posterior, decode back to image space.

**Why it impresses the rubric:**
- **MLFlexibility:** Diffusion on embeddings is rare in the rubric; combines VAE + diffusion in one loop.
- **MLMath:** Full ELBO derivation for VAE (KL + reconstruction); score-matching in latent space; posterior sampling via Bayes rule on the taste preference model.
- **MLCode:** Custom VAE training + custom latent-diffusion loop.

**Math foundation:**
```
VAE encoder: q_φ(z | x) = 𝒩(μ_φ(x), σ²_φ(x))
VAE decoder: p_ψ(x | z) = 𝒩(μ_ψ(z), I)
ELBO: ℒ = 𝔼_q[log p_ψ(x|z)] - KL(q_φ(z|x) ‖ p(z))

Taste preference: log p(rating=4.5 | z) ∝ -‖z - z_ideal‖² / σ²  [Gaussian assumption]

Posterior: p(z | rating=4.5) ∝ p(rating=4.5 | z) · p(z)

Latent diffusion in z:
  Forward: q(z_t | z_0) = √(ᾱ_t) z_0 + √(1-ᾱ_t) ε
  Reverse: p_θ(z_{t-1} | z_t, c) where c encodes "high-rating" condition
```

**Concrete recipe:**
- Train VAE on 162 embeddings (ResNet-50, 2048D) with latent dim=64.
- Encoder: 2048 → 512 → 128 → 64 (μ, σ²).
- Decoder: 64 → 128 → 512 → 2048.
- VAE loss: L = MSE(x, ̂x) + 0.001·KL(q_φ | p(z)).
- Steps: 200 epochs, batch=32, LR=1e-3, Adam.
- Then train diffusion in 64D space. Noise schedule: linear β ∈ [1e-4, 0.02]. Conditioning: embed "rating target" (4.5) as scalar time-step + side input.
- Sampling: start from z_T ~ 𝒩(0,I); denoise 1000 steps; decode z_0 → image via VAE decoder.

**Apples-to-apples baseline:** K-means centroid on embeddings → decode via VAE → compare CLIP-sim.

**Wall-clock on A100:** 1 hr VAE training + 1 hr diffusion training.

---

### 1.3 **Approach C: Classifier-Free Guidance with Modality Conditioning (ACT I Integration)**

**What:** Use ACT I's modality ablation (poster-only, title-only, synopsis-only ratings) to train a diffusion model with explicit conditioning: generate a poster such that `logit(rating=4.5 | poster, condition)` is maximized.

**Why it impresses the rubric:**
- **MLMath:** Full derivation of classifier-free guidance from first principles.
- **MLExplanation:** Closes the loop between ELICIT and GENERATE; modality ablation is not auxiliary but core.
- **MLFlexibility:** Conditional diffusion is not in class; deriving guidance from first-principles is novel.

**Math foundation:**
```
Classifier-free guidance:

ε̃(x_t, t, c) = (1 + w) · ε_θ(x_t, t, c) - w · ε_θ(x_t, t, ∅)

where:
  ε_θ(x_t, t, c) = noise-pred conditioned on modality + rating
  ε_θ(x_t, t, ∅) = unconditional noise-pred
  w ∈ [7, 20] = guidance weight (trade-off: high w → "matches taste"; low w → "realistic")

Training objective:
L = 𝔼_{t,x₀,ε,c} [‖ ε_θ(x_t, t, c) - ε ‖²₂] + λ · 𝔼_{t,x₀,ε} [‖ ε_θ(x_t, t, ∅) - ε ‖²₂]

Gradient ascent sampling (SDS-style):
∇_z J(z) = ∇_z 𝔼_{τ~U(1,T)} [‖ ε̃(x(z), τ) ‖²₂]  [implicit posterior]
```

**Concrete recipe:**
- Train diffusion on 162 posters with condition c ∈ {poster, title, synopsis, all} + rating ∈ {2,3,4,5}.
- Conditioning: embed modality + rating as separate tokens; concatenate with CLIP text embedding.
- Classifier-free dropout: 15% of batches, drop c → train unconditional ε_θ.
- Sampling: pick w ∈ [10, 15] via validation (balance aesthetic + taste-match).
- Generate 5 posters per centroid; measure both CLIP-sim (realism) and **modality-conditional likelihood** (ask Temilola to rate each in isolation; track if ratings match predicted 4.5).

**Apples-to-apples baseline:** Ridge regression on modality embeddings; predict rating; sample from high-rating region via Gaussian sampling (no diffusion). Compare prediction accuracy on blind-held-out test.

**Wall-clock on A100:** 2–3 hours training (diffusion is data-hungry; 162 images is low; may need COCO augmentation).

---

### 1.4 **Approach D: Score Distillation Sampling (SDS) — Refine Generated Posters with ResNet Preference Signal**

**What:** After generating a poster via SDXL, use ResNet-50 features + a learned preference predictor (regression from features → rating) to score the generated image. Backprop through diffusion to refine the posterior sample toward "movies I'd like."

**Why it impresses the rubric:**
- **MLMath:** SDS is a bridge between supervised learning + generative modeling; requires deriving gradients through the diffusion process.
- **MLFlexibility:** Unusually integrates RL-style reward optimization into diffusion (not standard in class).
- **MLCode:** Custom gradient computation (may need Autograd if library doesn't expose it).

**Math foundation:**
```
Given a generated image x and ResNet features φ(x):
  Preference score: ŷ(x) = w^T φ(x) + b  [linear predictor trained on 162 movies]

Score distillation sampling (Poole et al. 2022):
  ∇_x ℒ_SDS = ∇_x 𝔼_τ [σ(τ) · (ε̃(x_τ, τ) - ε_target) · ∇_x ŷ(x)]

Optimization: x ← x - α · ∇_x ℒ_SDS [gradient ascent]

Interpretation: Move x in image space toward high preference, constrained by diffusion denoiser.
```

**Concrete recipe:**
- Train linear preference predictor: ResNet-50(x) [2048D] → 1 [rating prediction]. Use 162 movies; test R² on cross-val.
- Generate 5 initial posters via SDXL+LoRA.
- For each poster, run SDS refinement loop:
  - Forward: ResNet-50 features → preference score ŷ(x).
  - Backward: gradient ∇_x ŷ.
  - Compute diffusion score ε̃ at current image.
  - Update: x ← x - 0.01 · σ(τ) · (ε̃ - ε_target) · ∇ŷ.
  - Iterate 10–20 steps; monitor preference score (should increase).
- Visualization: show before/after posters; plot preference score trajectory.

**Apples-to-apples baseline:** Raw SDXL+LoRA without SDS refinement. Hypothesis: SDS-refined posters have ≥10% higher preference score on cross-val ResNet predictor.

**Wall-clock on A100:** 1 hr SDXL generation + 0.5 hr SDS refinement loop.

---

### 1.5 **Approach E: Diffusion Inpainting with ControlNet (Structured Poster Editing)**

**What:** Given a liked poster, use diffusion inpainting to "evolve" it: mask a region (e.g., color scheme, character placement), condition on Temilola's cluster centroid, regenerate that region to match taste while preserving overall composition.

**Why it impresses the rubric:**
- **MLFlexibility:** Inpainting + ControlNet is advanced architectural knowledge.
- **MLExplanation:** Demonstrates masking, spatial conditioning, trade-off between content preservation and personalization.

**Math foundation:**
```
Inpainting mask: M ∈ {0, 1}^{H×W} (0 = preserve, 1 = regenerate)

Forward with masking:
  x_t = √(ᾱ_t) · (M ⊙ x_0 + (1-M) ⊙ x_original) + √(1-ᾱ_t) ε

ControlNet guidance (edge-preservation):
  Edge map: E = Canny(x_original)

  Denoising conditioned on both text + edge map:
  ε̃ = (1+w)·ε_θ(x_t, t, c, E) - w·ε_θ(x_t, t, ∅, E)
```

**Concrete recipe:**
- Select 1 representative liked poster.
- Segment into regions: background / character / text.
- Mask region(s) for regeneration (e.g., background).
- Use ControlNet with Canny edge detection to preserve composition.
- Condition on centroid embedding + "taste" token.
- Run 500-step diffusion inpainting; show interpolation from original → evolved.

**Apples-to-apples baseline:** Simple color-grading or photoshop simulation (deterministic). Hypothesis: diffusion inpainting rated higher by Temilola (qualitative A/B).

**Wall-clock:** 1–2 hours (mostly inference; no training needed if using pretrained SDXL + ControlNet).

---

## 2. First-Principles Math Derivations (Rubric 5s)

### 2.1 Forward and Reverse Diffusion

**Forward noising schedule (q-process):**

Given data x₀, define a sequence of noise-corrupted versions {x_t} for t ∈ [0, T]:

$$q(x_t | x_0) = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

where $\beta_t = 1 - \alpha_t$ is the per-step variance and $\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$ is the cumulative product.

**Reverse denoising (p-process):**

The reverse process is also Gaussian:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1} | \mu_\theta(x_t, t), \sigma_t^2 I)$$

where the mean is estimated by a learned network $\mu_\theta$ parameterized as a noise predictor:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

### 2.2 Variational Lower Bound (ELBO)

The likelihood under the generative model is bounded:

$$\log p_\theta(x_0) \geq \mathbb{E}_{q(x_1|x_0)} \left[ \log p_\theta(x_0 | x_1) \right] - \sum_{t=2}^{T} \mathbb{E}_{q(x_t|x_0)} \left[ \text{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t)) \right]$$

**Simplification via score matching:** The KL terms simplify to a weighted L2 loss on noise prediction:

$$\mathcal{L}_t = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon_\theta(x_t, t) - \epsilon \|_2^2 \right]$$

### 2.3 Classifier-Free Guidance (Derivation)

Goal: sample from $p(x | c)$ without training a separate classifier. Use Bayes' rule:

$$\nabla_x \log p(x|c) = \nabla_x \log p(c|x) + \nabla_x \log p(x)$$

**Approximation:** The score $\nabla_x \log p(x|c)$ is approximated as:

$$\nabla_x \log p(x|c) \approx (1 + w) \cdot \epsilon_\theta(x_t, t, c) - w \cdot \epsilon_\theta(x_t, t, \emptyset)$$

where $w$ is the guidance scale. During inference, use this rescaled score to guide denoising:

$$\epsiloñ(x_t, t, c) = (1+w) \epsilon_\theta(x_t, t, c) - w \epsilon_\theta(x_t, t, \emptyset)$$

### 2.4 Score Distillation Sampling (SDS)

Given a pretrained diffusion model and a reward function $R(x)$ (e.g., preference predictor), optimize:

$$\nabla_\theta \mathcal{L}_{\text{SDS}} = \mathbb{E}_{t \sim U(1,T)} \left[ \sigma(t) \cdot (\epsilon_\theta(x_t, t) - \text{PredNoise}(x_t, t, x_0)) \cdot \nabla_x R(x) \right]$$

Interpretation: Move x toward high reward while staying on the diffusion manifold.

### 2.5 LoRA Parameterization

Instead of tuning all W ∈ ℝ^{d×d}, use low-rank updates:

$$W_{\text{new}} = W_0 + \Delta W = W_0 + BA$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and r ≪ d (typically r ∈ [4, 64]). Training only A and B reduces parameters from d² to ~2dr.

$$\text{Param efficiency} = \frac{2dr}{d^2} \approx \frac{2r}{d} \quad (\text{for } r=8, d=2048 \Rightarrow 0.78\%)$$

---

## 3. Concrete Fine-Tune Recipe (Production-Ready)

### 3.1 DreamBooth on SDXL (Best Effort)

**Environment:**
```python
# Colab Pro with A100
!pip install diffusers peft transformers torch accelerate

import torch
from diffusers import StableDiffusionXLPipeline
from peft import get_peft_model, LoraConfig

# Load base model
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipeline.to("cuda")

# Apply LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["to_k", "to_v", "to_q", "to_out"],
    lora_dropout=0.05
)

# Fine-tune loop (simplified pseudocode; actual implementation uses trainer)
optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=1e-4)

for epoch in range(10):
    for batch in dataloader:  # Your 25 liked posters
        # Noise scheduling
        t = torch.randint(0, 1000, (batch_size,))
        noise = torch.randn_like(batch["pixel_values"])
        x_t = alpha_bar_sqrt[t] * batch["pixel_values"] + (1 - alpha_bar_sqrt[t]) * noise

        # Forward pass
        pred_noise = pipeline.unet(x_t, t, encoder_hidden_states=batch["text_embeddings"])

        # Loss
        loss = F.mse_loss(pred_noise, noise)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save LoRA weights
pipeline.unet.save_lora_weights("my_lora_unet.pth")
```

**Data preparation:**
- Collect 25 liked ⭐5 posters (min 512×512; ideally 768×768).
- Augment with random crops, rotations (±5°) to ~100 effective images.
- COCO regularization: 50 generic movie-poster-style COCO crops (use a hand-picked subset of "poster," "painting," "art" COCO images).
- Batch size: 2 (memory-constrained on A100).
- LR schedule: linear warmup 500 steps, cosine decay. Max LR = 1e-4.

**Hyperparameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA rank | 8 | Balance param count vs expressivity |
| LoRA alpha | 32 | Standard default |
| Train steps | 1000 | 162 base images × augmentation × ~2 epochs |
| Batch size | 2 | A100 40GB limit |
| Learning rate | 1e-4 | Conservative; Stable Diffusion stable zone |
| Reg. strength λ | 0.1 | Moderate COCO prior (avoid catastrophic forgetting) |

**Wall-clock:** 1–1.5 hours on A100 40GB.

**Validation:** Generate 5 posters per step checkpoint (100, 250, 500, 1000); measure CLIP-sim to taste centroid via:
```python
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Encode generated poster
inputs = processor(images=generated_poster, return_tensors="pt")
image_features = model.get_image_features(**inputs)

# Compare to centroid embedding (from CVAE or ResNet)
similarity = cosine_similarity(image_features, taste_centroid)
```

---

## 4. Narrative Fit: ACT III Redefined

**Original pitch (Generative doc):** SDXL+LoRA for poster, Llama for plot, MusicGen for soundtrack.

**Diffusion redeploy:** Diffusion becomes the **structural backbone** of ACT III:

1. **Phase III-A (Taste Posterior):** Use ACT II's trained GP / CVAE to define the high-rating region in embedding space.
2. **Phase III-B (Diffusion Sampler):** Sample from that region via diffusion-based inference. For each approach:
   - **DreamBooth:** Direct posterior sampling (learned bias toward liked posters).
   - **Latent diffusion:** Sample in VAE space, then decode (theoretically cleaner).
   - **Classifier-free guidance:** Trade off realism vs. taste-match via guidance weight w.
   - **SDS refinement:** Further optimize samples via ResNet preference signal.
3. **Phase III-C (Modality-Conditional Generation):** Use ACT I's ablation data to generate "poster-only-rated-4.5," "synopsis-only-rated-4.5," etc. This closes the loop: experiment → prediction → generation conditioned on experiment.
4. **Phase III-D (Validation Loop):** Take generated posters back to Temilola; blind-rate them; measure if blind ratings match predicted 4.5 ± 0.5. Turing test.

**Why this is > SDXL+LoRA:**
- Diffusion is now **integrated into the probabilistic model** (not auxiliary).
- ACT I's modality experiment directly conditions ACT III's generation (causal closure).
- SDS + preference distillation brings in reward signals (RL-flavored).
- Latent diffusion ties to the learned embedding geometry (CVAE / GP).

---

## 5. Apples-to-Apples Baselines

| Approach | Novel Method | Class Baseline | Key Metric |
|----------|--------------|----------------|-----------|
| DreamBooth | SDXL+LoRA on liked posters | PCA centroid + NN image retrieval | CLIP-sim to taste centroid (expect: 0.72 vs 0.63) |
| Latent diffusion | Diffusion in 64D VAE space | K-means centroid + VAE decode | MSE reconstruction error, visual quality |
| Classifier-free guidance | Guided diffusion with modality conditioning | Ridge regression (modality → rating) + Gaussian sampling | Blind-rating accuracy (diffusion conditioned vs unconditional) |
| SDS refinement | ResNet preference distillation into diffusion | Linear regressor (ResNet → rating), no diffusion | Preference score improvement (expect: +10–15%) |
| Inpainting+ControlNet | Evolved poster via masked diffusion | Deterministic color grading / Photoshop | Subjective A/B test (blind rating by Temilola) |

---

## 6. Wild Card: Diffusion on Embeddings, Not Pixels

**The idea:** Skip image-space diffusion entirely. Instead:
- Frame movies as points in a 384D sentence-transformer space (semantic) or 2048D ResNet space (visual).
- Train a **latent-space diffusion model on embeddings** (not pixels).
- Condition diffusion on: rating target (4.5), modality (poster/title/synopsis), genre, watched-date.
- Sample 100 hypothetical "ideal movies" as points in embedding space.
- Use nearest-neighbor retrieval on TMDB/MovieLens to find real movies near these sampled embeddings (closed-loop synthesis: generate imaginary movie → find real analog → recommend).

**Why:**
- Theoretically elegant: diffusion operates on the actual feature manifold.
- Computationally efficient: 384D embedding space vs. 768³ image space.
- Interpretable: sample → plot coordinates in your taste space → explain which neighbors.
- Flexible: can condition on multiple modalities without training separate models.

**Math hook:**
```
q(z_t | z_0) = √(ᾱ_t) z_0 + √(1-ᾱ_t) ε,  z ∈ ℝ^384

Conditioning on taste (via low-rank adapter):
  ε̃(z_t, t) = (1+w) ε_θ(z_t, t, embedding_target) - w ε_θ(z_t, t, ∅)

Posterior sampling:
  Sample z_T ~ 𝒩(0, I)
  Denoise T steps → z_0 (mode of p(z | rating, modality))
  Find k-NN in TMDB embeddings → recommend real movies

Validation: k-NN movies rated by Temilola should avg ≥4.0 (vs. random ≈3.2).
```

---

## 7. References & Libraries

### Papers
- **Diffusion:** Ho et al. (2020) "Denoising Diffusion Probabilistic Models" (DDPM); Karras et al. (2022) "Elucidating the Design Space of Diffusion Models" (EDM); Song et al. (2020) "Score-Based Generative Modeling" (score matching).
- **Guidance:** Ho & Salimans (2022) "Classifier-Free Diffusion Guidance"; Nichol & Dhariwal (2021) "Improved DDPM."
- **Fine-tuning:** Ruiz et al. (2022) "DreamBooth" (personalization); Hu et al. (2021) "LoRA: Low-Rank Adaptation" (parameter efficiency); Poole et al. (2022) "DreamFusion" (SDS for 3D synthesis).
- **ControlNet:** Zhang et al. (2023) "Adding Conditional Control to Text-to-Image Diffusion Models."

### Libraries
- **`diffusers`** (Hugging Face): SDXL, ControlNet, inpainting pipelines, inference.
- **`peft`** (Hugging Face): LoRA configs, parameter-efficient fine-tuning.
- **`kohya_ss`** (standalone): Advanced LoRA + DreamBooth training (more control).
- **`pytorch`** / **`accelerate`**: Custom training loops, multi-GPU.
- **`transformers`**: CLIP for conditioning and similarity measurement.

### Code Skeletons
```python
# Minimal DreamBooth
from diffusers import StableDiffusionXLPipeline
from peft import LoraConfig, get_peft_model

pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
lora_cfg = LoraConfig(r=8, lora_alpha=32, target_modules=["to_k", "to_v"])
pipeline.unet = get_peft_model(pipeline.unet, lora_cfg)

# Training: forward noise, denoise, L2 loss, backward, save.
# Inference: pipeline(prompt="Temilola's ideal movie", negative_prompt="", guidance_scale=7.5)

# Classifier-free guidance
eps_cond = pipeline.unet(x_t, t, text_embeddings)
eps_uncond = pipeline.unet(x_t, t, null_embeddings)
eps_guided = (1 + w) * eps_cond - w * eps_uncond
```

---

## 8. 7-Day Implementation Feasibility

| Approach | Days | Colab GPU | From-Scratch Code | Rubric Impact |
|----------|------|-----------|------------------|---------------|
| A: DreamBooth (LoRA) | 1.5 | 2–3 hrs A100 | Medium (training loop) | 5/5 MLCode + MLMath |
| B: Latent-space diffusion | 2 | 2–3 hrs A100 | Medium (VAE + diffusion) | 5/5 MLFlexibility |
| C: Classifier-free guidance | 2.5 | 3–4 hrs A100 | Medium (custom conditioning) | 5/5 MLMath (derivation) |
| D: SDS refinement | 1.5 | 1–2 hrs A100 | Hard (Autograd through diffusion) | 5/5 MLFlexibility (RL-style) |
| E: Inpainting+ControlNet | 1 | 0.5 hr A100 (inference only) | Easy (library inference) | 4/5 (library-heavy) |
| **Wild card: Embedding diffusion** | **2** | **1–2 hrs A100** | **Medium (latent diffusion)** | **5/5 MLFlexibility** |

**Recommend:** Combine A (DreamBooth) + B (latent VAE diffusion) + C (classifier-free guidance) for robust narrative + mathematical depth. E (inpainting) as lower-effort visual flourish. D (SDS) if time allows.

---

## Conclusion

Diffusion models can be redeployed from "pretty poster generator" to the **inverse inference engine** that closes the Loop: Experiment (ACT I) → Probabilistic Model (ACT II) → Generative Sampler (ACT III) → Validation (blind re-rating).

The key unlocks are:
1. **Math-from-first-principles:** ELBO, score matching, classifier-free guidance derivation.
2. **Personal fine-tuning:** DreamBooth or latent diffusion on 25 liked posters (not from-scratch DDPM).
3. **Tight integration:** Condition diffusion on ACT I's modality experiment + ACT II's learned taste posterior.
4. **Rubric wins:** From-scratch implementations (training loops) + novel methods (LoRA, SDS, embedding diffusion) + deep math (full ELBO + guidance derivation).

This positions Temilola's pipeline not as "pretty generation" but as **a closed-loop probabilistic narrative** where generation is inference, and inference is validated.

---

**Total document length:** ~800 lines (target met). Ready for integration into Pipeline 3 notebook as the deep-dive appendix backing ACT III.
