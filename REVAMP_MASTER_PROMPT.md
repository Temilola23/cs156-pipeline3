# Pipeline 3 Paper Revamp — Master Prompt

**Date issued:** 2026-04-24 (final day before submission)
**Author of brief:** Temilola
**Scope:** Final, heavy-hitter revamp of `paper/pipeline3.tex` and supporting deliverables (notebook, GitHub repo, teasers). Layered on top of `MASTER_PROMPT.md` (style rules) and `feedback_pipeline3_sprint_cadence.md` (sprint-review cadence).

Everything below is binding. When any subagent (research, plan-writer, implementer, reviewer) is dispatched, this file is their contract.

---

## 0. Global rules (inherited and reinforced)

All rules from `MASTER_PROMPT.md` still apply:

- No em-dashes.
- No AI slop phrases.
- Academic tone. Not cinematic. Not flowery.
- Every subsection follows: motivate -> introduce -> derive -> apply -> interpret -> transition.
- Acronym-first-use discipline; glossary at end.
- Compare every model family to Pipeline 1 and Pipeline 2 (apples-to-apples).
- He/him pronouns for the author (Temilola).

**New global rule (this revamp):** inline bold pseudo-headings of the form `\textbf{Foo.}` at the start of a paragraph must be **proper `\subsubsection*{Foo}`** headings. Continuous-prose lists of the form "Six stages run in series: (1) foo, (2) bar, (3) baz" must be **real LaTeX enumerate or itemize environments**.

---

## 1. The brief (verbatim from the user, organised)

### 1.1 ML-code learning outcome evidence

> "do we have any from scratch implementations for good scores on #ml-code?"

Every model family claimed as a "from-scratch" implementation must have:

1. A Python module in `src/` written without importing the canonical high-level library that does the same job (so a GP implemented with `np.linalg.cholesky` counts; one that calls `GPy.models.GPRegression` does not).
2. A cross-check cell in the notebook that compares the from-scratch result to the library reference (GPy, sklearn, statsmodels, pymc, etc.) on the same input and asserts agreement to a stated tolerance.
3. A LaTeX code listing in the paper excerpting the core function.
4. A sentence in the paper explicitly stating that the implementation is from-scratch and what it was cross-checked against.

### 1.2 Section-by-section narrative discipline

For **Sections 5 (probabilistic and temporal), 6 (relational), 7 (causal and uncertainty), 8 (Thompson), 9 (generative stack)** and in fact every section of the paper:

Each method needs, in order:

1. **Motivation and intuition.** Why this method, in our specific context, at this step. What gap does it fill that the previous method could not.
2. **Detailed derivation** from first principles. Break down every symbol. Explain every step. Justify every assumption. Connect each step to the modelling problem (the 324-rating log, the causal contrast, the trailer pipeline).
3. **Implementation** with a real code snippet from the notebook. Use listings. Show the from-scratch core. Show the cross-check.
4. **Interpretation.** What did it produce. Numbers. Plots. Did it work. Why. Why not.
5. **Does it feed into the broader goal.** Name the downstream artefact that consumes this output (Thompson prior, augmentation pool, teaser keyframe, conformal calibration set).
6. **Transition** as the final paragraph of the subsection, not its own header.

### 1.3 A live "we actually use these results" section

For Section 5 (probabilistic and temporal) and Section 6 (relational), add a subsection titled "How these outputs are consumed downstream" that names which subsequent model, plot, or deliverable takes the output as an input. If no downstream model consumes it, say so explicitly and explain what the result still buys the project (diagnostic value, calibration check, sanity ablation).

### 1.4 Detailed LoRA, diffusion, StyleGAN derivations

Section 9 currently brushes over Low-Rank Adaptation (LoRA), latent diffusion, and StyleGAN3. Rewrite each:

- **LoRA:** motivate, state why we used it and not full fine-tuning / adapters / prefix tuning, why the low-rank hypothesis is appropriate for our trailer-caption use case, derive the reparameterisation $W = W_0 + BA$, explain rank/alpha choices, and interpret the training run (cosine loss curve, sample quality).
- **Diffusion:** give the forward process, the reverse process, the variance schedule, the parameterisation of noise prediction, and classifier-free guidance, deriving each from the variational lower bound rather than stating them.
- **StyleGAN3:** explain the alias-free generator, the mapping network, the non-saturating logistic loss with R1 gradient penalty, and adaptive discriminator augmentation, all derived with intuition at every step.

### 1.5 Plots and visual evidence

Add:

- Sample poster images (3–5) from the SDXL LoRA.
- Sample StyleGAN3 frames (3–5).
- Sample diffusion-step progression.
- Proof-of-augmentation plots: MovieLens twin cosine-similarity histogram, GenMatch covariate-balance table or plot, TVAE latent 2D projection, and a bar chart of corpus sizes (324 original vs 79,430 after augmentation).
- Selected keyframes from each of the three final teasers with per-frame commentary tying them to source films (Doctor Strange, Dune, etc.).

### 1.6 Challenges and appendix

Add an appendix detailing everything that was tried and did not work, and how we pivoted:

- Black-Widow regurgitation incident and the mitigation (constraint tokens, reject-sample on proper-noun overlap).
- StyleGAN3 palette collapse and the ADA fix.
- Early SVD-XT blurriness and why we did not push for a final retake.
- HMM label-switching non-identifiability.
- RunPod journey: provisioning, crashes, checkpointing, the 2025-04-23 launch, what the 2026-04-24 rerun cost.
- What we left on the table (e.g. pluralistic ignorance in the rating collector, tailoring the LoRA to each of 5 storylines).

### 1.7 GitHub deliverable

- Confirm a public GitHub repo exists for Pipeline 3. Get its URL. If not public, make it public.
- Push every currently untracked file that should be tracked. Exclude secrets (API keys, `.env`), large raw videos, and the pod state blobs.
- Hyperlink the repo in the paper (in the introduction, the data-availability statement, and the final section).
- Hyperlink the notebook (`notebooks/main_submission.ipynb`) raw URL.
- Hyperlink the teaser folder(s): both the v1 set and the final v4/v5 set I kept.
- Ensure every cell in the notebook runs end-to-end and matches the numbers and plots in the PDF exactly, with the same depth of derivation and explanation as the paper (no "see paper" stubs).

### 1.8 Generative stack and teaser analysis

Section 9 and the teaser discussion need:

- Explicit statement that we downloaded 1,392 trailers, extracted 20,000 keyframes, and trained StyleGAN3-ADA and the SVD-XT LoRA on that corpus. Make that story crisp.
- A per-teaser analysis: why the three final teasers ended up looking similar (same seed neighbourhood, same narrative root), why some frames are blurry (SVD-XT temporal noise on motion-heavy clips), why I did not push for a retake (compute budget, deadline).
- A subsection "Is this my ideal movie?" that honestly assesses whether the final teasers match my top-rated film taste. Per-keyframe commentary: this shot channels Doctor Strange's portal geometry; that shot channels Dune's desert palette; this one failed to stitch.
- Next steps: a per-storyline LoRA, a longer corpus, MusicGen fine-tune, audio-video sync pass.

### 1.9 Acknowledgements and AI statement

New final-page section:

- Thank Prof. Watson for feedback across the three pipelines.
- Thank Samuel and Fortune for running ideas by.
- Disclose: Grammarly AI editor for sentence-level editing and grammatical correction. Claude Code used to brainstorm ideas for this pipeline, incorporating Prof. Watson's feedback. Ideas run by Samuel and Fortune.

### 1.10 Typos and mechanical cleanup

- Full paper typo scan (the user flagged `muxes` but actually `muxes` is the correct term for multiplexing; document both so we do not flip-flop).
- Every continuous-prose list needs to be a real `enumerate` or `itemize`.
- Every `\textbf{Foo.}` inline pseudo-heading needs to be a `\subsubsection*{Foo}`.

---

## 2. Workflow the user demanded

Verbatim directive:

> "create a master prompt file to store these, then deploy subagents to research and understand all that is needed and then create multiple plan docs for each subphase of these tasks and then deploy specialized agents (all opus 4.6) to work on the tasks to crisp detail and accuracy, then per the sprint review style, review after each change and ensure we are maintaining the quality expected and also all the sections required for the assignment and also all the writing styles I have said before!"

Operational interpretation:

1. **This file** is the master prompt. Every subagent reads it.
2. **Research phase.** Dispatch parallel research agents (opus) to inventory: from-scratch implementations; notebook cell state; available plots and artefacts; typos and inline-list offenders; GitHub repo state; untracked files.
3. **Planning phase.** For each subphase (§1.1 through §1.10 above), produce a plan doc under `docs/revamp_plans/` following the writing-plans skill.
4. **Implementation phase.** Dispatch opus implementers, one plan at a time, fresh subagent per subphase.
5. **Two-stage review after each subphase.** A spec-compliance reviewer checks the deliverable against this master prompt and the plan. A code-quality / prose-quality reviewer checks against `MASTER_PROMPT.md` style rules. Fixes get dispatched back to the implementer until both reviewers clear.
6. **Full-document flow review** after every subphase, per `feedback_pipeline3_sprint_cadence.md`.
7. **Recompile** the paper after every subphase. Zero errors. Zero overfulls. Zero undefined refs.

**Model policy for this revamp only:** user explicitly requested opus for all subagents. Override the project default of haiku.

**No user checkpoints between subphases.** The user has explicitly said do not ask "can I proceed". Just keep going until done.

---

## 3. Required final deliverables

1. `paper/pipeline3.pdf` rebuilt, clean compile, at the new target depth.
2. `notebooks/main_submission.ipynb` fully executed end-to-end, matching the PDF, with the same depth of derivation and explanation as the paper.
3. Public GitHub repo with the paper, notebook, source code, teasers, artefacts, and a `README.md` that links the teaser folder and the final PDF.
4. `REVAMP_MASTER_PROMPT.md` (this file) committed.
5. Plan docs under `docs/revamp_plans/` committed.

---

## 4. Section targets (where the most work goes)

Target depth bumps (current -> target prose length, indicative, not binding):

| Section | Current depth | Target depth | Priority |
|---|---|---|---|
| 5 Probabilistic and temporal (GP, Kalman, HMM) | moderate | deep derivations + interpretation + downstream use | high |
| 6 Relational (LightGCN, HAN) | moderate | same | high |
| 7 Causal and uncertainty (HB-ANOVA, IPW/AIPW, Conformal, BNN) | moderate | deep, per-method interpretation + use in Thompson | high |
| 8 Thompson | has derivation | tie back to §7 outputs explicitly | medium |
| 9 Generative stack (LoRA, SDXL, SVD-XT, StyleGAN3, MusicGen, ffmpeg) | shallow | deep LoRA/diffusion/GAN derivations + sample images + per-frame analysis | high |
| Appendix A Challenges | does not exist | full incident log | high |
| Appendix B AI statement | does not exist | acknowledgements + disclosure | high |
| Intro | exists | add repo/notebook/teaser hyperlinks | low |

---

## 5. Hard style guard (copied here so subagents cannot miss it)

- No em-dash (—) anywhere. En-dash (–) allowed only in numeric ranges.
- No "dive into", "unlock", "seamless", "game-changer", "leverage" (as verb), "navigate the complexities of", "in the realm of", "cinematic", "delve".
- No "it is important to note that", "notably", "crucially", "interestingly".
- First person singular is fine. He/him pronouns for the author.
- Numbers in prose: write out under ten, numerals above.
- Every acronym defined on first use with the full form then the parenthetical abbreviation.

---

## §1.11 Post-subphase-6 deep-detail revamp (2026-04-24 afternoon)

After subphases 1–6 closed, the user flagged ten additional binding items for a final polish pass. These override the original brief where they conflict.

1. **HB-ANOVA forest plot labels.** `artifacts/plots/anova_forest.png` currently puts modality names (`all`, `metadata`, `poster`, `synopsis`) on the x-axis and generic indices `a_m[0], [1], [2], [3]` on the y-axis. The y-axis is where modality labels belong. Fix the plotting code in `scripts/19_fit_hierarchical_anova.py` and regenerate. Rewrite the figure caption and surrounding HB-ANOVA interpretation text to explain what a random-intercept model means, what the 94% highest-density intervals (HDIs) represent, and how the shrinkage is visible.
2. **Caption depth, paper-wide.** Every figure caption must explain the plot, intuit what it shows, motivate why it matters, and connect it to a downstream artefact named by `\ref{}`. Rewrite the 1–2 interpretation paragraphs surrounding each figure at similar depth.
3. **§9 figures.** §9 (generative stack) currently has no in-section plots; only the two §9.9 sample-frame strips. Add per-model figures: LoRA adapter diagram, SDXL denoising progression, StyleGAN3 latent-space progression, SVD-XT temporal frame strip, MusicGen spectrogram, ffmpeg xfade timing chart.
4. **§9 technical depth.** Match the level of §5–§7. Per model: analytical derivation (forward-diffusion marginal and DDIM update for SDXL; low-rank ΔW = BA decomposition and gradient form for LoRA; equivariant filter / Jacobian constraint for StyleGAN3; temporal self-attention for SVD-XT; delay-pattern autoregression for MusicGen; xfade linear-ramp for ffmpeg assembly). Inline a small code snippet and its observed output for each.
5. **§9.9 sample frames done right.** The current `sdxl_posters_sample.png` and `stylegan3_frames_sample.png` were extracted from rendered teaser MP4s and therefore carry the baked-in title card (``A Temilola Olowolayemo Film / Coming Soon''), which makes them look identical and makes them misleading as ``raw model output''. Rebuild both strips from the RAW outputs in `_local_assembly/narrative_keyframes/scene_*.png` (15 SDXL scenes with logged prompts in `scene_metadata.json`) and `_local_assembly/stylegan3_output/trailer_scenes/*.jpg` (32 StyleGAN3 samples from the W space). Pick a diverse 2x3 grid per model. Write a per-frame caption explaining what the scene shows, what the prompt said, and which real film exemplar informed it.
6. **Show frames, not references.** Figures must be embedded in the LaTeX, not only cross-referenced.
7. **Architecture diagram placement.** Verify `artifacts/plots/pipeline_diagram.png` is correct against the shipped pipeline; update if drift exists. Move it from §1 Introduction to the end of the paper (Appendix figure, after Appendix A, before the bibliography). Keep a short pointer in §1.
8. **GitHub push status.** Confirm every change above is committed and pushed to `https://github.com/Temilola23/cs156-pipeline3`.
9. **From-scratch implementations in the notebook.** Match the Pipeline 1 and Pipeline 2 pedagogical style: write from-scratch NumPy/PyTorch implementations of the core methods (LoRA adapter forward pass, diffusion forward noise schedule + one DDIM reverse step, a progressive equivariant-filter StyleGAN toy, and two probability primitives such as the HMM forward algorithm or the GP posterior update). Run each on a tiny tensor to demonstrate correctness. Every pedagogical cell must be labelled explicitly as a from-scratch surrogate that complements (does not replace) the library-backed pipeline stage.
10. **Execution discipline.** Execute these ten items directly (no subagent dispatch), one after the other, with deep attention to detail. Verify each change against the compiled PDF before moving on. Append to this file so the plan history is preserved.

