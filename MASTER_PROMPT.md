# Master Prompt — Pipeline 3 Paper Rewrite

**Captured 2026-04-24.** Governs every paper-writing decision from this point forward. Refer back to this doc at the start of every subsection sprint.

---

## What the user hated about the current draft

1. **No apples-to-apples comparison with prior assignments** (Pipeline 1, Pipeline 2). The brief explicitly asks for this. Every model family used now must be compared against what was used before.
2. **Bad prose.** Flowery, assertive, generic. "Supervised tournament" is a nonsense phrase. "Portfolio, not tournament" — fixed nouns matter.
3. **Uninterpreted tables and outputs.** Tables dumped without narrative; code outputs not interpreted; transitions absent.
4. **Dense paragraphs like the Latin-square block** that claim a result without motivating, introducing, or deriving. Reader cannot follow.
5. **AI slop and em-dashes everywhere.** Must be stripped.
6. **"ACT I / ACT II / ACT III" framing** was never introduced. Question whether to use it at all in an academic paper (the answer is: probably not — use standard academic structure).
7. **Formatting is weak.** Lists are not formatted as lists. No LaTeX template. No Minerva / CS156 logo. Pipeline 1's LaTeX is the target look.
8. **Acronyms used without definition.** No glossary.
9. **Methods spammed without motivation.** Each method must be *motivated → introduced → derived → applied → interpreted → transitioned-from*.
10. **No derivation on core concepts.** Latin-square, Thompson bandit, etc. — derive from first principles for `#cs156-MLMath`.

## Non-negotiable style rules

- **No em-dashes.** Use commas, parentheses, colons, or new sentences.
- **No AI slop phrases.** Search-and-destroy: "navigate", "delve", "in today's world", "tapestry", "landscape" (metaphorical), "robust" (unless technically defined), "leverage" (as verb), "seamless", "journey" (metaphorical), "testament", "at the heart of", "underscore", "unpacks", "dive into", "rich tapestry", "it's important to note", "to put it simply", "as we can see", "a nuanced understanding", "critical", "pivotal", "key takeaways", "elevate", "game-changer", "cutting-edge" (unless technically true), "lush", "boasts", "paradigm shift", "in the realm of", "shines a light", "sheds light on", "intricate", "stands as a testament", "at its core".
- **Short, direct sentences.** Each sentence earns its place.
- **Every claim is either derived, cited, or labelled an opinion.**
- **Every table is introduced before it appears and interpreted after.**
- **Every figure has a caption that says what the reader should notice.**
- **Every acronym defined on first use and present in the glossary.**

## Learning outcomes (from LOs.txt) that must be hit

- `#cs156-MLCode` — Implement non-trivial methods by hand (NumPy), not just library calls. Explain and justify each.
- `#cs156-MLExplanation` — Intuitive and analytical explanation of each ML choice, using ML lingo.
- `#cs156-MLFlexibility` — Go beyond what was covered in class (Sessions 1 to 24 cover through Transformers).
- `#cs156-MLMath` — Derive from first principles with linear algebra, multivariate calculus, Bayesian statistics, using examples and intuition.

## Structural framing decision

**Drop "ACT I / II / III".** It is cinematic framing that does not belong in an academic ML paper. Use standard sections:

1. Introduction and scope
2. Data collection protocol (self-elicitation experiment)
3. Exploratory data analysis
4. Data augmentation (MovieLens twin, GenMatch, TVAE)
5. Probabilistic and temporal models (GP, Kalman ladder, HMM)
6. Relational models (LightGCN, HAN)
7. Causal and uncertainty estimation (Hierarchical Bayes, IPW and AIPW, Conformal, BNN)
8. Exploration (Thompson sampling)
9. Generative stack for teaser production (Llama + QLoRA, SDXL + LoRA, SVD-XT + LoRA, StyleGAN3 + ADA, MusicGen, ffmpeg)
10. Apples-to-apples comparison with Pipeline 1 and Pipeline 2
11. Honest diagnosis of failures
12. Conclusions and learning-outcome mapping
13. Glossary
14. References

## Writing workflow (per subsection sprint)

For each subsection:

1. **Motivate.** One paragraph. Why does the reader care? What problem from the EDA demands this method? Name the ML lingo.
2. **Introduce.** Define the method in plain English. Define each acronym on first use. Say what family it is in.
3. **Derive.** First-principles derivation for the core object. Use inline math sparingly and boxed equations for key results. Numbered steps for multi-line derivations.
4. **Apply.** Code cell that runs against either synthetic data (from-scratch sanity check) or the real artifact.
5. **Interpret.** Paragraph that reads the numerical output and says what it means. Connect back to the motivation.
6. **Transition.** One sentence that bridges to the next subsection.

After each sprint: self-review against this prompt, run the notebook cell, check LaTeX compile, only then move on.

## Comparison to Pipeline 1 and Pipeline 2 (mandatory)

Pipeline 3 must include a comparison section that lists, per model family:

| Family | Pipeline 1 | Pipeline 2 | Pipeline 3 | Why we upgraded |
|--------|------------|------------|------------|------------------|
| Supervised baseline | (what P1 used) | (what P2 used) | (what P3 uses) | (justification) |
| Ensemble | ... | ... | ... | ... |
| Graph / relational | none | none | LightGCN + HAN | ... |
| Uncertainty | none | bootstrap CI | conformal + BNN | ... |
| Causal | none | none | IPW + AIPW | ... |
| Time series | none | none | HMM + Kalman ladder | ... |

The "why we upgraded" column is the intellectual through-line of the paper.

## LaTeX target

Use the Pipeline 1 `CS114.tex` template (located in `archived/pipeline1_root/`). Include the `minerva_logo.png` on the title page. Match Pipeline 1's typography, margins, and section-header style.

## Acceptance gate per sprint

A subsection ships when:

1. Prose reads cleanly with no em-dashes, no AI slop.
2. Every acronym is defined on first use.
3. Tables and figures have narrative scaffolding before and after.
4. At least one derivation from first principles appears.
5. Code executes and output is interpreted.
6. Transitions to and from neighbours flow.
7. LaTeX compiles without warnings.

---

**End of master prompt.** Return here at the start of every sprint.
