#!/usr/bin/env python3
"""
Insert propensity_overlap.png figure into the IPW/AIPW section (§7.3).
Located after the AIPW interpretation paragraph and before "Split conformal prediction" subsection.
"""

from pathlib import Path

# Read the paper
paper_path = Path(__file__).parent.parent / "paper" / "pipeline3.tex"
with open(paper_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the anchor text (end of AIPW interpretation section)
anchor_text = "The honest reading is that modality does not drive ratings under this Latin-square design. Section~\\ref{sec:conclusion} returns to implications for sample size planning."

# Verify anchor exists
if anchor_text not in content:
    print("ERROR: Could not find anchor text for propensity figure insertion.")
    print(f"Expected: {anchor_text[:80]}...")
    exit(1)

# Create the figure block with raw string for proper escaping
figure_block = """

\\begin{figure}[ht]
  \\centering
  \\includegraphics[width=0.85\\linewidth]{../artifacts/plots/propensity_overlap.png}
  \\caption{Propensity-score overlap diagnostic between treated (synopsis) and control (metadata) cohorts after GenMatch matching. The region of common support validates that the propensity model has balanced the covariate distributions sufficiently for the AIPW estimator to apply.}
  \\label{fig:propensity_overlap}
\\end{figure}
"""

# Insert the figure after the anchor
new_content = content.replace(
    anchor_text,
    anchor_text + figure_block
)

# Verify the insertion
if new_content == content:
    print("ERROR: Insertion failed—content unchanged.")
    exit(1)

# Write back
with open(paper_path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f"✓ Inserted propensity_overlap.png figure into §7.3 IPW/AIPW section")
print(f"  Location: after AIPW interpretation paragraph")
print(f"  Label: fig:propensity_overlap")
print(f"  Paper: {paper_path}")
