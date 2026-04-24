#!/usr/bin/env python3
"""
Add a prose reference to Figure~\ref{fig:propensity_overlap} in the AIPW interpretation.
"""

from pathlib import Path

# Read the paper
paper_path = Path(__file__).parent.parent / "paper" / "pipeline3.tex"
with open(paper_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the sentence to update
old_text = "The interval contains zero by a substantial margin. At $n = 162$, we cannot reject the null hypothesis that presenting a film as a synopsis versus as structured metadata has no causal effect on rating."

new_text = "The interval contains zero by a substantial margin. Figure~\\ref{fig:propensity_overlap} shows the propensity-score overlap diagnostic, confirming that the GenMatch covariate balance is sufficient for identification. At $n = 162$, we cannot reject the null hypothesis that presenting a film as a synopsis versus as structured metadata has no causal effect on rating."

# Verify both strings exist
if old_text not in content:
    print(f"ERROR: Could not find old text")
    exit(1)

# Replace
new_content = content.replace(old_text, new_text)

# Verify change happened
if new_content == content:
    print("ERROR: Replacement failed")
    exit(1)

# Write back
with open(paper_path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print("✓ Added prose reference to Figure~\\ref{fig:propensity_overlap} in AIPW interpretation")
