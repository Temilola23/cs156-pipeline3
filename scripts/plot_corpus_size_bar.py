#!/usr/bin/env python3
"""
Bar chart of training corpus size before and after augmentation.

Displays: 324 real ratings, 77,678 MovieLens twin, 492 GenMatch cohort,
1,000 TVAE samples, and the total 79,430 augmented rows.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data (from paper)
sources = ['Original\nRatings', 'MovieLens\nTwin', 'GenMatch\nCohort', 'TVAE\nSynthetic', 'Total\nAugmented']
sizes = [324, 77678, 492, 1000, 79430]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Create horizontal bar chart
fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(sources))

bars = ax.barh(y_pos, sizes, color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)

ax.set_yticks(y_pos)
ax.set_yticklabels(sources, fontsize=11)
ax.set_xlabel('Number of Rows', fontsize=12, fontweight='bold')
ax.set_title('Training Corpus: Real Data vs. Augmentation Pillars', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, (bar, size) in enumerate(zip(bars, sizes)):
    label = f'{size:,}' if i < 4 else f'{size:,}'
    ax.text(size + 1000, bar.get_y() + bar.get_height() / 2,
            label, va='center', ha='left', fontsize=10, fontweight='bold')

# Log scale on x to make small values visible
ax.set_xscale('log')
ax.set_xlim(100, 150000)

plt.tight_layout()
output_path = Path(__file__).parent.parent / "artifacts" / "plots" / "corpus_size_bar.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved: {output_path}")
plt.close()
