import matplotlib.pyplot as plt
import numpy as np

# --- CHART 1: CLASS DISTRIBUTION (class_dist.png) ---
# Goal: Show the perfect balance (310 samples per class)
labels = ['Bioinformatics', 'Neuroscience', 'Material Science']
springer_counts = [310, 310, 310]
arxiv_counts = [310, 310, 310]

x = np.arange(len(labels))  # Label locations
width = 0.35  # Bar width

fig1, ax1 = plt.subplots(figsize=(8, 6))
# Plot Springer (Blue) and arXiv (Orange)
rects1 = ax1.bar(x - width/2, springer_counts, width, label='Springer (Source)', color='#1f77b4')
rects2 = ax1.bar(x + width/2, arxiv_counts, width, label='arXiv (Target)', color='#ff7f0e')

# Styling
ax1.set_ylabel('Number of Samples', fontsize=12)
ax1.set_title('Dataset Class Distribution (Balanced)', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=11)
ax1.legend()
ax1.set_ylim(0, 400) # Headroom for labels
ax1.bar_label(rects1, padding=3)
ax1.bar_label(rects2, padding=3)

plt.tight_layout()
plt.savefig('class_dist.png')
print("Saved class_dist.png")

# --- CHART 2: GENERALIZATION GAP (generalization_gap.png) ---
# Goal: Show SciBERT staying high while others drop
models = ['SciBERT', 'TextCNN', 'Bi-LSTM']
# Data from your results
springer_acc = [98.7, 97.6, 78.0]  # Rounded for cleaner chart
arxiv_acc = [91.2, 38.2, 40.0]

x2 = np.arange(len(models))
width2 = 0.35

fig2, ax2 = plt.subplots(figsize=(10, 6))
# Plot Springer (Green) and arXiv (Red)
rects_s = ax2.bar(x2 - width2/2, springer_acc, width2, label='Springer (Source)', color='#2ca02c')
rects_a = ax2.bar(x2 + width2/2, arxiv_acc, width2, label='arXiv (Target)', color='#d62728')

# Styling
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Generalization Gap: Source vs. Target Domain Performance', fontsize=14)
ax2.set_xticks(x2)
ax2.set_xticklabels(models, fontsize=12)
ax2.legend()
ax2.set_ylim(0, 115) # Headroom for labels

# Add percentage labels on top of bars
ax2.bar_label(rects_s, fmt='%.1f%%', padding=3)
ax2.bar_label(rects_a, fmt='%.1f%%', padding=3)

plt.tight_layout()
plt.savefig('generalization_gap.png')
print("Saved generalization_gap.png")