import numpy as np
import matplotlib.pyplot as plt

# Data extracted from Table VIII 
# Models: PaddleOCR, Claude Haiku (VLM), Transformer (Pretrained), Transformer (Finetuned)
categories = ['Low (<10%)', 'Med (10-25%)', 'High (≥25%)']
paddle = [4.4, 17.5, 25.4]
haiku = [6.8, 2.6, 3.0]
trans_pre = [4.7, 1.2, 1.4]
trans_ft = [6.3, 0.1, 0.1]

x = np.arange(len(categories))
width = 0.2

fig, ax = plt.subplots(figsize=(7, 4.5))

rects1 = ax.bar(x - 1.5*width, paddle, width, label='PaddleOCR', color='#7f7f7f', hatch='//')
rects2 = ax.bar(x - 0.5*width, haiku, width, label='Claude Haiku', color='#bcbd22', hatch='..')
rects3 = ax.bar(x + 0.5*width, trans_pre, width, label='VietOCR (Pre)', color='#1f77b4', hatch='\\\\')
rects4 = ax.bar(x + 1.5*width, trans_ft, width, label='VietOCR (ft)', color='#2ca02c')

ax.set_ylabel('Character Error Rate (CER %)', fontweight='bold')
ax.set_title('Performance Comparison across Diacritic Density Levels', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(fontsize=9)

# Sử dụng thang Log để làm nổi bật sự khác biệt ở mức CER thấp của mô hình ft
ax.set_yscale('log')
ax.set_ylim(0.05, 50) 
ax.grid(axis='y', linestyle='--', alpha=0.5)

fig.tight_layout()
plt.savefig('diacritic_density_impact.pdf', dpi=600)

