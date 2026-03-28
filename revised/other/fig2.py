import matplotlib.pyplot as plt

# Data extracted from Table IV 
iters = [500, 1000, 1500, 3000, 5000, 6000, 8000]
val_loss = [2.380, 1.317, 0.580, 0.545, 0.515, 0.537, 0.597]
char_acc = [0.172, 0.276, 0.887, 0.951, 0.982, 0.979, 0.981]

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10

fig, ax1 = plt.subplots(figsize=(6, 4))

# Trục tung bên trái cho Validation Loss
color = '#d62728' # IEEE Red
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Validation Loss', color=color, fontweight='bold')
line1, = ax1.plot(iters, val_loss, 'o--', color=color, linewidth=1.5, markersize=6, label='Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle=':', alpha=0.6)

# Trục tung bên phải cho Character Accuracy
ax2 = ax1.twinx()
color = '#1f77b4' # IEEE Blue
ax2.set_ylabel('Character Accuracy', color=color, fontweight='bold')
line2, = ax2.plot(iters, char_acc, 's-', color=color, linewidth=1.5, markersize=6, label='Accuracy')
ax2.tick_params(axis='y', labelcolor=color)

# Gộp chú thích (Legend)
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', frameon=True)

plt.title('Training Dynamics during Domain-Specific Finetuning', fontsize=11, fontweight='bold')
fig.tight_layout()
plt.savefig('convergence_analysis.pdf', dpi=600) # Xuất file vector cho IEEE

