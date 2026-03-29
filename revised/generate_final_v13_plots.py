import pandas as pd
import matplotlib.pyplot as plt

# 1. Load dữ liệu
df = pd.read_csv('final_v13_rigorous.csv')
df['Qubits'] = df['Users'] * 4 # Giả sử N_T = 4

# Cấu hình font chuẩn IEEE
plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'axes.grid': True})

# --- FIGURE 2: Average Sum Rate vs. System Scale ---
plt.figure(figsize=(7, 5))
plt.plot(df['Qubits'], df['Rate_Joint'], 'k--', label='Joint ICA (Upper Bound)', alpha=0.7)
plt.plot(df['Qubits'], df['Rate_QIO'], 'b-o', linewidth=2, markersize=8, label='Proposed QIO-JLSPA')
plt.plot(df['Qubits'], df['Rate_GA'], 'g-s', linewidth=1.5, label='Genetic Algorithm')
plt.plot(df['Qubits'], df['Rate_Greedy'], 'r-^', linewidth=1.5, label='Greedy Heuristic')

# Vẽ điểm Ground-Truth (Vét cạn) tại 24 Qubits
plt.scatter([24], [2.125], color='gold', marker='*', s=200, zorder=5, label='True Global Optimum (N=24)')

plt.xlabel('System Scale (Number of Qubits)')
plt.ylabel('Average Sum Rate (bits/s/Hz)')
plt.title('Spectral Efficiency Comparison')
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('fig2_sumrate_final.pdf')
print("Đã tạo fig2_sumrate_final.pdf")

# --- FIGURE 3: Runtime vs. System Scale ---
plt.figure(figsize=(7, 5))
plt.plot(df['Qubits'], df['Time_QIO'], 'b-o', linewidth=2, markersize=8, label='QIO Execution Time')
plt.plot(df['Qubits'], df['Time_GA'], 'g-s', linewidth=1.5, label='GA Execution Time')

# Đường ngưỡng 1 giây (Real-time threshold)
plt.axhline(y=1000, color='r', linestyle=':', alpha=0.6)
plt.text(30, 1050, '1-second Latency Threshold', color='r', fontsize=9)

plt.xlabel('System Scale (Number of Qubits)')
plt.ylabel('Execution Time (ms)')
plt.title('Computational Scalability')
plt.legend()
plt.tight_layout()
plt.savefig('fig3_runtime_final.pdf')
print("Đã tạo fig3_runtime_final.pdf")

# --- (Tùy chọn) FIGURE 4: Bar chart về Gain tại 96 Qubits ---
plt.figure(figsize=(6, 4))
labels = ['vs. Greedy', 'vs. GA']
gains = [((1.647/1.487)-1)*100, ((1.647/1.466)-1)*100]
plt.bar(labels, gains, color=['red', 'green'], alpha=0.7)
plt.ylabel('Sum-Rate Improvement (%)')
plt.title('QIO Gain at 96-Qubit Scale')
for i, v in enumerate(gains):
    plt.text(i, v + 0.5, f'+{v:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('fig4_gain_comparison.pdf')
