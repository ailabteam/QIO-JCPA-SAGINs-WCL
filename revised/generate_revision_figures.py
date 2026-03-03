import pandas as pd
import matplotlib.pyplot as plt

# Load dữ liệu
df = pd.read_csv('final_paper_results_with_gap.csv')

# Cấu hình font chuẩn IEEE (Serif)
plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'axes.grid': True})

# --- FIGURE 1: Sum Rate Comparison ---
plt.figure(figsize=(7, 5))
plt.plot(df['Qubits'], df['Rate_Optimal'], 'k--', label='Near-Optimal Bound', alpha=0.6)
plt.plot(df['Qubits'], df['Rate_QIO'], 'b-o', linewidth=2, label='Proposed QIO-JLSPA')
plt.plot(df['Qubits'], df['Rate_Greedy'], 'r-s', linewidth=2, label='Greedy Baseline')
plt.xlabel('System Scale (Number of Qubits)')
plt.ylabel('Average Sum Rate (nats/s/Hz)')
plt.title('Spectral Efficiency Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('fig_sumrate.pdf')
print("Đã tạo fig_sumrate.pdf")

# --- FIGURE 2: Gain vs Greedy (%) ---
plt.figure(figsize=(7, 5))
plt.bar(df['Qubits'].astype(str), df['Gain_vs_Greedy_Pct'], color='skyblue', edgecolor='navy', alpha=0.8)
plt.xlabel('System Scale (Number of Qubits)')
plt.ylabel('Sum-Rate Gain vs. Greedy (%)')
plt.title('Performance Advantage over Heuristics')
for i, val in enumerate(df['Gain_vs_Greedy_Pct']):
    plt.text(i, val + 0.1, f'{val:.2f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('fig_gain.pdf')
print("Đã tạo fig_gain.pdf")

# --- FIGURE 3: Runtime Scalability ---
plt.figure(figsize=(7, 5))
plt.plot(df['Qubits'], df['Time_QIO_ms'], 'g-D', linewidth=2, markersize=8, label='QIO Runtime')
plt.axhline(y=1000, color='r', linestyle=':', label='1-second Latency Threshold')
plt.xlabel('System Scale (Number of Qubits)')
plt.ylabel('Total Execution Time (ms)')
plt.title('Computational Scalability')
plt.legend()
plt.tight_layout()
plt.savefig('fig_runtime.pdf')
print("Đã tạo fig_runtime.pdf")
