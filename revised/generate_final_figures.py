import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load dữ liệu từ CSV
df = pd.read_csv('final_paper_data.csv')

# Thiết lập phong cách vẽ hình chuẩn IEEE
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# --- FIGURE 1: Sum-Rate vs. System Scale ---
plt.figure(figsize=(7, 5))
plt.plot(df['Qubits'], df['SumRate_QIO_Mean'], 'b-o', linewidth=2, markersize=8, label='Proposed QIO-JLSPA')
plt.plot(df['Qubits'], df['SumRate_GRY_Mean'], 'r--s', linewidth=2, markersize=8, label='Greedy Heuristic')

plt.xlabel('System Scale (Number of Qubits)')
plt.ylabel('Average Sum Rate (nats/s/Hz)')
plt.title('Spectral Efficiency vs. Network Scale')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.xticks(df['Qubits'])
plt.tight_layout()
plt.savefig('figure1_sumrate_scale.pdf')
print("Đã lưu figure1_sumrate_scale.pdf")

# --- FIGURE 2: Runtime vs. System Scale ---
plt.figure(figsize=(7, 5))
plt.plot(df['Qubits'], df['Runtime_QIO_ms'], 'g-^', linewidth=2, markersize=8, label='QIO Execution Time')

plt.xlabel('System Scale (Number of Qubits)')
plt.ylabel('Average Total Runtime (ms)')
plt.title('Computational Scalability')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.xticks(df['Qubits'])
plt.tight_layout()
plt.savefig('figure2_runtime_scale.pdf')
print("Đã lưu figure2_runtime_scale.pdf")

# --- XUẤT DỮ LIỆU CHO TABLE II (LATEX) ---
print("\n--- NỘI DUNG CHO TABLE II (Copy vào LaTeX) ---")
for index, row in df.iterrows():
    print(f"{int(row['Users'])} & {int(row['Qubits'])} & {row['SumRate_GRY_Mean']:.3f} & {row['SumRate_QIO_Mean']:.3f} & {row['Improvement_Pct']:+.2f}\\% & {row['Runtime_QIO_ms']:.1f} \\\\")
