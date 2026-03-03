import numpy as np
import time
import pandas as pd
import sys
import os

# Thêm path để nhận file cũ
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import neal
from qubo_mapping import run_snapshot_mapping, N_TRANS
from classical_optimization import solve_qubo_and_calculate_rate

# --- CẤU HÌNH THỰC TẾ ---
N_USERS_SCALES = [4, 8, 12, 16] # Scale lên 16 Users (64 Qubits)
R_ITERATIONS = 3 # Chạy 3 lần lấy trung bình cho nhanh

def run_real_benchmarks_snapshot(G_matrix, Q_matrix, n_users):
    
    # 1. Thuật toán QIO của bạn (100 reads - Tốc độ nhanh)
    def qih_neal_solver(Q_mat):
        bqm = {(i, j): Q_mat[i, j] for i in range(Q_mat.shape[0]) for j in range(i, Q_mat.shape[1])}
        sampler = neal.SimulatedAnnealingSampler()
        response = sampler.sample_qubo(bqm, num_reads=100)
        best_sample = response.first.sample
        return [best_sample[i] for i in range(Q_mat.shape[0])]

    # 2. Thuật toán HPQ (1000 reads - Đại diện cho Cận Tối Ưu Lượng Tử)
    def optimal_hpq_solver(Q_mat):
        bqm = {(i, j): Q_mat[i, j] for i in range(Q_mat.shape[0]) for j in range(i, Q_mat.shape[1])}
        sampler = neal.SimulatedAnnealingSampler()
        response = sampler.sample_qubo(bqm, num_reads=1000)
        best_sample = response.first.sample
        return [best_sample[i] for i in range(Q_mat.shape[0])]

    # 3. Thuật toán Greedy (Tham lam)
    def greedy_solver(Q_mat):
        x_gry = np.zeros((N_TRANS, n_users), dtype=int)
        for u in range(n_users):
            t_best = np.argmax(G_matrix[:, u])
            x_gry[t_best, u] = 1
        return x_gry.flatten().tolist()

    # CHẠY VÀ ĐO THỜI GIAN THẬT
    rate_hpq, runtime_hpq, _ = solve_qubo_and_calculate_rate(Q_matrix, G_matrix, n_users, optimal_hpq_solver)
    rate_qih, runtime_qih, _ = solve_qubo_and_calculate_rate(Q_matrix, G_matrix, n_users, qih_neal_solver)
    rate_gry, runtime_gry, _ = solve_qubo_and_calculate_rate(Q_matrix, G_matrix, n_users, greedy_solver)

    return {
        'Users': n_users,
        'Qubits': N_TRANS * n_users,
        'R_HPQ (Near-Optimum)': rate_hpq, 'T_HPQ_ms': runtime_hpq,
        'R_QIO (Proposed)': rate_qih,     'T_QIO_ms': runtime_qih,
        'R_GRY (Greedy)': rate_gry,       'T_GRY_ms': runtime_gry,
    }

if __name__ == "__main__":
    print("BẮT ĐẦU CHẠY MÔ PHỎNG SCALE-UP (DÙNG THỜI GIAN THỰC)")
    all_data = []
    
    for n_users in N_USERS_SCALES:
        print(f"\n--- Đang chạy quy mô {n_users} Users ({n_users * N_TRANS} Qubits) ---")
        for r_iter in range(R_ITERATIONS):
            seed = 42 + r_iter
            G, Q = run_snapshot_mapping(n_users, seed=seed)
            
            result = run_real_benchmarks_snapshot(G, Q, n_users)
            result['Iteration'] = r_iter
            all_data.append(result)
            print(f"  Lần {r_iter+1}: QIO Rate = {result['R_QIO (Proposed)']:.3f}, Greedy Rate = {result['R_GRY (Greedy)']:.3f}")

    # Tính trung bình
    df = pd.DataFrame(all_data)
    df_mean = df.groupby('Users').mean().reset_index()
    
    print("\n" + "="*50)
    print("KẾT QUẢ CUỐI CÙNG (TRUNG BÌNH)")
    print("="*50)
    print(df_mean[['Users', 'R_HPQ (Near-Optimum)', 'R_QIO (Proposed)', 'R_GRY (Greedy)']])
    
    df_mean.to_csv('real_scale_results.csv', index=False)
    print("\nĐã lưu kết quả vào real_scale_results.csv")
