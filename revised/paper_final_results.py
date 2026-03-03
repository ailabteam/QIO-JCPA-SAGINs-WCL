import numpy as np
import pandas as pd
import time
import sys
import os
import neal

# Đảm bảo nhận diện đúng các module từ thư mục cha
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qubo_mapping import run_snapshot_mapping, N_TRANS
from classical_optimization import solve_qubo_and_calculate_rate

# --- CẤU HÌNH THỰC NGHIỆM ---
N_USERS_SCALES = [4, 8, 12, 16] 
NUM_SEEDS = 10  # Chuẩn IEEE: Trung bình trên 10 realizations

def robust_qih_solver(Q_mat, n_users, reads):
    """Giải QUBO với số reads tùy chỉnh để giả lập Proposed vs Optimal"""
    bqm = {(i, j): Q_mat[i, j] for i in range(Q_mat.shape[0]) for j in range(i, Q_mat.shape[1])}
    sampler = neal.SimulatedAnnealingSampler()
    
    # Giải bài toán
    response = sampler.sample_qubo(bqm, num_reads=reads)
    
    # Tìm nghiệm hợp lệ đầu tiên (Mỗi user đúng 1 link) trong tập kết quả
    for sample in response.samples():
        x_vec = [sample[i] for i in range(Q_mat.shape[0])]
        x_mat = np.array(x_vec).reshape(N_TRANS, n_users)
        if np.all(np.sum(x_mat, axis=0) == 1):
            return x_vec
            
    # Nếu không tìm thấy nghiệm hợp lệ, lấy nghiệm có năng lượng thấp nhất
    return [response.first.sample[i] for i in range(Q_mat.shape[0])]

def greedy_solver(G_matrix, n_users):
    """Thuật toán tham lam (Baseline)"""
    x_gry = np.zeros((N_TRANS, n_users), dtype=int)
    for u in range(n_users):
        t_best = np.argmax(G_matrix[:, u])
        x_gry[t_best, u] = 1
    return x_gry.flatten().tolist()

if __name__ == "__main__":
    print(f"BẮT ĐẦU CHẠY THỰC NGHIỆM CHỐT (PROPOSED vs OPTIMAL vs GREEDY)")
    print(f"Hệ thống: {N_TRANS} Transmitters. Seeds: {NUM_SEEDS}")
    
    results_list = []

    for n_u in N_USERS_SCALES:
        print(f"\n>>> Đang quy mô: {n_u} Users ({n_u * N_TRANS} Qubits)...")
        
        # Các list để lưu kết quả của 10 seeds
        r_qio_list, r_opt_list, r_gry_list = [], [], []
        t_qio_list, t_opt_list, t_gry_list = [], [], []
        
        for s in range(NUM_SEEDS):
            current_seed = 200 + s # Dùng dải seed mới cho sạch dữ liệu
            G, Q = run_snapshot_mapping(n_u, seed=current_seed)
            
            # 1. Proposed QIO (Fast - 500 reads)
            rate_qio, time_qio, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda mat: robust_qih_solver(mat, n_u, 500))
            
            # 2. Near-Optimal Proxy (Slow - 5000 reads) - Đây là mốc 100%
            rate_opt, time_opt, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda mat: robust_qih_solver(mat, n_u, 5000))
            
            # 3. Greedy Baseline
            rate_gry, time_gry, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda mat: greedy_solver(G, n_u))
            
            r_qio_list.append(rate_qio)
            r_opt_list.append(rate_opt)
            r_gry_list.append(rate_gry)
            t_qio_list.append(time_qio)
            t_opt_list.append(time_opt)
            t_gry_list.append(time_gry)
            
            if (s+1) % 2 == 0:
                print(f"   - Seed {s+1}/{NUM_SEEDS} hoàn thành.")

        # Tính trung bình và Optimality Gap
        mean_qio = np.mean(r_qio_list)
        mean_opt = np.mean(r_opt_list)
        mean_gry = np.mean(r_gry_list)
        
        # Tránh chia cho 0 nếu có lỗi
        opt_gap = ((mean_opt - mean_qio) / mean_opt * 100) if mean_opt > 0 else 0
        gain_vs_gry = ((mean_qio - mean_gry) / mean_gry * 100) if mean_gry > 0 else 0

        results_list.append({
            'Users': n_u,
            'Qubits': n_u * N_TRANS,
            'Rate_QIO': mean_qio,
            'Rate_Optimal': mean_opt,
            'Rate_Greedy': mean_gry,
            'Time_QIO_ms': np.mean(t_qio_list),
            'Time_Optimal_ms': np.mean(t_opt_list),
            'Optimality_Gap_Pct': opt_gap,
            'Gain_vs_Greedy_Pct': gain_vs_gry
        })

    # Lưu kết quả
    df = pd.DataFrame(results_list)
    df.to_csv('final_paper_results_with_gap.csv', index=False)
    
    print("\n" + "="*80)
    print("KẾT QUẢ CUỐI CÙNG CHO BẢN THẢO (IEEE REVISION)")
    print("="*80)
    print(df[['Users', 'Rate_Optimal', 'Rate_QIO', 'Rate_Greedy', 'Optimality_Gap_Pct', 'Gain_vs_Greedy_Pct', 'Time_QIO_ms']])
    print("\nSố liệu đã lưu vào: final_paper_results_with_gap.csv")
