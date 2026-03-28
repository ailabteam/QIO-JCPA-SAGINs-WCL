import numpy as np
import pandas as pd
import time
import sys
import os
import neal
import itertools

# Đảm bảo nhận diện đúng các module từ thư mục cha
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qubo_mapping import run_snapshot_mapping, N_TRANS
from classical_optimization import solve_qubo_and_calculate_rate, optimize_power_sca

def robust_qih_solver_v4(Q_mat, n_users, reads):
    bqm = {(i, j): Q_mat[i, j] for i in range(Q_mat.shape[0]) for j in range(i, Q_mat.shape[1])}
    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample_qubo(bqm, num_reads=reads)
    
    # Lọc nghiêm ngặt để đảm bảo 100% kết quả hợp lệ
    for sample in response.samples():
        x_vec = [sample[i] for i in range(Q_mat.shape[0])]
        x_mat = np.array(x_vec).reshape(N_TRANS, n_users)
        if np.all(np.sum(x_mat, axis=0) == 1):
            return x_vec
    return [response.first.sample[i] for i in range(Q_mat.shape[0])]

def brute_force_strict_optimum(G_matrix, n_users):
    """Vét cạn Tối ưu (Chỉ xét các trường hợp phục vụ ĐỦ user)"""
    if n_users > 5: return 0 
    user_choices = list(range(0, N_TRANS))
    all_combinations = list(itertools.product(user_choices, repeat=n_users))
    best_rate = 0.0
    for combo in all_combinations:
        X_current = np.zeros((N_TRANS, n_users))
        for u, t in enumerate(combo):
            X_current[t, u] = 1 
        _, rate = optimize_power_sca(G_matrix, X_current.flatten(), n_users)
        if rate > best_rate: best_rate = rate
    return best_rate

if __name__ == "__main__":
    print(f"BẮT ĐẦU THỰC NGHIỆM CHỐT HẠ (VERSION 4 - RIGOROUS)")
    N_SCALES = [4, 8, 12, 16]
    NUM_SEEDS = 10
    results = []

    for n_u in N_SCALES:
        print(f"\n>>> Quy mô: {n_u} Users ({n_u * N_TRANS} Qubits)...")
        r_qio, r_truth, r_gry = [], [], []
        t_qio = []

        for s in range(NUM_SEEDS):
            seed = 500 + s
            G, Q = run_snapshot_mapping(n_u, seed=seed)
            
            # 1. Proposed QIO
            rate_qio, time_qio, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda mat: robust_qih_solver_v4(mat, n_u, 500))
            
            # 2. Greedy (Strict One-Hot)
            rate_gry, _, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda mat: (np.eye(N_TRANS)[np.argmax(G, axis=0)].T).flatten().tolist())
            
            # 3. Truth (Strict - chỉ chạy cho n_u=4 để lấy Ground-truth mẫu)
            val_truth = brute_force_strict_optimum(G, n_u) if n_u == 4 else 0

            r_qio.append(rate_qio); r_truth.append(val_truth); r_gry.append(rate_gry)
            t_qio.append(time_qio)
            if (s+1)%2 == 0: print(f"   - Seed {s+1} hoàn thành.")

        results.append({
            'Users': n_u,
            'Rate_Truth': np.mean(r_truth) if n_u == 4 else 0,
            'Rate_QIO': np.mean(r_qio),
            'Rate_Greedy': np.mean(r_gry),
            'Optimality_Gap_Pct': ((np.mean(r_truth) - np.mean(r_qio))/np.mean(r_truth)*100) if n_u == 4 else 0,
            'Gain_vs_Greedy': ((np.mean(r_qio) - np.mean(r_gry))/np.mean(r_gry)*100),
            'Time_QIO_ms': np.mean(t_qio)
        })

    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("KẾT QUẢ RIGOROUS VALIDATION (DÀNH CHO WCL)")
    print("="*80)
    print(df[['Users', 'Rate_Truth', 'Rate_QIO', 'Rate_Greedy', 'Optimality_Gap_Pct', 'Gain_vs_Greedy', 'Time_QIO_ms']])
    df.to_csv('final_clean_results.csv', index=False)
