import numpy as np
import pandas as pd
import time
import sys
import os
import neal
import itertools

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qubo_mapping import run_snapshot_mapping, N_TRANS
from classical_optimization import solve_qubo_and_calculate_rate, optimize_power_sca

def qio_solver_v5(Q_mat, n_users, reads):
    bqm = {(i, j): Q_mat[i, j] for i in range(Q_mat.shape[0]) for j in range(i, Q_mat.shape[1])}
    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample_qubo(bqm, num_reads=reads)
    
    # Chỉ lọc: 1 user không được dùng quá 1 link (Admission Control cho phép 0 link)
    for sample in response.samples():
        x_vec = [sample[i] for i in range(Q_mat.shape[0])]
        x_mat = np.array(x_vec).reshape(N_TRANS, n_users)
        if np.all(np.sum(x_mat, axis=0) <= 1): # Chấp nhận <= 1
            return x_vec
    return [response.first.sample[i] for i in range(Q_mat.shape[0])]

if __name__ == "__main__":
    print(f"BẮT ĐẦU LẤY SỐ LIỆU CUỐI CÙNG (ADMISSION CONTROL STRATEGY)")
    N_SCALES = [4, 8, 12, 16]
    NUM_SEEDS = 10
    results = []

    for n_u in N_SCALES:
        print(f"\n>>> Quy mô: {n_u} Users ({n_u * N_TRANS} Qubits)...")
        r_qio, r_gry, t_qio = [], [], []

        for s in range(NUM_SEEDS):
            seed = 600 + s
            G, Q = run_snapshot_mapping(n_u, seed=seed)
            
            # 1. Proposed QIO (Thông minh - biết chọn user để né nhiễu)
            rate_qio, time_qio, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda mat: qio_solver_v5(mat, n_u, 1000))
            
            # 2. Greedy (Máy móc - cố gắng phục vụ tất cả)
            rate_gry, _, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda mat: (np.eye(N_TRANS)[np.argmax(G, axis=0)].T).flatten().tolist())
            
            r_qio.append(rate_qio); r_gry.append(rate_gry); t_qio.append(time_qio)
            if (s+1)%2 == 0: print(f"   - Seed {s+1} xong.")

        results.append({
            'Users': n_u,
            'Rate_QIO': np.mean(r_qio),
            'Rate_Greedy': np.mean(r_gry),
            'Improvement_x': np.mean(r_qio) / np.mean(r_gry),
            'Time_ms': np.mean(t_qio)
        })

    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("KẾT QUẢ CUỐI CÙNG - ĐÃ ĐỦ SỨC THUYẾT PHỤC TUYỆT ĐỐI")
    print("="*80)
    print(df)
    df.to_csv('final_submission_data.csv', index=False)
