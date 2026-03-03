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
NUM_SEEDS = 10  # Chạy 10 seeds để lấy trung bình (chuẩn IEEE)

def robust_qih_solver(Q_mat, n_users):
    bqm = {(i, j): Q_mat[i, j] for i in range(Q_mat.shape[0]) for j in range(i, Q_mat.shape[1])}
    sampler = neal.SimulatedAnnealingSampler()
    # Tăng num_reads lên một chút để đảm bảo chất lượng khi mạng lớn
    response = sampler.sample_qubo(bqm, num_reads=500)
    
    # Tìm nghiệm hợp lệ (Mỗi user đúng 1 link)
    for sample in response.samples():
        x_vec = [sample[i] for i in range(Q_mat.shape[0])]
        x_mat = np.array(x_vec).reshape(N_TRANS, n_users)
        if np.all(np.sum(x_mat, axis=0) == 1):
            return x_vec
    return [response.first.sample[i] for i in range(Q_mat.shape[0])]

def greedy_solver(G_matrix, n_users):
    x_gry = np.zeros((N_TRANS, n_users), dtype=int)
    for u in range(n_users):
        t_best = np.argmax(G_matrix[:, u])
        x_gry[t_best, u] = 1
    return x_gry.flatten().tolist()

if __name__ == "__main__":
    print(f"BẮT ĐẦU CHẠY THỰC NGHIỆM TỔNG THỂ ({NUM_SEEDS} SEEDS PER SCALE)")
    print(f"Hệ thống: {N_TRANS} Transmitters.")
    
    results_list = []

    for n_u in N_USERS_SCALES:
        print(f"\n>>> Đang chạy quy mô: {n_u} Users ({n_u * N_TRANS} Qubits)...")
        
        scale_rates_qio = []
        scale_rates_gry = []
        scale_times_qio = []
        
        for s in range(NUM_SEEDS):
            current_seed = 100 + s # Dùng các seed khác nhau
            G, Q = run_snapshot_mapping(n_u, seed=current_seed)
            
            # Giải bằng QIO
            r_qio, t_qio, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda mat: robust_qih_solver(mat, n_u))
            
            # Giải bằng Greedy
            r_gry, t_gry, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda mat: greedy_solver(G, n_u))
            
            scale_rates_qio.append(r_qio)
            scale_rates_gry.append(r_gry)
            scale_times_qio.append(t_qio)
            
            if (s+1) % 2 == 0:
                print(f"   - Seed {current_seed} xong.")

        # Tính trung bình cho mỗi quy mô
        results_list.append({
            'Users': n_u,
            'Qubits': n_u * N_TRANS,
            'SumRate_QIO_Mean': np.mean(scale_rates_qio),
            'SumRate_GRY_Mean': np.mean(scale_rates_gry),
            'Runtime_QIO_ms': np.mean(scale_times_qio),
            'Improvement_Pct': ((np.mean(scale_rates_qio) - np.mean(scale_rates_gry)) / np.mean(scale_rates_gry)) * 100
        })

    # Lưu kết quả ra CSV để vẽ hình
    df = pd.DataFrame(results_list)
    df.to_csv('final_paper_data.csv', index=False)
    
    print("\n" + "="*50)
    print("KẾT QUẢ CUỐI CÙNG (ĐÃ LẤY TRUNG BÌNH)")
    print("="*50)
    print(df[['Users', 'SumRate_QIO_Mean', 'SumRate_GRY_Mean', 'Improvement_Pct', 'Runtime_QIO_ms']])
    print("\nĐã lưu số liệu sạch vào file: final_paper_data.csv")
