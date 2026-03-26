import numpy as np
import pandas as pd
import time
import sys
import os
import neal
import pygad # Thư viện Genetic Algorithm
import itertools

# Đảm bảo nhận diện đúng các module từ thư mục cha
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qubo_mapping import run_snapshot_mapping, N_TRANS
from classical_optimization import solve_qubo_and_calculate_rate, optimize_power_sca

# --- CẤU HÌNH THỰC NGHIỆM ---
N_USERS_SCALES = [4, 8, 12, 16] 
NUM_SEEDS = 10 

# --- 1. QIO SOLVER (Proposed) ---
def robust_qih_solver(Q_mat, n_users, reads):
    bqm = {(i, j): Q_mat[i, j] for i in range(Q_mat.shape[0]) for j in range(i, Q_mat.shape[1])}
    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample_qubo(bqm, num_reads=reads)
    for sample in response.samples():
        x_vec = [sample[i] for i in range(Q_mat.shape[0])]
        x_mat = np.array(x_vec).reshape(N_TRANS, n_users)
        if np.all(np.sum(x_mat, axis=0) == 1):
            return x_vec
    return [response.first.sample[i] for i in range(Q_mat.shape[0])]

# --- 2. GENETIC ALGORITHM SOLVER (New Baseline) ---
def ga_solver(Q_mat, n_users):
    N_QUBITS = N_TRANS * n_users
    
    def fitness_func(ga_instance, solution, solution_idx):
        # Tính năng lượng QUBO: x^T * Q * x
        energy = solution @ Q_mat @ solution.T
        # Phạt nặng nếu vi phạm ràng buộc mỗi user 1 link
        x_mat = solution.reshape(N_TRANS, n_users)
        penalty = 0
        if not np.all(np.sum(x_mat, axis=0) == 1):
            penalty = 1e6
        return - (energy + penalty) # GA là Maximize nên dùng dấu âm

    ga_instance = pygad.GA(num_generations=100,
                           num_parents_mating=5,
                           fitness_func=fitness_func,
                           sol_per_pop=20,
                           num_genes=N_QUBITS,
                           gene_space=[0, 1],
                           parent_selection_type="sss",
                           stop_criteria="reach_0")
    ga_instance.run()
    solution, _, _ = ga_instance.best_solution()
    return solution.tolist()

# --- 3. BRUTE-FORCE SOLVER (Ground-Truth cho mạng nhỏ) ---
def brute_force_optimum(G_matrix, n_users):
    if n_users > 6: return 0 # Quá lớn để duyệt
    user_choices = list(range(0, N_TRANS))
    all_combinations = list(itertools.product(user_choices, repeat=n_users))
    best_rate = 0.0
    for combo in all_combinations:
        X_current = np.zeros((N_TRANS, n_users))
        for u, t in enumerate(combo): X_current[t, u] = 1
        _, rate = optimize_power_sca(G_matrix, X_current.flatten(), n_users)
        if rate > best_rate: best_rate = rate
    return best_rate

# --- 4. MAIN PROCESS ---
if __name__ == "__main__":
    print(f"BẮT ĐẦU THỰC NGHIỆM ĐÓNG ĐINH (QIO vs GA vs BRUTE-FORCE)")
    results_list = []

    for n_u in N_USERS_SCALES:
        print(f"\n>>> Quy mô: {n_u} Users ({n_u * N_TRANS} Qubits)...")
        r_qio, r_opt, r_gry, r_ga, r_truth = [], [], [], [], []
        t_qio, t_ga = [], []

        for s in range(NUM_SEEDS):
            seed = 300 + s
            G, Q = run_snapshot_mapping(n_u, seed=seed)
            
            # Proposed QIO
            rate_qio, time_qio, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda mat: robust_qih_solver(mat, n_u, 500))
            
            # Near-Optimal Proxy
            rate_opt, _, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda mat: robust_qih_solver(mat, n_u, 5000))
            
            # Greedy
            rate_gry, _, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda mat: np.argmax(G, axis=0)) # simplified logic for speed
            
            # GA Baseline (New)
            start_ga = time.time()
            rate_ga, _, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda mat: ga_solver(mat, n_u))
            t_ga.append((time.time() - start_ga)*1000)

            # Ground-Truth (Chỉ chạy cho scale nhỏ nhất n_u=4)
            val_truth = brute_force_optimum(G, n_u) if n_u == 4 else 0

            r_qio.append(rate_qio); r_opt.append(rate_opt); r_gry.append(rate_gry); r_ga.append(rate_ga); r_truth.append(val_truth)
            t_qio.append(time_qio)
            print(f"   - Seed {s+1} hoàn thành.")

        results_list.append({
            'Users': n_u,
            'Rate_GroundTruth': np.mean(r_truth) if n_u == 4 else "N/A",
            'Rate_OptimalProxy': np.mean(r_opt),
            'Rate_Proposed_QIO': np.mean(r_qio),
            'Rate_GA': np.mean(r_ga),
            'Rate_Greedy': np.mean(r_gry),
            'Time_QIO_ms': np.mean(t_qio),
            'Time_GA_ms': np.mean(t_ga)
        })

    df = pd.DataFrame(results_list)
    df.to_csv('final_rigorous_results.csv', index=False)
    print("\n" + "="*80)
    print("KẾT QUẢ RIGOROUS CHO VÒNG PHẢN BIỆN CUỐI")
    print("="*80)
    print(df)
