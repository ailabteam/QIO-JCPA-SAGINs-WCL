import numpy as np
import pandas as pd
import time
import sys
import os
import neal
import pygad
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

# --- 2. GENETIC ALGORITHM SOLVER (Standard Baseline) ---
def ga_solver(Q_mat, n_users):
    N_QUBITS = N_TRANS * n_users
    
    def fitness_func(ga_instance, solution, solution_idx):
        # Tính năng lượng: x^T * Q * x
        energy = np.dot(solution, np.dot(Q_mat, solution))
        # Phạt nặng nếu vi phạm constraint (mỗi user 1 link)
        x_mat = solution.reshape(N_TRANS, n_users)
        penalty = 0
        if not np.all(np.sum(x_mat, axis=0) == 1):
            penalty = 1e5
        return - (energy + penalty)

    ga_instance = pygad.GA(num_generations=50,
                           num_parents_mating=10,
                           fitness_func=fitness_func,
                           sol_per_pop=40,
                           num_genes=N_QUBITS,
                           gene_space=[0, 1],
                           parent_selection_type="sss",
                           crossover_type="single_point",
                           mutation_type="random",
                           mutation_probability=0.1,
                           stop_criteria="reach_0")
    ga_instance.run()
    solution, _, _ = ga_instance.best_solution()
    return solution.tolist()

# --- 3. GREEDY SOLVER (Sửa lỗi Reshape) ---
def greedy_solver_binary(G_matrix, n_users):
    x_mat = np.zeros((N_TRANS, n_users))
    for u in range(n_users):
        t_best = np.argmax(G_matrix[:, u])
        x_mat[t_best, u] = 1
    return x_mat.flatten().tolist()

# --- 4. BRUTE-FORCE OPTIMUM ---
def brute_force_optimum(G_matrix, n_users):
    if n_users > 5: return 0 
    user_choices = list(range(0, N_TRANS))
    all_combinations = list(itertools.product(user_choices, repeat=n_users))
    best_rate = 0.0
    for combo in all_combinations:
        X_current = np.zeros((N_TRANS, n_users))
        for u, t in enumerate(combo): X_current[t, u] = 1
        _, rate = optimize_power_sca(G_matrix, X_current.flatten(), n_users)
        if rate > best_rate: best_rate = rate
    return best_rate

# --- 5. TIẾN TRÌNH CHÍNH ---
if __name__ == "__main__":
    print(f"BẮT ĐẦU THỰC NGHIỆM RIGOROUS (QIO vs GA vs BRUTE-FORCE)")
    print(f"Hệ thống: {N_TRANS} Transmitters. Seeds: {NUM_SEEDS}")
    results_list = []

    for n_u in N_USERS_SCALES:
        print(f"\n>>> Quy mô: {n_u} Users ({n_u * N_TRANS} Qubits)...")
        r_qio, r_opt, r_gry, r_ga, r_truth = [], [], [], [], []
        t_qio, t_ga = [], []

        for s in range(NUM_SEEDS):
            seed = 400 + s
            G, Q = run_snapshot_mapping(n_u, seed=seed)
            
            # 1. Proposed QIO (500 reads)
            rate_qio, time_qio, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda mat: robust_qih_solver(mat, n_u, 500))
            
            # 2. Near-Optimal Proxy (5000 reads)
            rate_opt, _, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda mat: robust_qih_solver(mat, n_u, 5000))
            
            # 3. Greedy
            rate_gry, _, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda mat: greedy_solver_binary(G, n_u))
            
            # 4. GA Baseline
            start_ga = time.time()
            rate_ga, _, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda mat: ga_solver(mat, n_u))
            t_ga.append((time.time() - start_ga)*1000)

            # 5. Ground-Truth (Chỉ cho scale nhỏ nhất n_u=4 để tiết kiệm thời gian)
            val_truth = brute_force_optimum(G, n_u) if n_u == 4 else 0

            r_qio.append(rate_qio); r_opt.append(rate_opt); r_gry.append(rate_gry); r_ga.append(rate_ga); r_truth.append(val_truth)
            t_qio.append(time_qio)
            print(f"   - Seed {s+1} hoàn thành.")

        results_list.append({
            'Users': n_u,
            'Qubits': n_u * N_TRANS,
            'Rate_Truth_4u': np.mean(r_truth) if n_u == 4 else 0,
            'Rate_OptProxy': np.mean(r_opt),
            'Rate_QIO': np.mean(r_qio),
            'Rate_GA': np.mean(r_ga),
            'Rate_Greedy': np.mean(r_gry),
            'Time_QIO_ms': np.mean(t_qio),
            'Time_GA_ms': np.mean(t_ga)
        })

    df = pd.DataFrame(results_list)
    df.to_csv('final_rigorous_results_v3.csv', index=False)
    print("\n" + "="*80)
    print("BẢNG KẾT QUẢ CUỐI CÙNG CHO PHẢN BIỆN (RIGOROUS VALIDATION)")
    print("="*80)
    print(df[['Users', 'Rate_Truth_4u', 'Rate_OptProxy', 'Rate_QIO', 'Rate_GA', 'Rate_Greedy', 'Time_QIO_ms', 'Time_GA_ms']])
