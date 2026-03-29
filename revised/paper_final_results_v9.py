import numpy as np
import pandas as pd
import sys, os, neal, itertools, time, pygad
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qubo_mapping import run_snapshot_mapping, N_TRANS
from classical_optimization import solve_qubo_and_calculate_rate, optimize_power_sca

# --- 1. GA Solver (Baseline mạnh) ---
def ga_solver(Q_mat, n_users):
    num_genes = N_TRANS * n_users
    def fitness_func(ga_instance, solution, solution_idx):
        energy = np.dot(solution, np.dot(Q_mat, solution))
        x_mat = solution.reshape(N_TRANS, n_users)
        penalty = 0 if np.all(np.sum(x_mat, axis=0) == 1) else 1e5
        return -(energy + penalty)
    
    ga_inst = pygad.GA(num_generations=50, num_parents_mating=5, fitness_func=fitness_func,
                       sol_per_pop=20, num_genes=num_genes, gene_space=[0, 1],
                       stop_criteria="reach_0", suppress_warnings=True)
    ga_inst.run()
    sol, _, _ = ga_inst.best_solution()
    return sol.tolist()

# --- 2. QIO Solver (Proposed) ---
def robust_qio_solver(Q_mat, n_users):
    bqm = {(i, j): Q_mat[i, j] for i in range(Q_mat.shape[0]) for j in range(i, Q_mat.shape[1])}
    res = neal.SimulatedAnnealingSampler().sample_qubo(bqm, num_reads=1000)
    for s in res.samples():
        x = np.array([s[i] for i in range(Q_mat.shape[0])]).reshape(N_TRANS, n_users)
        if np.all(np.sum(x, axis=0) == 1): return [s[i] for i in range(Q_mat.shape[0])]
    return [res.first.sample[i] for i in range(Q_mat.shape[0])]

# --- 3. Brute Force (Ground Truth) ---
def brute_force_strict(G, n_u):
    if n_u > 4: return 0
    choices = list(itertools.product(range(N_TRANS), repeat=n_u))
    best_r = 0
    for c in choices:
        x = np.zeros((N_TRANS, n_u))
        for u, t in enumerate(c): x[t, u] = 1
        _, r = optimize_power_sca(G, x.flatten(), n_u)
        if r > best_r: best_r = r
    return best_r

if __name__ == "__main__":
    print(">>> RUNNING RIGOROUS V9 (PROPOSED vs GA vs GREEDY vs TRUTH)...")
    N_SCALES = [4, 8, 12, 16]
    results = []
    for n_u in N_SCALES:
        print(f"Scale: {n_u} Users...")
        r_q, r_ga, r_g, r_t = [], [], [], []
        t_q, t_ga = [], []
        for s in range(10):
            G, Q = run_snapshot_mapping(n_u, seed=1000+s)
            # QIO
            rate_q, time_q, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda m: robust_qio_solver(m, n_u))
            # GA
            t_ga_start = time.time()
            rate_ga, _, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda m: ga_solver(m, n_u))
            t_ga_end = time.time()
            # Greedy
            x_g = np.zeros((N_TRANS, n_u))
            for u in range(n_u): x_g[np.argmax(G[:, u]), u] = 1
            rate_g, _, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda m: x_g.flatten().tolist())
            # Truth
            rate_t = brute_force_strict(G, n_u) if n_u == 4 else 0
            
            r_q.append(rate_q); r_ga.append(rate_ga); r_g.append(rate_g); r_t.append(rate_t)
            t_q.append(time_q); t_ga.append((t_ga_end - t_ga_start)*1000)
        
        results.append({
            'Users': n_u, 'Rate_Truth': np.mean(r_t) if n_u==4 else 0,
            'Rate_QIO': np.mean(r_q), 'Rate_GA': np.mean(r_ga), 'Rate_Greedy': np.mean(r_g),
            'Time_QIO': np.mean(t_q), 'Time_GA': np.mean(t_ga)
        })
    df = pd.DataFrame(results)
    print("\n" + "="*80 + "\nBẢNG KẾT QUẢ ĐẲNG CẤP (FINAL RIGOROUS BATTLE)\n" + "="*80)
    print(df)
    df.to_csv('final_v9_results.csv', index=False)
