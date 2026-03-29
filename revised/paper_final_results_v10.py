import numpy as np
import pandas as pd
import sys, os, neal, itertools, time, pygad
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from qubo_mapping import run_snapshot_mapping, N_TRANS
from classical_optimization import solve_qubo_and_calculate_rate, optimize_power_sca

def robust_qio_solver(Q_mat, n_users, reads=1000):
    bqm = {(i, j): Q_mat[i, j] for i in range(Q_mat.shape[0]) for j in range(i, Q_mat.shape[1])}
    sampler = neal.SimulatedAnnealingSampler()
    res = sampler.sample_qubo(bqm, num_reads=reads)
    for s in res.samples():
        x = np.array([s[i] for i in range(Q_mat.shape[0])]).reshape(N_TRANS, n_users)
        if np.all(np.sum(x, axis=0) == 1): return [s[i] for i in range(Q_mat.shape[0])]
    return [res.first.sample[i] for i in range(Q_mat.shape[0])]

def ga_solver(Q_mat, n_users):
    num_genes = N_TRANS * n_users
    def fitness_func(ga_inst, sol, sol_idx):
        energy = np.dot(sol, np.dot(Q_mat, sol))
        x_m = sol.reshape(N_TRANS, n_users)
        penalty = 0 if np.all(np.sum(x_m, axis=0) == 1) else 1e6
        return -(energy + penalty)
    ga = pygad.GA(num_generations=100, num_parents_mating=10, fitness_func=fitness_func,
                  sol_per_pop=50, num_genes=num_genes, gene_space=[0, 1], stop_criteria="reach_0")
    ga.run()
    solution, _, _ = ga.best_solution()
    return solution.tolist()

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
    print(">>> RUNNING RIGOROUS V10 (10 SEEDS, N_T=6, CONNECTIVITY=1)...")
    N_SCALES = [4, 8, 12, 16]
    results = []
    for n_u in N_SCALES:
        print(f"Processing Scale: {n_u} Users...")
        r_q, r_opt, r_ga, r_g, r_t, t_q = [], [], [], [], [], []
        for s in range(10):
            G, Q = run_snapshot_mapping(n_u, seed=5000+s)
            # QIO
            rate_q, time_q, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda m: robust_qio_solver(m, n_u, 500))
            # Optimal Proxy
            rate_opt, _, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda m: robust_qio_solver(m, n_u, 5000))
            # GA
            rate_ga, _, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda m: ga_solver(m, n_u))
            # Greedy
            x_g = np.zeros((N_TRANS, n_u))
            for u in range(n_u): x_g[np.argmax(G[:, u]), u] = 1
            rate_g, _, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda m: x_g.flatten().tolist())
            # Truth
            rate_t = brute_force_strict(G, n_u) if n_u == 4 else 0
            
            r_q.append(rate_q); r_opt.append(rate_opt); r_ga.append(rate_ga); r_g.append(rate_g); r_t.append(rate_t); t_q.append(time_q)
        
        results.append({
            'Users': n_u, 'Qubits': n_u*N_TRANS,
            'Rate_Truth': np.mean(r_t) if n_u==4 else 0,
            'Rate_Opt': np.mean(r_opt), 'Rate_QIO': np.mean(r_q), 
            'Rate_GA': np.mean(r_ga), 'Rate_Greedy': np.mean(r_g),
            'Opt_Gap_Pct': (1 - np.mean(r_q)/np.mean(r_opt))*100 if np.mean(r_opt)>0 else 0,
            'Time_ms': np.mean(t_q)
        })
    df = pd.DataFrame(results)
    print("\n" + "="*80 + "\nFINAL RIGOROUS DATA FOR IEEE REVISION\n" + "="*80)
    print(df[['Users', 'Rate_Truth', 'Rate_Opt', 'Rate_QIO', 'Rate_GA', 'Rate_Greedy', 'Opt_Gap_Pct', 'Time_ms']])
    df.to_csv('final_rigorous_results_v10.csv', index=False)
