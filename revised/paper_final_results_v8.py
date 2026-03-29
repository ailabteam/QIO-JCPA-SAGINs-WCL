import numpy as np
import pandas as pd
import sys, os, neal, itertools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qubo_mapping import run_snapshot_mapping, N_TRANS
from classical_optimization import solve_qubo_and_calculate_rate, optimize_power_sca

def robust_qio_solver(Q_mat, n_users):
    bqm = {(i, j): Q_mat[i, j] for i in range(Q_mat.shape[0]) for j in range(i, Q_mat.shape[1])}
    sampler = neal.SimulatedAnnealingSampler()
    res = sampler.sample_qubo(bqm, num_reads=1000)
    for s in res.samples():
        x = np.array([s[i] for i in range(Q_mat.shape[0])]).reshape(N_TRANS, n_users)
        if np.all(np.sum(x, axis=0) == 1): return x.flatten().tolist()
    return np.array([res.first.sample[i] for i in range(Q_mat.shape[0])]).tolist()

def brute_force_fair(G, n_u):
    if n_u > 5: return 0
    choices = list(itertools.product(range(N_TRANS), repeat=n_u))
    best_r = 0
    for c in choices:
        x = np.zeros((N_TRANS, n_u))
        for u, t in enumerate(c): x[t, u] = 1
        _, r = optimize_power_sca(G, x.flatten(), n_u)
        if r > best_r: best_r = r
    return best_r

if __name__ == "__main__":
    print(">>> RUNNING RIGOROUS REVISION (N_T=4, 10 SEEDS)...")
    N_SCALES = [4, 8, 12, 16]
    results = []
    for n_u in N_SCALES:
        print(f"Scale: {n_u} Users...")
        r_q, r_g, r_t = [], [], []
        for s in range(10):
            G, Q = run_snapshot_mapping(n_u, seed=900+s)
            rate_q, time_q, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda m: robust_qio_solver(m, n_u))
            # Greedy: Mỗi user chọn Link mạnh nhất
            x_g = np.zeros((N_TRANS, n_u))
            for u in range(n_u): x_g[np.argmax(G[:, u]), u] = 1
            _, rate_g = solve_qubo_and_calculate_rate(Q, G, n_u, lambda m: x_g.flatten().tolist())
            # Truth
            rate_t = brute_force_fair(G, n_u) if n_u == 4 else 0
            
            r_q.append(rate_q); r_g.append(rate_g); r_t.append(rate_t)
        
        results.append({
            'Users': n_u, 'Qubits': n_u*N_TRANS,
            'Rate_Truth': np.mean(r_t) if n_u==4 else 0,
            'Rate_QIO': np.mean(r_q), 
            'Rate_Greedy': np.mean(r_g),
            'Opt_Gap_Pct': ((np.mean(r_t)-np.mean(r_q))/np.mean(r_t)*100) if n_u==4 else 0,
            'Improvement_Pct': (np.mean(r_q)-np.mean(r_g))/np.mean(r_g)*100,
            'Runtime_ms': np.mean(time_q)
        })

    df = pd.DataFrame(results)
    print("\n" + "="*80 + "\nRESULTS SUMMARY FOR IEEE WCL\n" + "="*80)
    print(df[['Users', 'Rate_Truth', 'Rate_QIO', 'Rate_Greedy', 'Opt_Gap_Pct', 'Improvement_Pct', 'Runtime_ms']])
    df.to_csv('final_rigorous_results_v8.csv', index=False)
