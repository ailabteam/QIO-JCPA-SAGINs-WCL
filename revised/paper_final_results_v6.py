import numpy as np
import pandas as pd
import sys, os, neal, itertools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qubo_mapping import run_snapshot_mapping, N_TRANS
from classical_optimization import solve_qubo_and_calculate_rate, optimize_power_sca

def robust_qio_solver(Q_mat, n_users, reads=1000):
    bqm = {(i, j): Q_mat[i, j] for i in range(Q_mat.shape[0]) for j in range(i, Q_mat.shape[1])}
    sampler = neal.SimulatedAnnealingSampler()
    res = sampler.sample_qubo(bqm, num_reads=reads)
    for s in res.samples():
        x = np.array([s[i] for i in range(Q_mat.shape[0])]).reshape(N_TRANS, n_users)
        if np.all(np.sum(x, axis=0) == 1): return x.flatten().tolist()
    return np.array([res.first.sample[i] for i in range(Q_mat.shape[0])]).tolist()

def brute_force_strict(G, n_u):
    if n_u > 4: return 0
    choices = list(itertools.product(range(N_TRANS), repeat=n_u))
    best_r = 0
    for c in choices:
        x = np.zeros((N_TRANS, n_u)); 
        for u, t in enumerate(c): x[t, u] = 1
        _, r = optimize_power_sca(G, x.flatten(), n_u)
        if r > best_r: best_r = r
    return best_r

if __name__ == "__main__":
    print(">>> RUNNING RIGOROUS IEEE REVISION EVALUATION (10 SEEDS)...")
    N_SCALES = [4, 8, 12, 16]
    results = []
    for n_u in N_SCALES:
        print(f"\nScale: {n_u} Users...")
        r_q, r_g, r_t = [], [], []
        for s in range(10):
            G, Q = run_snapshot_mapping(n_u, seed=700+s)
            rate_q, time_q, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda m: robust_qio_solver(m, n_u))
            rate_g, _, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda m: (np.eye(N_TRANS)[np.argmax(G, axis=0)].T).flatten().tolist())
            rate_t = brute_force_strict(G, n_u) if n_u == 4 else 0
            r_q.append(rate_q); r_g.append(rate_g); r_t.append(rate_t)
        
        results.append({
            'Users': n_u, 'Qubits': n_u*N_TRANS,
            'Rate_Truth': np.mean(r_t) if n_u==4 else "N/A",
            'Rate_QIO': np.mean(r_q), 'Rate_Greedy': np.mean(r_g),
            'Improvement': (np.mean(r_q)-np.mean(r_g))/np.mean(r_g)*100,
            'Runtime_ms': np.mean(time_q)
        })
        print(f"   Done. Improvement: {results[-1]['Improvement']:.2f}%")

    df = pd.DataFrame(results)
    print("\n" + "="*60 + "\nFINAL RESULTS TABLE\n" + "="*60)
    print(df)
    df.to_csv('rigorous_final_results.csv', index=False)
