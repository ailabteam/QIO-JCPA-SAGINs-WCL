import numpy as np
import pandas as pd
import neal, itertools, time
from qubo_mapping import run_snapshot_mapping, N_TRANS
from classical_optimization import solve_link_selection_and_get_rate

def qio_solver(Q_mat, n_users):
    bqm = {(i, j): Q_mat[i, j] for i in range(Q_mat.shape[0]) for j in range(i, Q_mat.shape[1])}
    res = neal.SimulatedAnnealingSampler().sample_qubo(bqm, num_reads=1000)
    for s in res.samples():
        x = np.array([s[i] for i in range(Q_mat.shape[0])]).reshape(N_TRANS, n_users)
        if np.all(np.sum(x, axis=0) == 1): return [s[i] for i in range(Q_mat.shape[0])]
    return [res.first.sample[i] for i in range(Q_mat.shape[0])]

def greedy_solver(G, n_users):
    x = np.zeros((N_TRANS, n_users))
    for u in range(n_users): x[np.argmax(G[:, u]), u] = 1
    return x.flatten().tolist()

def brute_force_truth(G, n_u):
    if n_u > 5: return 0
    choices = list(itertools.product(range(N_TRANS), repeat=n_u))
    best_r = 0
    for c in choices:
        x = np.zeros((N_TRANS, n_u))
        for u, t in enumerate(c): x[t, u] = 1
        r = solve_link_selection_and_get_rate(x.flatten(), G, n_u)
        if r > best_r: best_r = r
    return best_r

if __name__ == "__main__":
    print(">>> RUNNING RIGOROUS V11 (UNIFORM POWER, N_T=6)...")
    N_SCALES = [4, 8, 12, 16]
    results = []
    for n_u in N_SCALES:
        print(f"Processing {n_u} Users...")
        r_q, r_g, r_t, t_q = [], [], [], []
        for s in range(10):
            G, Q = run_snapshot_mapping(n_u, seed=9000+s)
            start = time.time()
            x_q = qio_solver(Q, n_u)
            t_q.append((time.time()-start)*1000)
            
            rate_q = solve_link_selection_and_get_rate(x_q, G, n_u)
            rate_g = solve_link_selection_and_get_rate(greedy_solver(G, n_u), G, n_u)
            rate_t = brute_force_truth(G, n_u) if n_u == 4 else 0
            
            r_q.append(rate_q); r_g.append(rate_g); r_t.append(rate_t)
        
        results.append({
            'Users': n_u, 'Rate_Truth': np.mean(r_t), 'Rate_QIO': np.mean(r_q),
            'Rate_Greedy': np.mean(r_g), 'Gain_vs_Greedy_%': (np.mean(r_q)/np.mean(r_g)-1)*100,
            'Gap_vs_Truth_%': (1-np.mean(r_q)/np.mean(r_t))*100 if n_u==4 else 0,
            'Time_ms': np.mean(t_q)
        })

    print("\n" + "="*80 + "\nDATA FOR SUBMISSION (LOGICAL & RIGOROUS)\n" + "="*80)
    print(pd.DataFrame(results))
