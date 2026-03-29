import numpy as np
import pandas as pd
import sys, os, neal, itertools

# Đảm bảo nhận diện đúng các module từ thư mục cha
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qubo_mapping import run_snapshot_mapping, N_TRANS
from classical_optimization import solve_qubo_and_calculate_rate, optimize_power_sca

def robust_qio_solver(Q_mat, n_users):
    bqm = {(i, j): Q_mat[i, j] for i in range(Q_mat.shape[0]) for j in range(i, Q_mat.shape[1])}
    sampler = neal.SimulatedAnnealingSampler()
    res = sampler.sample_qubo(bqm, num_reads=1000)
    # Tìm nghiệm hợp lệ 100%
    for s in res.samples():
        x = np.array([s[i] for i in range(Q_mat.shape[0])]).reshape(N_TRANS, n_users)
        if np.all(np.sum(x, axis=0) == 1):
            return [s[i] for i in range(Q_mat.shape[0])]
    return [res.first.sample[i] for i in range(Q_mat.shape[0])]

def brute_force_fair(G, n_u):
    """Tìm tối ưu toàn cục thực sự cho mạng nhỏ (phục vụ 100% user)"""
    if n_u > 5: return 0
    choices = list(itertools.product(range(N_TRANS), repeat=n_u))
    best_r = 0.0
    for c in choices:
        x = np.zeros((N_TRANS, n_u))
        for u, t in enumerate(c):
            x[t, u] = 1
        # optimize_power_sca trả về (p, rate) -> unpack 2 biến
        _, r = optimize_power_sca(G, x.flatten(), n_u)
        if r > best_r:
            best_r = r
    return best_r

if __name__ == "__main__":
    print(">>> RUNNING RIGOROUS REVISION (N_T=4, 10 SEEDS)...")
    N_SCALES = [4, 8, 12, 16]
    results = []

    for n_u in N_SCALES:
        print(f"\n>>> Scale: {n_u} Users...")
        r_q, r_g, r_t = [], [], []
        t_q = []

        for s in range(10):
            current_seed = 900 + s
            G, Q = run_snapshot_mapping(n_u, seed=current_seed)
            
            # 1. Proposed QIO - Trả về 3 giá trị
            rate_q, time_q, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda m: robust_qio_solver(m, n_u))
            
            # 2. Greedy Baseline (Ép chọn 1 link mạnh nhất) - Trả về 3 giá trị
            x_g = np.zeros((N_TRANS, n_u))
            for u in range(n_u): 
                x_g[np.argmax(G[:, u]), u] = 1
            rate_g, _, _ = solve_qubo_and_calculate_rate(Q, G, n_u, lambda m: x_g.flatten().tolist())
            
            # 3. Ground Truth (Chỉ cho n_u=4)
            rate_t = brute_force_fair(G, n_u) if n_u == 4 else 0
            
            r_q.append(rate_q); r_g.append(rate_g); r_t.append(rate_t)
            t_q.append(time_q)
            if (s+1)%2 == 0: print(f"   - Seed {s+1} hoàn thành.")

        mean_t = np.mean(r_t) if n_u == 4 else 0
        mean_q = np.mean(r_q)
        mean_g = np.mean(r_g)

        results.append({
            'Users': n_u, 
            'Qubits': n_u*N_TRANS,
            'Rate_Truth': mean_t if n_u==4 else 0,
            'Rate_QIO': mean_q, 
            'Rate_Greedy': mean_g,
            'Opt_Gap_Pct': ((mean_t - mean_q)/mean_t*100) if n_u==4 else 0,
            'Improvement_Pct': ((mean_q - mean_g)/mean_g*100),
            'Runtime_ms': np.mean(t_q)
        })

    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("RESULTS SUMMARY FOR IEEE WCL (FINAL VALIDATION)")
    print("="*80)
    print(df[['Users', 'Rate_Truth', 'Rate_QIO', 'Rate_Greedy', 'Opt_Gap_Pct', 'Improvement_Pct', 'Runtime_ms']])
    df.to_csv('final_rigorous_results_v8.csv', index=False)
