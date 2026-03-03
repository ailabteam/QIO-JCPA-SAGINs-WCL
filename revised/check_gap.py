import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import neal
from qubo_mapping import run_snapshot_mapping, N_TRANS
from classical_optimization import solve_qubo_and_calculate_rate
from brute_force_solver import run_brute_force_true_optimum

def run_gap_check():
    N_USERS = 5
    SEED = 42
    
    print("1. ĐANG TÍNH TRUE GLOBAL OPTIMUM (BRUTE-FORCE)...")
    opt_rate = run_brute_force_true_optimum(N_USERS=N_USERS, seed=SEED)
    
    print("\n2. ĐANG TÍNH QIO VÀ GREEDY...")
    G, Q = run_snapshot_mapping(N_USERS, seed=SEED)
    
    # Hàm QIO (Neal)
    def qih_neal_solver(Q_mat):
        bqm = {(i, j): Q_mat[i, j] for i in range(Q_mat.shape[0]) for j in range(i, Q_mat.shape[1])}
        sampler = neal.SimulatedAnnealingSampler()
        return [sampler.sample_qubo(bqm, num_reads=1000).first.sample[i] for i in range(Q_mat.shape[0])]

    # Hàm Greedy
    def greedy_solver(Q_mat):
        x_gry = np.zeros((N_TRANS, N_USERS), dtype=int)
        for u in range(N_USERS):
            t_best = np.argmax(G[:, u])
            x_gry[t_best, u] = 1
        return x_gry.flatten().tolist()

    qio_rate, _, _ = solve_qubo_and_calculate_rate(Q, G, N_USERS, qih_neal_solver)
    gry_rate, _, _ = solve_qubo_and_calculate_rate(Q, G, N_USERS, greedy_solver)
    
    print("\n" + "="*50)
    print("TỔNG KẾT OPTIMALITY GAP (KHOẢNG CÁCH TỐI ƯU)")
    print("="*50)
    print(f"- True Global Optimum : {opt_rate:.5f} nats/s/Hz (100.0%)")
    print(f"- QIO-JLSPA (Proposed): {qio_rate:.5f} nats/s/Hz ({qio_rate/opt_rate*100:.2f}%)")
    print(f"- Greedy Heuristic    : {gry_rate:.5f} nats/s/Hz ({gry_rate/opt_rate*100:.2f}%)")
    print("="*50)

if __name__ == "__main__":
    run_gap_check()
