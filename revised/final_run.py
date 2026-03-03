import numpy as np
import itertools
import time
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import neal
from qubo_mapping import run_snapshot_mapping, N_TRANS
from classical_optimization import solve_qubo_and_calculate_rate

def robust_qih_solver(Q_mat, n_users, num_reads=1000):
    """Giải QUBO và LỌC NGHIỆM ĐỂ ĐẢM BẢO KHÔNG VI PHẠM RÀNG BUỘC"""
    bqm = {(i, j): Q_mat[i, j] for i in range(Q_mat.shape[0]) for j in range(i, Q_mat.shape[1])}
    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample_qubo(bqm, num_reads=num_reads)

    # Tìm nghiệm hợp lệ đầu tiên (Mỗi user đúng 1 link)
    for sample in response.samples():
        x_vec = [sample[i] for i in range(Q_mat.shape[0])]
        x_mat = np.array(x_vec).reshape(N_TRANS, n_users)
        # Kiểm tra constraint: Mỗi cột (user) có đúng 1 số 1
        if np.all(np.sum(x_mat, axis=0) == 1):
            return x_vec

    # Nếu xui xẻo không có nghiệm nào hợp lệ (rất hiếm), trả về nghiệm tốt nhất
    return [response.first.sample[i] for i in range(Q_mat.shape[0])]

def greedy_solver_fixed(G_matrix, n_users):
    """Thuật toán tham lam: Mỗi user chọn Tx mạnh nhất"""
    x_gry = np.zeros((N_TRANS, n_users), dtype=int)
    for u in range(n_users):
        t_best = np.argmax(G_matrix[:, u])
        x_gry[t_best, u] = 1
    return x_gry.flatten().tolist()

def run_brute_force_fixed(G_matrix, n_users):
    """Vét cạn Tối ưu (Đã sửa để ép buộc mỗi user CÓ ĐÚNG 1 LINK)"""
    # Lựa chọn chỉ từ 0 đến N_TRANS-1 (Bỏ lựa chọn -1)
    user_choices = list(range(0, N_TRANS))
    all_combinations = list(itertools.product(user_choices, repeat=n_users))

    best_rate = 0.0
    from classical_optimization import optimize_power_sca

    for combo in all_combinations:
        X_current = np.zeros((N_TRANS, n_users))
        for u, t in enumerate(combo):
            X_current[t, u] = 1
        x_vector = X_current.flatten()
        _, rate = optimize_power_sca(G_matrix, x_vector, n_users)
        if rate > best_rate:
            best_rate = rate
    return best_rate

if __name__ == "__main__":
    print("\n" + "="*50)
    print("PHẦN 1: TÍNH OPTIMALITY GAP (MẠNG 4 TX, 5 USERS)")
    print("="*50)
    N_USERS_GAP = 5
    SEED = 42
    G, Q = run_snapshot_mapping(N_USERS_GAP, seed=SEED)

    print("Đang vét cạn (True Global Optimum)...")
    opt_rate = run_brute_force_fixed(G, N_USERS_GAP)

    print("Đang chạy QIO-JLSPA...")
    qio_rate, qio_time, _ = solve_qubo_and_calculate_rate(Q, G, N_USERS_GAP, lambda q: robust_qih_solver(q, N_USERS_GAP, 1000))

    print("Đang chạy Greedy...")
    gry_rate, gry_time, _ = solve_qubo_and_calculate_rate(Q, G, N_USERS_GAP, lambda q: greedy_solver_fixed(G, N_USERS_GAP))

    print("\n--- KẾT QUẢ GAP ---")
    print(f"True Global Optimum : {opt_rate:.5f} (100.0%)")
    print(f"QIO-JLSPA (Proposed): {qio_rate:.5f} ({qio_rate/opt_rate*100:.2f}%) - Time: {qio_time:.1f} ms")
    print(f"Greedy Heuristic    : {gry_rate:.5f} ({gry_rate/opt_rate*100:.2f}%) - Time: {gry_time:.1f} ms")


    print("\n" + "="*50)
    print("PHẦN 2: CHẠY SCALE-UP (ĐÁNH BẠI GREEDY Ở QUY MÔ LỚN)")
    print("="*50)

    # Scale lên lớn hơn để chứng minh sự khác biệt
    N_SCALES = [8, 12, 16]
    for n_u in N_SCALES:
        print(f"\n--- Đang chạy {n_u} Users ({n_u * N_TRANS} Qubits) ---")
        g, q = run_snapshot_mapping(n_u, seed=42)

        rate_qio, t_qio, _ = solve_qubo_and_calculate_rate(q, g, n_u, lambda mat: robust_qih_solver(mat, n_u, 1000))
        rate_gry, t_gry, _ = solve_qubo_and_calculate_rate(q, g, n_u, lambda mat: greedy_solver_fixed(g, n_u))

        print(f"  QIO    Rate: {rate_qio:.3f} | Time: {t_qio:.1f} ms")
        print(f"  Greedy Rate: {rate_gry:.3f} | Time: {t_gry:.1f} ms")
		print(f" -> QIO Sum-Rate Improvement: {((rate_qio - rate_gry)/rate_gry)*100:+.2f}%")


    print("\nHOÀN THÀNH TẤT CẢ MÔ PHỎNG CHO BÀI BÁO!")
