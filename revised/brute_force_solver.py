import numpy as np
import itertools
import time
import sys
import os

# Thêm thư mục cha vào đường dẫn để import các file cũ
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qubo_mapping import run_snapshot_mapping, N_TRANS
from classical_optimization import optimize_power_sca

def run_brute_force_true_optimum(N_USERS=5, seed=42):
    print(f"--- BẮT ĐẦU VÉT CẠN (BRUTE-FORCE) CHO {N_TRANS} TX, {N_USERS} USERS ---")
    
    # 1. Tạo ma trận Kênh truyền G (dùng chung seed với các file khác để công bằng)
    G_matrix, _ = run_snapshot_mapping(N_USERS, seed=seed)
    
    start_time = time.time()
    
    # 2. Tạo không gian nghiệm hợp lệ (Mỗi user chỉ nối max 1 Tx)
    # Lựa chọn: -1 (Không nối), 0, 1, 2, 3 (Nối với Tx tương ứng)
    user_choices = list(range(-1, N_TRANS))
    all_combinations = list(itertools.product(user_choices, repeat=N_USERS))
    total_cases = len(all_combinations)
    
    print(f"Tổng số trường hợp hợp lệ cần duyệt: {total_cases} (={len(user_choices)}^{N_USERS})")
    
    best_rate = 0.0
    best_X_flat = None
    
    # 3. Duyệt toàn bộ
    for idx, combo in enumerate(all_combinations):
        # In tiến độ cho đỡ chán
        if idx % 500 == 0:
            print(f" Đang chạy... {idx}/{total_cases}")
            
        X_current = np.zeros((N_TRANS, N_USERS))
        for u, t in enumerate(combo):
            if t != -1: # Nếu có kết nối
                X_current[t, u] = 1
                
        # Nếu không có link nào thì bỏ qua
        if np.sum(X_current) == 0:
            continue
            
        # Chuyển X thành vector 1D vì hàm optimize_power_sca của bạn yêu cầu thế
        x_vector = X_current.flatten()
        
        # Chạy Power Allocation (SCA) để lấy Sum Rate
        _, current_rate = optimize_power_sca(G_matrix, x_vector, N_USERS)
        
        if current_rate > best_rate:
            best_rate = current_rate
            best_X_flat = x_vector

    end_time = time.time()
    
    print("-" * 50)
    print(f"HOÀN THÀNH BRUTE-FORCE!")
    print(f"Thời gian chạy: {end_time - start_time:.2f} giây")
    print(f"TRUE GLOBAL OPTIMUM RATE: {best_rate:.5f} nats/s/Hz")
    print(f"Ma trận Link tốt nhất (X):\n{best_X_flat.reshape(N_TRANS, N_USERS)}")
    print("-" * 50)
    
    return best_rate

if __name__ == "__main__":
    # Bạn hãy chạy file này, lấy được số liệu Global Optimum.
    # Sau đó so sánh với Sum Rate 1.629 (của QIO) xem đạt bao nhiêu % nhé!
    run_brute_force_true_optimum(N_USERS=5, seed=42)
