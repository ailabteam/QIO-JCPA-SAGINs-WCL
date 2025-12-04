# ==============================================================================
# File: classical_optimization.py
# Mô tả: Chứa hàm tối ưu hóa công suất (Step 2) cho bất kỳ vector x (Link Selection) nào.
# FIX: Sử dụng N_USERS_ACTUAL làm tham số thay vì hằng số global.
# ==============================================================================
import numpy as np
import scipy.optimize
import sys
import time # <--- Thêm dòng này

# Import các hằng số cố định
from qubo_mapping import N_TRANS, C_CHANNELS, LAMBDA_CONSTRAINT

# --- CẤU HÌNH HỆ THỐNG CỐ ĐỊNH ---
P_MAX_T = 1.0  # Công suất truyền tối đa (Normalization: 1 Watt)
NOISE_POWER = 1e-4 # Mức nhiễu nền (sigma^2)
MAX_ITER_PA = 20 # Số lần lặp tối đa cho Power Allocation (SCA)
CONVERGENCE_TOL = 1e-3 # Ngưỡng hội tụ


# --- 1. Hàm Mục tiêu và Ràng buộc (Max Sum Rate) ---

def calculate_sinr(p, G, x, t_i, u_i, N_USERS_ACTUAL):
    """Tính SINR cho liên kết (t_i, u_i)"""
    
    signal = G[t_i, u_i] * p[t_i, u_i]
    
    interference = 0.0
    for t_prime in range(N_TRANS):
        for u_prime in range(N_USERS_ACTUAL):
            # Cần kiểm tra x[t', u']
            
            # X là ma trận 4xN_USERS_ACTUAL
            if x[t_prime, u_prime] == 1: 
                # Liên kết (t_prime, u_prime) gây nhiễu cho người dùng u_i
                if (t_prime != t_i) or (u_prime != u_i):
                     # G[t', u_i] là gain từ máy phát t' đến máy thu u_i
                     interference += G[t_prime, u_i] * p[t_prime, u_prime]
                     
    return signal / (NOISE_POWER + interference)

def calculate_sum_rate(p, G, x, N_USERS_ACTUAL):
    """Tính tổng Sum Rate của tất cả các liên kết được chọn (x=1)"""
    
    total_rate = 0.0
    
    for t in range(N_TRANS):
        for u in range(N_USERS_ACTUAL):
            if x[t, u] == 1:
                sinr = calculate_sinr(p, G, x, t, u, N_USERS_ACTUAL)
                total_rate += np.log2(1 + sinr)
                
    return total_rate, np.sum(x)


# --- 2. Thuật toán Tối ưu hóa Công suất (SCA-Based) ---

def optimize_power_sca(G, x_vector, N_USERS_ACTUAL):
    """
    Tối ưu hóa công suất P sử dụng thuật toán lặp (SCA)
    với vector Link Selection x đã cố định.
    """
    
    # Biến x phải được reshape thành N_TRANS x N_USERS_ACTUAL
    x = x_vector.reshape(N_TRANS, N_USERS_ACTUAL)
    N_TOTAL_VARS = N_TRANS * N_USERS_ACTUAL
    
    # Khởi tạo công suất p
    p = np.zeros((N_TRANS, N_USERS_ACTUAL))
    num_active_links = np.sum(x)
    
    if num_active_links > 0:
        # Khởi tạo công suất đều cho các liên kết hoạt động
        p = p + (x * (P_MAX_T / N_TRANS)) 
    else:
        return p, 0.0
        
    prev_rate = 0.0
    
    for iter_count in range(MAX_ITER_PA):
        p_old = p.copy()
        
        def objective_to_minimize(p_flat):
            p_temp = p_flat.reshape(N_TRANS, N_USERS_ACTUAL)
            if np.any(p_temp < -1e-6): return 1e10
            rate, _ = calculate_sum_rate(p_temp, G, x, N_USERS_ACTUAL)
            return -rate # Minimize negative rate

        # Ràng buộc công suất (0 <= p <= P_MAX_T)
        bounds = [(0, P_MAX_T) for _ in range(N_TOTAL_VARS)]
        
        # Ràng buộc công suất tổng P_max trên mỗi Transmitter
        constraints = [{'type': 'ineq', 'fun': 
                        lambda p_flat, t=t: P_MAX_T - np.sum(p_flat.reshape(N_TRANS, N_USERS_ACTUAL)[t, :])} 
                       for t in range(N_TRANS)]


        p_init = p_old.flatten()
        
        result = scipy.optimize.minimize(
            objective_to_minimize, 
            p_init, 
            method='SLSQP', 
            bounds=bounds,
            constraints=constraints,
            tol=CONVERGENCE_TOL
        )
        
        p = result.x.reshape(N_TRANS, N_USERS_ACTUAL)
        
        # Chỉ giữ công suất cho các liên kết đã được chọn
        p = p * x
        
        current_rate, _ = calculate_sum_rate(p, G, x, N_USERS_ACTUAL)
        
        if np.abs(current_rate - prev_rate) < CONVERGENCE_TOL:
            break
        
        prev_rate = current_rate
    
    final_rate, _ = calculate_sum_rate(p, G, x, N_USERS_ACTUAL)
    return p, final_rate

# --- 3. Hàm Giải QUBO và Tính Sum Rate (Wrapper) ---

def map_qubo_solution_to_x(qubo_solution, N_USERS_ACTUAL):
    """Chuyển đổi vector QUBO sang ma trận x (N_TRANS x N_USERS_ACTUAL)"""
    x_vector = np.array(qubo_solution)
    x = x_vector.reshape(N_TRANS, N_USERS_ACTUAL)
    return x.flatten() 

def solve_qubo_and_calculate_rate(Q, G, N_USERS_ACTUAL, solver_func):
    """
    Thực hiện Step 1 (Giải QUBO) và Step 2 (Tối ưu hóa Công suất)
    """
    
    N_QUBITS_ACTUAL = N_TRANS * N_USERS_ACTUAL * C_CHANNELS
    
    # Step 1: Giải bài toán Tổ hợp
    start_time_comb = time.time()
    # solver_func chỉ cần nhận ma trận Q
    qubo_solution = solver_func(Q) 
    runtime_comb = (time.time() - start_time_comb) * 1000 # ms

    x_vector = map_qubo_solution_to_x(qubo_solution, N_USERS_ACTUAL)
    
    # Kiểm tra ràng buộc cơ bản (Mỗi người dùng chỉ được 1 link)
    if np.sum(x_vector.reshape(N_TRANS, N_USERS_ACTUAL), axis=0).max() > 1:
        print("Warning: QIH solution violates fundamental assignment constraint.")
        final_rate = 0.0
        runtime_total = (time.time() - start_time_comb) * 1000
        return final_rate, runtime_total, x_vector

    # Step 2: Tối ưu hóa Công suất
    start_time_pa = time.time()
    _, final_rate = optimize_power_sca(G, x_vector, N_USERS_ACTUAL)
    runtime_pa = (time.time() - start_time_pa) * 1000 # ms
    
    runtime_total = runtime_comb + runtime_pa
    
    return final_rate, runtime_total, x_vector


if __name__ == '__main__':
    # Chỉ dùng để kiểm tra cấu trúc
    print("Classical Optimization Module loaded.")
