# ==============================================================================
# File: qubo_mapping.py
# Mô tả: Thiết lập mô hình NTN cơ sở và xây dựng ma trận QUBO Q.
# FIX: Loại bỏ biến global N_USERS/N_QUBITS; sử dụng tham số rõ ràng.
# ==============================================================================
import numpy as np
import itertools
import pandas as pd
import sys

# --- Hằng số Toàn cục (Fixed Constants) ---
C_CHANNELS = 1      
N_TRANS = 4         
FREQUENCY_GHZ = 28.0 
LAMBDA_CONSTRAINT = 50000.0 
LAMBDA_BENEFIT = 1.0       


# --- 1. Hàm Tiện ích ---

def map_index_to_vars(i, N_USERS):
    """Ánh xạ chỉ số Qubit i thành (t, u, c=0), dùng N_USERS hiện tại."""
    N_QUBITS = N_TRANS * N_USERS * C_CHANNELS
    if i >= N_QUBITS:
        raise ValueError("Index out of bounds")
        
    c = i % C_CHANNELS 
    temp = i // C_CHANNELS 
    u = temp % N_USERS
    t = temp // N_USERS
    return t, u, c

# map_vars_to_index không cần thiết cho quá trình xây dựng QUBO

# --- 2. Mô hình Kênh Cơ sở (Tạo Channel Gain G_t,u) ---
def generate_channel_gains(N_USERS, seed=42):
    """Tạo ma trận Channel Gain G[N_TRANS, N_USERS]."""
    np.random.seed(seed)
    G = np.zeros((N_TRANS, N_USERS))
    
    # ... (Logic Path Loss/Fading giữ nguyên) ...
    D_LEO = 600 
    D_HAPS = 20
    
    def path_loss_db(D):
        return 20 * np.log10(D) + 20 * np.log10(FREQUENCY_GHZ) + 20 

    for t in range(N_TRANS):
        for u in range(N_USERS):
            if t < N_TRANS - 1: 
                D = D_LEO * np.random.uniform(0.9, 1.1)
                PL_dB = path_loss_db(D)
            else: 
                D = D_HAPS * np.random.uniform(0.8, 1.2)
                PL_dB = path_loss_db(D)
            
            PL_linear = 10**(-PL_dB / 10)
            
            K = 10.0
            sigma = 1.0 / np.sqrt(K + 1)
            mu = np.sqrt(K / (K + 1))
            
            rayleigh = np.random.randn() * sigma + 1j * np.random.randn() * sigma
            los = mu + 0j
            fading_gain = np.abs(los + rayleigh)**2
            
            G[t, u] = PL_linear * fading_gain
    
    G_max = np.max(G)
    G = G / G_max
    
    return G

# --- 3. Xây dựng Ma trận QUBO Q (N_QUBITS x N_QUBITS) ---
def build_qubo_matrix(G, N_USERS):
    """Xây dựng ma trận QUBO Q, sử dụng N_USERS từ G."""
    N_QUBITS = N_TRANS * N_USERS * C_CHANNELS
    Q = np.zeros((N_QUBITS, N_QUBITS))
    
    for i in range(N_QUBITS):
        for j in range(i, N_QUBITS):
            # Lấy t, u, c từ chỉ số i, j
            t_i, u_i, c_i = map_index_to_vars(i, N_USERS)
            t_j, u_j, c_j = map_index_to_vars(j, N_USERS)
            
            # --- 3.1 Thành phần Bậc hai (i != j) ---
            if i != j:
                
                # 1. Ràng buộc C1: Phạt nếu 2 liên kết phục vụ cùng 1 người dùng (u_i == u_j)
                if u_i == u_j:
                    Q[i, j] += 2 * LAMBDA_CONSTRAINT 
                    
                # 2. Chi phí Lợi ích: Phạt nếu 2 liên kết đồng kênh gây nhiễu
                if u_i != u_j: # Chỉ tính nhiễu giữa các người dùng khác nhau
                    interference_i_to_j = G[t_i, u_j] 
                    interference_j_to_i = G[t_j, u_i] 
                    
                    Q[i, j] += -LAMBDA_BENEFIT * (interference_i_to_j + interference_j_to_i)

            # --- 3.2 Thành phần Bậc nhất (i = j) ---
            else: # i == j
                # Term Ràng buộc C1: -1 * lambda_A * x_i
                Q[i, i] += -1 * LAMBDA_CONSTRAINT 
                
    Q = Q + Q.T - np.diag(Q.diagonal())
    
    return Q

# --- 4. Hàm Chạy Snapshot (Hàm chính được gọi) ---
def run_snapshot_mapping(n_users, seed):
    """
    Tạo G và Q cho quy mô n_users cụ thể.
    """
    if n_users <= 0:
        raise ValueError("N_USERS must be greater than 0")
        
    G_matrix = generate_channel_gains(n_users, seed=seed)
    Q_matrix = build_qubo_matrix(G_matrix, n_users)
    
    # Lưu G cho benchmark_solver
    np.save('channel_gain_G_4x5.npy', G_matrix) # Tên này sẽ bị ghi đè mỗi lần
    
    return G_matrix, Q_matrix

# --- 5. Thực thi chính (Chỉ để kiểm tra/tạo file mẫu) ---
if __name__ == "__main__":
    N_USERS_TEST = 5 # Chạy với 5 users để kiểm tra
    G_test, Q_test = run_snapshot_mapping(N_USERS_TEST, seed=42)
    N_QUBITS_TEST = N_TRANS * N_USERS_TEST * C_CHANNELS
    
    Q_FILENAME = f'qubo_matrix_Q_{N_QUBITS_TEST}x{N_QUBITS_TEST}.npy'
    np.save(Q_FILENAME, Q_test)
    
    print(f"Kiểm tra {N_USERS_TEST} Users -> {N_QUBITS_TEST} Qubits.")
    print(f"Ma trận Q đã lưu vào '{Q_FILENAME}'")
