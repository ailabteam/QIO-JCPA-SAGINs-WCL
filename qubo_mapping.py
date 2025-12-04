# ==============================================================================
# File: qubo_mapping.py
# Mô tả: Thiết lập mô hình NTN cơ sở và xây dựng ma trận QUBO Q (20x20)
# Fix: Lỗi mã hóa Term Bậc nhất của ràng buộc C1.
# ==============================================================================
import numpy as np
import itertools
import pandas as pd
import sys

# --- 1. Tham số Hệ thống (4 Transmitters, 5 Users, 1 Channel) ---
C_CHANNELS = 1      # Số kênh tần số (Co-Channel Interference)
N_USERS = 5         # Số người dùng (UEs)
N_TRANS = 4         # Số bộ truyền (3 LEOs, 1 HAPS)
N_QUBITS = N_TRANS * N_USERS * C_CHANNELS # 4 * 5 * 1 = 20 Qubits

# Tham số Kênh Vô tuyến
FREQUENCY_GHZ = 28.0 
LIGHT_SPEED = 3e8

# Tham số QUBO
LAMBDA_CONSTRAINT = 50000.0 # Hệ số phạt TĂNG MẠNH
LAMBDA_BENEFIT = 1.0       

# --- 2. Hàm Tiện ích ---

def map_index_to_vars(i):
    """Ánh xạ chỉ số Qubit i (0 đến 19) thành (t, u, c=0)"""
    # Vì C_CHANNELS = 1, c luôn bằng 0
    c = i % C_CHANNELS 
    temp = i // C_CHANNELS # temp = t * N_USERS + u
    u = temp % N_USERS
    t = temp // N_USERS
    return t, u, c

def map_vars_to_index(t, u, c):
    """Ánh xạ (t, u, c) thành chỉ số Qubit i"""
    return (t * N_USERS + u) * C_CHANNELS + c

# --- 3. Mô hình Kênh Cơ sở (Tạo Channel Gain G_t,u) ---
def generate_channel_gains(seed=42):
    """
    Tạo ma trận Channel Gain G[t, u].
    Tái sử dụng mô hình Rician Fading + Path Loss chuẩn hóa.
    """
    np.random.seed(seed)
    G = np.zeros((N_TRANS, N_USERS))
    
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
    
    # Chuẩn hóa
    G_max = np.max(G)
    G = G / G_max
    
    return G

# --- 4. Xây dựng Ma trận QUBO Q (N_QUBITS x N_QUBITS) ---
def build_qubo_matrix(G):
    Q = np.zeros((N_QUBITS, N_QUBITS))
    
    for i in range(N_QUBITS):
        for j in range(i, N_QUBITS):
            t_i, u_i, c_i = map_index_to_vars(i)
            t_j, u_j, c_j = map_index_to_vars(j)
            
            # --- 4.1 Thành phần Bậc hai (i != j) ---
            if i != j:
                
                # 1. Ràng buộc C1: Phạt nếu 2 liên kết phục vụ cùng 1 người dùng (u_i == u_j)
                if u_i == u_j:
                    # Hệ số 2 * lambda_A cho x_i * x_j (vì (x_i + x_j - 1)^2)
                    Q[i, j] += 2 * LAMBDA_CONSTRAINT 
                    
                # 2. Chi phí Lợi ích: Phạt nếu 2 liên kết đồng kênh (c_i == c_j) gây nhiễu
                # Vì C_CHANNELS=1, c_i luôn bằng c_j
                if u_i != u_j:
                    interference_i_to_j = G[t_i, u_j] 
                    interference_j_to_i = G[t_j, u_i] 
                    
                    # Cost = - LAMBDA_BENEFIT * (Tổng Nhiễu Tiềm năng)
                    Q[i, j] += -LAMBDA_BENEFIT * (interference_i_to_j + interference_j_to_i)

            # --- 4.2 Thành phần Bậc nhất (i = j) ---
            else: # i == j
                # Term Ràng buộc C1: (x_i - 1)^2 -> -1 * lambda_A * x_i (khi khai triển)
                # FIX LỖI: Sử dụng -1 * LAMBDA_CONSTRAINT
                Q[i, i] += -1 * LAMBDA_CONSTRAINT 
                
    # Đảm bảo tính đối xứng
    Q = Q + Q.T - np.diag(Q.diagonal())
    
    return Q

# --- 5. Thực thi và Xuất kết quả ---
if __name__ == "__main__":
    
    # Lấy lại Channel Gain G để đảm bảo nhất quán với benchmark_solver
    G_matrix = generate_channel_gains(seed=42)
    
    # Xây dựng Ma trận QUBO
    Q_matrix = build_qubo_matrix(G_matrix)
    
    Q_FILENAME = f'qubo_matrix_Q_{N_QUBITS}x{N_QUBITS}.npy'
    
    # Lưu ma trận Q vào file (quan trọng cho bước tiếp theo)
    np.save(Q_FILENAME, Q_matrix)
    
    # Lưu ma trận G (Để tránh phải tính lại trong benchmark_solver)
    np.save('channel_gain_G_4x5.npy', G_matrix)
    
    print(f"Bắt đầu thiết kế QUBO cho {N_QUBITS} Qubits...")
    print(f"--- Channel Gain Matrix G (Normalized) ---")
    print(G_matrix)
    print(f"\nMa trận QUBO Q đã được lưu vào '{Q_FILENAME}'")
    
    # (Tùy chọn) In một phần ma trận để kiểm tra
    df_Q = pd.DataFrame(Q_matrix).head(10).T.head(10)
    print("\n--- 10x10 Phần đầu của Ma trận Q ---")
    print(df_Q)
