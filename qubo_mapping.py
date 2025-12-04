# ==============================================================================
# File: qubo_mapping.py
# Mô tả: Thiết lập mô hình NTN cơ sở và xây dựng ma trận QUBO Q (32x32)
# ==============================================================================
import numpy as np
import itertools
import pandas as pd # Dùng để in ma trận Q dễ nhìn hơn (tùy chọn)

# --- 1. Tham số Hệ thống (4 Transmitters, 4 Users, 2 Channels) ---
C_CHANNELS = 2  # Số kênh tần số
N_USERS = 4     # Số người dùng (UEs)
N_TRANS = 4     # Số bộ truyền (3 LEOs, 1 HAPS)
N_QUBITS = N_TRANS * N_USERS * C_CHANNELS # 4 * 4 * 2 = 32 Qubits

# Tham số Kênh Vô tuyến
FREQUENCY_GHZ = 28.0 # Tần số mmWave/Ka band
LIGHT_SPEED = 3e8

# Tham số QUBO
LAMBDA_CONSTRAINT = 1000.0 # Hệ số phạt cho ràng buộc (phải đủ lớn)
LAMBDA_BENEFIT = 1.0       # Hệ số cho chi phí lợi ích

# --- 2. Hàm Tiện ích ---

def map_index_to_vars(i):
    """Ánh xạ chỉ số Qubit i (0 đến 31) thành (t, u, c)"""
    c = i % C_CHANNELS
    temp = i // C_CHANNELS
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
    Để đơn giản hóa, chúng ta dùng mô hình Path Loss + Rician Fading.
    Giả định: Tất cả công suất truyền Ptx = 1.
    """
    np.random.seed(seed)
    G = np.zeros((N_TRANS, N_USERS))
    
    # Giả định vị trí cơ bản và khoảng cách (đơn vị: km)
    # LEO: ~ 600km, HAPS: ~ 20km.
    D_LEO = 600 
    D_HAPS = 20
    
    # Tính Path Loss (PL)
    def path_loss_db(D):
        # Công thức cơ bản: PL = 20 log10(D) + 20 log10(f) + C
        # Chúng ta dùng một proxy đơn giản cho suy hao
        return 20 * np.log10(D) + 20 * np.log10(FREQUENCY_GHZ) + 20 

    for t in range(N_TRANS):
        for u in range(N_USERS):
            if t < N_TRANS - 1: # 3 LEOs đầu tiên
                D = D_LEO * np.random.uniform(0.9, 1.1)
                PL_dB = path_loss_db(D)
            else: # HAPS cuối cùng
                D = D_HAPS * np.random.uniform(0.8, 1.2)
                PL_dB = path_loss_db(D)
            
            # Chuyển từ dB sang tuyến tính
            PL_linear = 10**(-PL_dB / 10)
            
            # Rician Fading (giả sử Rician factor K=10, strong LOS)
            # Fading (Alpha^2) ~ Rice(K)
            K = 10.0
            sigma = 1.0 / np.sqrt(K + 1)
            mu = np.sqrt(K / (K + 1))
            
            # Mô phỏng Rician: |Re + j Im|^2
            rayleigh = np.random.randn() * sigma + 1j * np.random.randn() * sigma
            los = mu + 0j
            fading_gain = np.abs(los + rayleigh)**2
            
            # G[t, u] là tổng lợi ích (Channel Gain)
            G[t, u] = PL_linear * fading_gain
    
    # Chuẩn hóa để tránh số quá nhỏ/lớn
    G_max = np.max(G)
    G = G / G_max
    
    print(f"--- Channel Gain Matrix G (Normalized) ---")
    print(G)
    return G

# --- 4. Xây dựng Ma trận QUBO Q (N_QUBITS x N_QUBITS) ---
def build_qubo_matrix(G):
    Q = np.zeros((N_QUBITS, N_QUBITS))
    
    # Lặp qua tất cả các cặp Qubit (i, j)
    # i, j là chỉ số 0 đến 31
    for i in range(N_QUBITS):
        for j in range(i, N_QUBITS):
            t_i, u_i, c_i = map_index_to_vars(i)
            t_j, u_j, c_j = map_index_to_vars(j)
            
            # Thành phần Bậc hai (i != j)
            if i != j:
                # 1. Ràng buộc C1: Phạt nếu 2 liên kết phục vụ cùng 1 người dùng
                if u_i == u_j:
                    # Ràng buộc C1: (x_i + x_j - 1)^2 -> 2*x_i*x_j + ...
                    # Chỉ số bậc hai là 2*lambda
                    Q[i, j] += 2 * LAMBDA_CONSTRAINT 
                    
                # 2. Chi phí Lợi ích: Phạt nếu 2 liên kết đồng kênh gây nhiễu
                if c_i == c_j:
                    # Tính nhiễu tiềm năng: G(t_j -> u_i) * G(t_i -> u_j)
                    # Đây là ma trận đối xứng, G(t', u) * G(t, u')
                    
                    # Chúng ta muốn TỐI THIỂU HÓA H_Benefit = - (tổng lợi ích)
                    # Tức là: Tối đa hóa Sum Rate -> Tối thiểu hóa TÍCH CỦA NÓ.
                    
                    # Giả sử chúng ta dùng proxy đơn giản:
                    # Tránh phân bổ đồng kênh cho hai liên kết nếu chúng có gain mạnh
                    # Cost = - G_i * G_j
                    
                    # **Lưu ý quan trọng**: Vì chúng ta đang tối ưu hóa x (kênh),
                    # công suất p chưa được xác định. Chúng ta giả định P_tx = 1
                    # và dùng Proxy: Giảm thiểu sự chồng chéo giữa các liên kết có Channel Gain cao.
                    
                    # Ký hiệu t_i, u_i là người nhận/gửi
                    # Nhiễu i gây ra cho j, và j gây ra cho i
                    
                    interference_i_to_j = G[t_i, u_j] # t_i truyền đến u_j
                    interference_j_to_i = G[t_j, u_i] # t_j truyền đến u_i
                    
                    # Tổng hình phạt nhiễu tiềm năng (dùng dấu trừ cho tối đa hóa)
                    # H_Benefit = - (I_i_j + I_j_i) * x_i * x_j
                    # => Q_ij = - LAMBDA_BENEFIT * (I_i_j + I_j_i)
                    
                    # Chỉ áp dụng nếu liên kết i và j không phục vụ cùng 1 người dùng
                    if u_i != u_j:
                        Q[i, j] += -LAMBDA_BENEFIT * (interference_i_to_j + interference_j_to_i)

            # Thành phần Bậc nhất (i = j)
            else: # i == j
                # Từ Ràng buộc C1: (x_i - 1)^2 -> -2*x_i + 1 (trong H = x^T Q x)
                # Chỉ số bậc nhất là -LAMBDA_CONSTRAINT * (2k-1) 
                
                # Hàm phạt C1 mở rộng: Sum(x_i)^2 - 2*Sum(x_i) + N_U
                # Phần bậc nhất: -2*x_i
                Q[i, i] += -2 * LAMBDA_CONSTRAINT 
                
    # Đảm bảo tính đối xứng và fill các phần tử dưới đường chéo
    Q = Q + Q.T - np.diag(Q.diagonal())
    
    return Q

# --- 5. Thực thi và Xuất kết quả ---
if __name__ == "__main__":
    print(f"Bắt đầu thiết kế QUBO cho {N_QUBITS} Qubits...")
    
    # 1. Tạo Channel Gain
    G_matrix = generate_channel_gains(seed=42)
    
    # 2. Xây dựng Ma trận QUBO
    Q_matrix = build_qubo_matrix(G_matrix)
    
    print("\n--- Kích thước Ma trận QUBO Q ---")
    print(Q_matrix.shape)
    
    # 3. Lưu ma trận Q vào file (quan trọng cho bước tiếp theo)
    # Chúng ta dùng định dạng NumPy binary để lưu chính xác các giá trị float
    np.save('qubo_matrix_Q_32x32.npy', Q_matrix)
    
    print("\nMa trận QUBO Q đã được lưu vào 'qubo_matrix_Q_32x32.npy'")
    print("Vui lòng kiểm tra các giá trị lớn (Penalty) và nhỏ (Benefit).")
    
    # (Tùy chọn) In một phần ma trận để kiểm tra
    df_Q = pd.DataFrame(Q_matrix).head(10).T.head(10)
    print("\n--- 10x10 Phần đầu của Ma trận Q ---")
    print(df_Q)

# Thao tác Git tiếp theo:
# 1. Thêm file qubo_mapping.py và qubo_matrix_Q_32x32.npy
# 2. Commit và Push lên GitHub.
