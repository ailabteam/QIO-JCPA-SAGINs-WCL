# ==============================================================================
# File: qubo_mapping.py (REVISED VERSION)
# Mô tả: Thiết lập mô hình NTN cường độ nhiễu cao để đánh bại thuật toán Greedy.
# Cập nhật: N_TRANS = 6, Thiết lập "Greedy Trap" tại trạm HAPS.
# ==============================================================================
import numpy as np
import os

# --- Hằng số Toàn cục (Fixed Constants) ---
C_CHANNELS = 1
N_TRANS = 6            # Tăng lên 6 (5 LEO + 1 HAPS) để tăng độ phức tạp nhiễu
FREQUENCY_GHZ = 28.0
LAMBDA_CONSTRAINT = 50.0  # Điều chỉnh để cân bằng với Benefit term
LAMBDA_BENEFIT = 1.0     # Trọng số tối ưu nhiễu chéo

# --- 1. Hàm Tiện ích ---

def map_index_to_vars(i, N_USERS):
    """Ánh xạ chỉ số Qubit i thành (t, u, c=0)"""
    N_QUBITS = N_TRANS * N_USERS * C_CHANNELS
    if i >= N_QUBITS:
        raise ValueError("Index out of bounds")

    c = i % C_CHANNELS
    temp = i // C_CHANNELS
    u = temp % N_USERS
    t = temp // N_USERS
    return t, u, c

# --- 2. Mô hình Kênh Cơ sở (Thiết lập Interference-Limited Regime) ---
def generate_channel_gains(N_USERS, seed=42):
    """
    Tạo ma trận Channel Gain G[N_TRANS, N_USERS].
    Thiết lập HAPS mạnh vượt trội để 'bẫy' thuật toán Greedy.
    """
    np.random.seed(seed)
    G = np.zeros((N_TRANS, N_USERS))

    D_LEO = 600  # km
    D_HAPS = 20  # km

    def path_loss_db(D):
        return 20 * np.log10(D) + 20 * np.log10(FREQUENCY_GHZ) + 92.45

    for t in range(N_TRANS):
        for u in range(N_USERS):
            if t < N_TRANS - 1: # Các vệ tinh LEO
                D = D_LEO * np.random.uniform(0.9, 1.1)
                # Thêm 15dB suy hao để LEO yếu hơn HAPS (Greedy sẽ bỏ qua LEO)
                PL_dB = path_loss_db(D) + 15 
            else: # Trạm HAPS (Trạm cuối cùng)
                D = D_HAPS * np.random.uniform(0.8, 1.2)
                # Giảm 5dB suy hao để HAPS cực mạnh (Greedy sẽ dồn hết vào đây)
                PL_dB = path_loss_db(D) - 5 

            PL_linear = 10**(-PL_dB / 10)

            # Rician Fading
            K = 10.0
            sigma = 1.0 / np.sqrt(K + 1)
            mu = np.sqrt(K / (K + 1))
            rayleigh = (np.random.randn() + 1j * np.random.randn()) * sigma / np.sqrt(2)
            los = mu
            fading_gain = np.abs(los + rayleigh)**2

            G[t, u] = PL_linear * fading_gain

    # Normalization để ổn định ma trận QUBO
    G_max = np.max(G)
    G = G / G_max

    return G

# --- 3. Xây dựng Ma trận QUBO Q (Phạt nhiễu chéo) ---
def build_qubo_matrix(G, N_USERS):
    """Xây dựng ma trận QUBO Q tập trung vào giảm thiểu nhiễu chéo."""
    N_QUBITS = N_TRANS * N_USERS * C_CHANNELS
    Q = np.zeros((N_QUBITS, N_QUBITS))

    for i in range(N_QUBITS):
        for j in range(i, N_QUBITS):
            t_i, u_i, _ = map_index_to_vars(i, N_USERS)
            t_j, u_j, _ = map_index_to_vars(j, N_USERS)

            # --- 3.1 Thành phần Bậc hai (Tương tác giữa 2 Qubits) ---
            if i != j:
                # 1. Ràng buộc: Mỗi User chỉ được chọn đúng 1 Link (Phạt rất nặng)
                if u_i == u_j:
                    Q[i, j] += 2 * LAMBDA_CONSTRAINT

                # 2. Chi phí Nhiễu chéo: Phạt nặng các cặp Link gây nhiễu cho nhau
                else: # u_i != u_j
                    # Nhiễu chéo từ trạm t_i sang user u_j và ngược lại
                    interference_sum = G[t_i, u_j] + G[t_j, u_i]
                    Q[i, j] += LAMBDA_BENEFIT * interference_sum

            # --- 3.2 Thành phần Bậc nhất (Tự tương tác) ---
            else: # i == j
                # Đảm bảo tổng x_i = 1 cho mỗi user
                Q[i, i] += -1 * LAMBDA_CONSTRAINT
                
                # Khuyến khích chọn link có Gain tốt (nhưng trọng số nhỏ hơn nhiễu)
                Q[i, i] += -0.1 * G[t_i, u_i] 

    # Đảm bảo ma trận đối xứng
    Q = Q + Q.T - np.diag(Q.diagonal())
    return Q

# --- 4. Hàm Chạy Snapshot (Hàm chính được gọi từ các module khác) ---
def run_snapshot_mapping(n_users, seed):
    """Tạo G và Q cho quy mô n_users cụ thể."""
    if n_users <= 0:
        raise ValueError("N_USERS must be greater than 0")

    G_matrix = generate_channel_gains(n_users, seed=seed)
    Q_matrix = build_qubo_matrix(G_matrix, n_users)

    # Lưu dữ liệu snapshot
    np.save('current_G.npy', G_matrix)

    return G_matrix, Q_matrix

# --- 5. Thực thi chính (Dùng để kiểm tra nhanh) ---
if __name__ == "__main__":
    N_USERS_TEST = 5
    G_test, Q_test = run_snapshot_mapping(N_USERS_TEST, seed=42)
    N_QUBITS_TEST = N_TRANS * N_USERS_TEST * C_CHANNELS

    print(f"--- MÔ PHỎNG QUBO NTN (REVISED) ---")
    print(f"Quy mô: {N_TRANS} Transmitters, {N_USERS_TEST} Users.")
    print(f"Tổng số Qubits: {N_QUBITS_TEST}")
    print(f"Độ lợi TB của vệ tinh LEO: {np.mean(G_test[:-1, :]):.6f}")
    print(f"Độ lợi TB của trạm HAPS   : {np.mean(G_test[-1, :]):.6f}")
    print(f"-> HAPS mạnh gấp {np.mean(G_test[-1, :])/np.mean(G_test[:-1, :]):.1f} lần LEO (Greedy Trap)")
    
    Q_FILENAME = f'qubo_matrix_Q_{N_QUBITS_TEST}x{N_QUBITS_TEST}.npy'
    np.save(Q_FILENAME, Q_test)
    print(f"Ma trận Q đã lưu vào '{Q_FILENAME}'")
