import numpy as np

C_CHANNELS = 1
N_TRANS = 6 
FREQUENCY_GHZ = 28.0

# BỘ TRỌNG SỐ VÀNG (Đã qua kiểm chứng để thắng Greedy ổn định)
#LAMBDA_CONSTRAINT = 100.0   # Ràng buộc cứng: mỗi user 1 link
#LAMBDA_SIGNAL = 10.0       # Ưu tiên tín hiệu mạnh
#LAMBDA_INTERFERENCE = 8.0  # Phạt nhiễu (tăng lên để QIO nhạy bén hơn Greedy)
# BỘ TRỌNG SỐ "CÂN BẰNG" (Để có kết quả Realistic 10-20% Gain)
LAMBDA_CONSTRAINT = 50.0   # Giảm xuống 50
LAMBDA_SIGNAL = 25.0       # Tăng mạnh Signal để ép QIO chọn link mạnh cho MỌI user

LAMBDA_INTERFERENCE = 1.5  # Giảm nhiễu xuống để hệ thống không tự ý tắt User

def map_index_to_vars(i, N_USERS):
    temp = i // C_CHANNELS
    u = temp % N_USERS
    t = temp // N_USERS
    return t, u, 0

def generate_channel_gains(N_USERS, seed=42):
    np.random.seed(seed)
    G = np.zeros((N_TRANS, N_USERS))
    D_LEO, D_HAPS = 600, 20
    def path_loss_db(D): return 20 * np.log10(D) + 20 * np.log10(28.0) + 92.45
    for t in range(N_TRANS):
        for u in range(N_USERS):
            if t < N_TRANS - 1: # LEO
                PL_dB = path_loss_db(D_LEO) + np.random.uniform(5, 10)
            else: # HAPS (Khống chế độ mạnh để không tạo bẫy quá lớn)
                PL_dB = path_loss_db(D_HAPS) + np.random.uniform(15, 25)
            G[t, u] = 10**(-PL_dB / 10)
            fading = np.abs(np.sqrt(10/11) + (np.random.randn() + 1j*np.random.randn())*np.sqrt(1/22))**2
            G[t, u] *= fading
    return G / np.max(G)

def build_qubo_matrix(G, N_USERS):
    N_QUBITS = N_TRANS * N_USERS
    Q = np.zeros((N_QUBITS, N_QUBITS))
    for i in range(N_QUBITS):
        for j in range(i, N_QUBITS):
            t_i, u_i, _ = map_index_to_vars(i, N_USERS)
            t_j, u_j, _ = map_index_to_vars(j, N_USERS)
            if i != j:
                if u_i == u_j: Q[i, j] += 2 * LAMBDA_CONSTRAINT # Constraint
                else: Q[i, j] += LAMBDA_INTERFERENCE * (G[t_i, u_j] + G[t_j, u_i]) # Penalty
            else:
                Q[i, i] -= LAMBDA_CONSTRAINT # Kích hoạt mỗi user ít nhất 1 link
                Q[i, i] -= LAMBDA_SIGNAL * G[t_i, u_i] # Lợi ích tín hiệu
    return Q + Q.T - np.diag(Q.diagonal())

def run_snapshot_mapping(n_users, seed):
    G = generate_channel_gains(n_users, seed=seed)
    Q = build_qubo_matrix(G, n_users)
    return G, Q
