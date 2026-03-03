import numpy as np

C_CHANNELS = 1
N_TRANS = 6 
FREQUENCY_GHZ = 28.0

# ĐIỀU CHỈNH TRỌNG SỐ (CỰC KỲ QUAN TRỌNG)
LAMBDA_CONSTRAINT = 10.0   # Giảm xuống để không lấn át tín hiệu
LAMBDA_SIGNAL = 15.0       # Tăng mạnh để QIO ưu tiên chọn link mạnh
LAMBDA_INTERFERENCE = 1.0  # Giữ nguyên để phạt nhiễu ở mức vừa phải

def map_index_to_vars(i, N_USERS):
    N_QUBITS = N_TRANS * N_USERS * C_CHANNELS
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
            # Tạo sự khác biệt lớn giữa các máy phát để Greedy dễ bị sai
            if t < N_TRANS - 1: # LEO
                PL_dB = path_loss_db(D_LEO) + np.random.uniform(5, 15)
            else: # HAPS (Rất mạnh)
                PL_dB = path_loss_db(D_HAPS) - np.random.uniform(5, 10)
            
            G[t, u] = 10**(-PL_dB / 10)
            K = 10.0
            sigma, mu = 1.0 / np.sqrt(K + 1), np.sqrt(K / (K + 1))
            fading = np.abs(mu + (np.random.randn() + 1j*np.random.randn())*sigma/np.sqrt(2))**2
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
                # Phạt nếu cùng 1 user chọn 2 trạm
                if u_i == u_j:
                    Q[i, j] += 2 * LAMBDA_CONSTRAINT
                # Phạt nhiễu chéo giữa các user khác nhau
                else:
                    Q[i, j] += LAMBDA_INTERFERENCE * (G[t_i, u_j] + G[t_j, u_i])
            else: # i == j
                # Ràng buộc chọn ít nhất 1 trạm
                Q[i, i] -= LAMBDA_CONSTRAINT
                # KHUYẾN KHÍCH CHỌN LINK MẠNH (Dấu trừ để cực tiểu hóa)
                Q[i, i] -= LAMBDA_SIGNAL * G[t_i, u_i]

    return Q + Q.T - np.diag(Q.diagonal())

def run_snapshot_mapping(n_users, seed):
    G = generate_channel_gains(n_users, seed=seed)
    Q = build_qubo_matrix(G, n_users)
    return G, Q
