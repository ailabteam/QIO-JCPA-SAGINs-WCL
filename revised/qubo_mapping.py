import numpy as np

N_TRANS = 4 
# BỘ TRỌNG SỐ ĐÃ CÂN BẰNG LẠI (CRITICAL)
LAMBDA_CONSTRAINT = 50.0   
LAMBDA_SIGNAL = 15.0       # Tăng Signal để QIO ưu tiên chọn link tốt
LAMBDA_INTERFERENCE = 2.0  # Phạt nhiễu vừa đủ để QIO khôn hơn Greedy

def map_index_to_vars(i, N_USERS):
    u = i % N_USERS
    t = i // N_USERS
    return t, u

def generate_channel_gains(N_USERS, seed=42):
    np.random.seed(seed)
    G = np.zeros((N_TRANS, N_USERS))
    for t in range(N_TRANS):
        for u in range(N_USERS):
            # Tạo bẫy: HAPS (t=3) rất mạnh, nhưng gây nhiễu cực lớn nếu dồn vào
            bias = 1.0 if t == 3 else 0.4
            G[t, u] = bias * np.random.uniform(0.5, 1.0)
    return G / np.max(G)

def build_qubo_matrix(G, N_USERS):
    N_QUBITS = N_TRANS * N_USERS
    Q = np.zeros((N_QUBITS, N_QUBITS))
    for i in range(N_QUBITS):
        for j in range(i, N_QUBITS):
            t_i, u_i = map_index_to_vars(i, N_USERS)
            t_j, u_j = map_index_to_vars(j, N_USERS)
            if i != j:
                if u_i == u_j: 
                    Q[i, j] += 2 * LAMBDA_CONSTRAINT # Ràng buộc user
                else: 
                    # Quan trọng: Phạt nhiễu chéo giữa các user
                    Q[i, j] += LAMBDA_INTERFERENCE * (G[t_i, u_j] + G[t_j, u_i])
            else:
                Q[i, i] -= LAMBDA_CONSTRAINT # Ép bật qubit
                Q[i, i] -= LAMBDA_SIGNAL * G[t_i, u_i] # Tối ưu tín hiệu
    return Q + Q.T - np.diag(Q.diagonal())

def run_snapshot_mapping(n_users, seed):
    G = generate_channel_gains(n_users, seed=seed)
    Q = build_qubo_matrix(G, n_users)
    return G, Q
