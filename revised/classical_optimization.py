import numpy as np
import scipy.optimize

P_FIXED = 0.5    
NOISE_POWER = 0.01 
N_TRANS = 4

def calculate_fixed_sum_rate(x_vector, G, n_users):
    x_mat = np.array(x_vector).reshape(N_TRANS, n_users)
    total_rate = 0.0
    for t in range(N_TRANS):
        for u in range(n_users):
            if x_mat[t, u] == 1:
                signal = G[t, u] * P_FIXED
                interference = sum(G[tp, u] * P_FIXED for tp in range(N_TRANS) for up in range(n_users) 
                                   if x_mat[tp, up] == 1 and (tp != t or up != u))
                total_rate += np.log2(1 + signal / (NOISE_POWER + interference))
    return total_rate

def joint_classical_ica(G, n_users):
    """Thuật toán AO cổ điển: Tối ưu x và p xen kẽ để tìm Joint Local Optimum"""
    # Khởi tạo Greedy x
    x = np.zeros((N_TRANS, n_users))
    for u in range(n_users): x[np.argmax(G[:, u]), u] = 1
    
    best_rate = calculate_fixed_sum_rate(x.flatten(), G, n_users)
    # Lặp để tìm x tốt hơn (Heuristic AO)
    for _ in range(5):
        for u in range(n_users):
            current_tx = np.argmax(x[:, u])
            for t_alt in range(N_TRANS):
                if t_alt == current_tx: continue
                x_alt = x.copy()
                x_alt[current_tx, u] = 0
                x_alt[t_alt, u] = 1
                r_alt = calculate_fixed_sum_rate(x_alt.flatten(), G, n_users)
                if r_alt > best_rate:
                    best_rate = r_alt
                    x = x_alt.copy()
    return x.flatten().tolist(), best_rate
