import numpy as np

P_FIXED = 0.5    # Tăng công suất để tín hiệu vượt qua nhiễu nền
NOISE_POWER = 0.01 
N_TRANS = 4      # Quay lại đúng 4 Transmitters như System Model trong bài

def calculate_fixed_sum_rate(x_vector, G, n_users):
    x_mat = np.array(x_vector).reshape(N_TRANS, n_users)
    total_rate = 0.0
    for t in range(N_TRANS):
        for u in range(n_users):
            if x_mat[t, u] == 1:
                signal = G[t, u] * P_FIXED
                interference = 0.0
                for tp in range(N_TRANS):
                    for up in range(n_users):
                        if x_mat[tp, up] == 1 and (tp != t or up != u):
                            interference += G[tp, u] * P_FIXED
                sinr = signal / (NOISE_POWER + interference)
                total_rate += np.log2(1 + sinr)
    return total_rate

def solve_link_selection_and_get_rate(x_vector, G, n_users):
    x_check = np.array(x_vector).reshape(N_TRANS, n_users)
    if not np.all(np.sum(x_check, axis=0) == 1):
        return 0.0
    return calculate_fixed_sum_rate(x_vector, G, n_users)
