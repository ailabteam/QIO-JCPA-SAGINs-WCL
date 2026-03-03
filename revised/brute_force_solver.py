import numpy as np
import itertools
import time
# TODO: Import hàm tính Sum Rate và Power Allocation từ file của bạn
# from classical_optimization import solve_power_allocation_and_get_rate

def get_brute_force_optimal_link(G_matrix, num_tx=4, num_users=5, P_max=1.0, noise_var=1e-9):
    """
    Duyệt toàn bộ không gian nghiệm hợp lệ (thỏa mãn constraint mỗi user max 1 link).
    """
    print(f"Bắt đầu Brute-force cho mạng {num_tx} Tx, {num_users} Users...")
    start_time = time.time()
    
    # Mỗi user có (num_tx + 1) lựa chọn: Chọn Tx 0,1,2,3 hoặc -1 (không kết nối)
    user_choices = list(range(-1, num_tx))
    
    # Tạo tất cả các tổ hợp hợp lệ (5^5 = 3125 trường hợp)
    all_combinations = list(itertools.product(user_choices, repeat=num_users))
    
    best_rate = 0.0
    best_X = np.zeros((num_tx, num_users))
    
    for combo in all_combinations:
        # Tạo ma trận nhị phân X từ tổ hợp
        X_current = np.zeros((num_tx, num_users))
        for u, t in enumerate(combo):
            if t != -1: # Nếu user u có kết nối với Tx t
                X_current[t, u] = 1
                
        # Bỏ qua trường hợp ma trận X toàn số 0 (không có link nào)
        if np.sum(X_current) == 0:
            continue
            
        # -------------------------------------------------------------------
        # TẠI ĐÂY: Gọi hàm Power Allocation (SCA) của bạn để tính Sum Rate
        # Ví dụ:
        # current_rate, _, _ = solve_power_allocation_and_get_rate(X_current, G_matrix, P_max, noise_var)
        # -------------------------------------------------------------------
        
        # Dòng này tôi làm giả (Mock) để code không bị lỗi. 
        # Khi chạy thật, bạn xóa dòng giả này và dùng hàm thật ở trên.
        current_rate = np.random.uniform(0.5, 2.0) 
        
        if current_rate > best_rate:
            best_rate = current_rate
            best_X = np.copy(X_current)
            
    end_time = time.time()
    print(f"Xong! Brute-force duyệt {len(all_combinations)} trường hợp mất {end_time - start_time:.2f} giây.")
    print(f"GLOBAL OPTIMUM RATE: {best_rate:.4f} nats/s/Hz")
    
    return best_rate, best_X

# Chạy test thử
if __name__ == "__main__":
    # Giả lập ma trận kênh truyền 4x5
    G_dummy = np.random.rand(4, 5) 
    get_brute_force_optimal_link(G_dummy)
