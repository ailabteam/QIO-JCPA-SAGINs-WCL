# ==============================================================================
# File: qaoa_solver.py
# Mô tả: Triển khai thuật toán QAOA p=2 trên ma trận QUBO 20x20.
# Cập nhật: Fix lỗi logging, đảm bảo output console liên tục.
# ==============================================================================
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import torch
import time
import pandas as pd
import logging
import sys # Dùng để flush output

# --- CẤU HÌNH LOGGING ---
# Tắt logging mặc định của PennyLane/PyTorch nếu cần, nhưng thường không cần
# Thiết lập logger gốc để đảm bảo output console và file
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Xóa handlers cũ để tránh output trùng lặp
if logger.hasHandlers():
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        
LOG_FILENAME = 'qaoa_run.log'

# File handler
file_handler = logging.FileHandler(LOG_FILENAME, mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Console handler (Chỉ in ra message)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(console_handler)


# --- 1. Cấu hình Hệ thống ---
N_QUBITS = 20
QAOA_P = 2  
N_ITERATIONS = 50 # <-- Giảm xuống 50 lần lặp để kiểm tra tốc độ ban đầu
LEARNING_RATE = 0.01
CHECKPOINT_INTERVAL = 100 

dev = qml.device("default.qubit", wires=N_QUBITS) 


# --- 2. Xây dựng QAOA Hamiltonian ---
def build_qaoa_hamiltonian(Q_matrix):
    """Xây dựng Cost Hamiltonian H_C và Mixer Hamiltonian H_M."""
    coeffs = []
    observables = []

    # Bảng Lookup để xử lý các term Z_i và Z_i Z_j
    for i in range(N_QUBITS):
        for j in range(i, N_QUBITS):
            Q_ij = Q_matrix[i, j]
            
            if Q_ij == 0.0: continue

            if i == j:
                # Term bậc nhất: Hằng số + (-Q_ii/2) * Z_i
                coeffs.append(-Q_ij / 2.0)
                observables.append(qml.PauliZ(i))
            else:
                # Term bậc hai: (Q_ij/4) * Z_i Z_j - (Q_ij/4) * Z_i - (Q_ij/4) * Z_j
                
                # Z_i @ Z_j term
                coeffs.append(Q_ij / 4.0)
                observables.append(qml.PauliZ(i) @ qml.PauliZ(j))
                
                # Z_i term
                coeffs.append(-Q_ij / 4.0)
                observables.append(qml.PauliZ(i))
                
                # Z_j term
                coeffs.append(-Q_ij / 4.0)
                observables.append(qml.PauliZ(j))

    H_C = qml.Hamiltonian(coeffs, observables)
    
    mixer_coeffs = [1.0] * N_QUBITS
    mixer_obs = [qml.PauliX(i) for i in range(N_QUBITS)]
    H_M = qml.Hamiltonian(mixer_coeffs, mixer_obs)
    
    return H_C, H_M


# --- 3. Thiết kế Mạch QAOA (QNode) ---
@qml.qnode(dev, interface="torch")
def qaoa_circuit(gamma, beta, H_C, H_M):
    """Thiết lập mạch QAOA."""
    qml.broadcast(qml.Hadamard, wires=range(N_QUBITS), pattern="single")

    for k in range(QAOA_P):
        # Cost Layer
        qml.ApproxTimeEvolution(H_C, gamma[k], 1)
        
        # Mixer Layer
        fill_value = (2 * beta[k]).item() 
        mixer_params = torch.full((N_QUBITS,), fill_value)
        qml.broadcast(qml.RX, wires=range(N_QUBITS), parameters=mixer_params, pattern="single")
        
    return qml.expval(H_C)

# --- 4. Chức năng Lập trình Chính ---
def run_qaoa_optimization(Q_matrix, N_ITERATIONS):
    logging.info(f"--- Bắt đầu Tối ưu hóa QAOA (p={QAOA_P}, Q={N_QUBITS}) ---")
    
    H_C, H_M = build_qaoa_hamiltonian(Q_matrix)
    params_init = torch.nn.Parameter(torch.rand(2 * QAOA_P))
    opt = torch.optim.Adam([params_init], lr=LEARNING_RATE)
    history = {'iteration': [], 'cost': [], 'params': []}
    start_time = time.time()
    
    for step in range(1, N_ITERATIONS + 1):
        step_start_time = time.time()
        gamma = params_init[:QAOA_P]
        beta = params_init[QAOA_P:]
        
        try:
            cost = qaoa_circuit(gamma, beta, H_C, H_M)
            
            opt.zero_grad()
            cost.backward()
            opt.step()
            
            # Ghi Log và Console Output
            current_cost = cost.item()
            history['iteration'].append(step)
            history['cost'].append(current_cost)
            history['params'].append(params_init.detach().numpy())
            
            step_end_time = time.time()
            step_runtime_ms = (step_end_time - step_start_time) * 1000

            # Log mỗi 10 bước
            if step % 10 == 0 or step == 1: 
                logging.info(f"Step {step}/{N_ITERATIONS} | Cost: {current_cost:.6f} | Runtime: {step_runtime_ms:.2f} ms")

            if step % CHECKPOINT_INTERVAL == 0:
                torch.save({'step': step, 'params': params_init.state_dict()}, 'qaoa_checkpoint.pth')
                logging.info(f"Checkpoint saved at step {step}.")
                
        except Exception as e:
            logging.error(f"Error at step {step}: {e}")
            break

    total_time = time.time() - start_time
    logging.info(f"\nOptimization Finished in {total_time:.2f} seconds.")
    
    # 5. Lấy kết quả tối ưu và vector x tối ưu
    optimal_params = params_init.detach().numpy()
    optimal_gamma = optimal_params[:QAOA_P]
    optimal_beta = optimal_params[QAOA_P:]

    @qml.qnode(dev)
    def probability_circuit(gamma, beta, H_C, H_M):
        qaoa_circuit(gamma, beta, H_C, H_M)
        return qml.probs(wires=range(N_QUBITS))

    probs = probability_circuit(optimal_gamma, optimal_beta, H_C, H_M)
    optimal_state_index = np.argmax(probs)
    optimal_x_bin = np.array(list(np.binary_repr(optimal_state_index, width=N_QUBITS)), dtype=int)

    return optimal_x_bin, history


if __name__ == "__main__":
    # Load ma trận Q (Cần đảm bảo file 20x20 đã được tạo)
    Q_FILENAME = 'qubo_matrix_Q_20x20.npy'
    try:
        Q_matrix = np.load(Q_FILENAME)
        if Q_matrix.shape != (N_QUBITS, N_QUBITS):
             logging.error(f"Kích thước ma trận Q không khớp ({Q_matrix.shape} != {N_QUBITS}x{N_QUBITS}).")
             sys.exit()
    except FileNotFoundError:
        logging.error(f"Lỗi: Không tìm thấy '{Q_FILENAME}'.")
        sys.exit()
        
    # Chạy Tối ưu hóa QAOA
    optimal_x_qaoa, history = run_qaoa_optimization(Q_matrix, N_ITERATIONS)
    
    logging.info("\n--- Kết quả QAOA ---")
    logging.info(f"Vector Phân bổ Tối ưu (x): {optimal_x_qaoa}")
    
    qubo_cost = optimal_x_qaoa @ Q_matrix @ optimal_x_qaoa.T
    logging.info(f"Chi phí QUBO tối thiểu tìm được: {qubo_cost:.2f}")
    
    # Lưu Log lịch sử hội tụ cuối cùng
    df_history = pd.DataFrame(history)
    df_history.to_csv('qaoa_history_p2.log', index=False)
    logging.info("Lịch sử hội tụ QAOA đã được lưu vào 'qaoa_history_p2.log'.")
