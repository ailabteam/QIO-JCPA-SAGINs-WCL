# ==============================================================================
# File: qaoa_solver.py
# Mô tả: Triển khai thuật toán QAOA p=2 trên ma trận QUBO 20x20.
# ==============================================================================
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import torch
import time
import logging # <-- Import thư viện logging


# --- CẤU HÌNH LOGGING ---
LOG_FILENAME = 'qaoa_run.log'
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
# Thêm console handler để in ra màn hình
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)

# Chỉ thêm console handler nếu nó chưa được thêm
if not logging.getLogger().hasHandlers():
    logging.getLogger().addHandler(console)

# --- 1. Cấu hình Hệ thống ---
N_QUBITS = 20
QAOA_P = 2  # Độ sâu của QAOA (p=2 là lựa chọn tối ưu cho Letter)
N_ITERATIONS = 500 # Số lần lặp tối ưu hóa
LEARNING_RATE = 0.01
CHECKPOINT_INTERVAL = 100 # Lưu checkpoint sau mỗi 100 bước


# Thiết lập Device (sử dụng lightning.qubit để tận dụng hiệu suất C++ / GPU)
# Cần kiểm tra xem lightning.qubit có dùng GPU tự động qua PyTorch không
# Hiện tại, chúng ta dùng CPU/RAM để mô phỏng chính xác statevector
dev = qml.device("default.qubit", wires=N_QUBITS) #

# --- 2. Xây dựng QAOA Hamiltonian ---

def build_qaoa_hamiltonian(Q_matrix):
    """Xây dựng Cost Hamiltonian H_C từ ma trận QUBO Q."""
    coeffs = []
    observables = []

    # Chuyển đổi ma trận Q thành danh sách các phép toán (terms)
    for i in range(N_QUBITS):
        for j in range(i, N_QUBITS):
            Q_ij = Q_matrix[i, j]

            if Q_ij == 0.0:
                continue

            if i == j:
                # Terms bậc nhất (Diagonal elements Q[i,i] * x_i)
                # Chuyển từ x_i = (1 - Z_i)/2 sang Z_i
                # Term: Q_ii * (1 - Z_i) / 2 = Q_ii/2 - (Q_ii/2) * Z_i

                # Cần xử lý hằng số (Q_ii/2) sau khi tính expectation value.
                # Observable: Z_i
                coeffs.append(-Q_ij / 2.0)
                observables.append(qml.PauliZ(i))

            else:
                # Terms bậc hai (Q[i,j] * x_i * x_j)
                # Chuyển từ x_i*x_j = (1 - Z_i - Z_j + Z_i*Z_j) / 4

                # Observable: Z_i @ Z_j
                coeffs.append(Q_ij / 4.0)
                observables.append(qml.PauliZ(i) @ qml.PauliZ(j))

                # Observable: Z_i
                coeffs.append(-Q_ij / 4.0)
                observables.append(qml.PauliZ(i))

                # Observable: Z_j
                coeffs.append(-Q_ij / 4.0)
                observables.append(qml.PauliZ(j))

    # Xây dựng Cost Hamiltonian H_C
    H_C = qml.Hamiltonian(coeffs, observables)

    # Xây dựng Mixer Hamiltonian H_M
    # Sử dụng cú pháp qml.Hamiltonian ổn định hơn để định nghĩa Sum of PauliX
    mixer_coeffs = [1.0] * N_QUBITS
    mixer_obs = [qml.PauliX(i) for i in range(N_QUBITS)]
    H_M = qml.Hamiltonian(mixer_coeffs, mixer_obs)


    return H_C, H_M


# --- 3. Thiết kế Mạch QAOA (QNode) ---

@qml.qnode(dev, interface="torch")
def qaoa_circuit(gamma, beta, H_C, H_M):
    """
    Thiết lập mạch QAOA.
    Initial state: Superposition |+>^N
    """
    # Bước 1: Khởi tạo trạng thái siêu vị trí
    qml.broadcast(qml.Hadamard, wires=range(N_QUBITS), pattern="single")

    # Bước 2: Lặp lại p lớp Cost và Mixer
    for k in range(QAOA_P):
        # Cost Layer (U_C = exp(-i * gamma_k * H_C))
        qml.ApproxTimeEvolution(H_C, gamma[k], 1)

        # Mixer Layer (U_M = exp(-i * beta_k * H_M))
        fill_value = (2 * beta[k]).item() #
        mixer_params = torch.full((N_QUBITS,), fill_value)

        qml.broadcast(qml.RX, wires=range(N_QUBITS), parameters=mixer_params, pattern="single")



    # Bước 3: Đo lường (Chúng ta đo lường expectation value của H_C)
    return qml.expval(H_C)

# --- 4. Chức năng Lập trình Chính ---

def run_qaoa_optimization(Q_matrix, N_ITERATIONS):
    logging.info(f"--- Bắt đầu Tối ưu hóa QAOA (p={QAOA_P}, Q={N_QUBITS}) ---")
    
    H_C, H_M = build_qaoa_hamiltonian(Q_matrix)
    
    # Khởi tạo tham số
    # Sử dụng torch.nn.Parameter để dễ dàng lưu/load trạng thái
    params_init = torch.nn.Parameter(torch.rand(2 * QAOA_P))

    opt = torch.optim.Adam([params_init], lr=LEARNING_RATE)
    
    history = {'iteration': [], 'cost': [], 'params': []}
    start_time = time.time()
    
    for step in range(1, N_ITERATIONS + 1):
        gamma = params_init[:QAOA_P]
        beta = params_init[QAOA_P:]
        
        # Cập nhật Cost
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

            if step % 10 == 0 or step == 1: # Log thường xuyên hơn
                logging.info(f"Step {step}/{N_ITERATIONS} | Cost: {current_cost:.6f}")

            # Checkpointing (Lưu trạng thái tối ưu hóa trung gian)
            if step % CHECKPOINT_INTERVAL == 0:
                torch.save({
                    'step': step,
                    'params': params_init.state_dict(),
                    'optimizer_state': opt.state_dict(),
                }, 'qaoa_checkpoint.pth')
                logging.info(f"Checkpoint saved at step {step}.")
                
        except Exception as e:
            # Nếu xảy ra lỗi giữa chừng, log và thoát
            logging.error(f"Error at step {step}: {e}")
            break


    end_time = time.time()
    logging.info(f"\nOptimization Finished in {end_time - start_time:.2f} seconds.")


    # 5. Lấy kết quả tối ưu và vector x tối ưu

    # Tham số tối ưu
    optimal_params = params_init.detach().numpy()
    optimal_gamma = optimal_params[:QAOA_P]
    optimal_beta = optimal_params[QAOA_P:]

    # Sử dụng tham số tối ưu để lấy xác suất trạng thái
    @qml.qnode(dev)
    def probability_circuit(gamma, beta, H_C, H_M):
        qaoa_circuit(gamma, beta, H_C, H_M)
        return qml.probs(wires=range(N_QUBITS))

    # Chạy mạch để lấy phân bố xác suất
    probs = probability_circuit(optimal_gamma, optimal_beta, H_C, H_M)

    # Lấy trạng thái có xác suất cao nhất
    optimal_state_index = np.argmax(probs)

    # Chuyển đổi index thành vector nhị phân x
    optimal_x_bin = np.array(list(np.binary_repr(optimal_state_index, width=N_QUBITS)), dtype=int)

    return optimal_x_bin, history, H_C.constant # Trả về hằng số H_C để tính chi phí thực tế


if __name__ == "__main__":
    # Load ma trận Q 
    try:
        Q_matrix = np.load('qubo_matrix_Q_20x20.npy')
    except FileNotFoundError:
        logging.error("Lỗi: Không tìm thấy 'qubo_matrix_Q_20x20.npy'.")
        exit()
        
    # Chạy Tối ưu hóa QAOA
    optimal_x_qaoa, history, H_C_constant = run_qaoa_optimization(Q_matrix, N_ITERATIONS)
    
    logging.info("\n--- Kết quả QAOA ---")
    logging.info(f"Vector Phân bổ Kênh Tối ưu (x): \n{optimal_x_qaoa}")
    
    qubo_cost = optimal_x_qaoa @ Q_matrix @ optimal_x_qaoa.T
    logging.info(f"Chi phí QUBO tối thiểu tìm được (x^T Q x): {qubo_cost:.2f}")
    
    # Lưu Log lịch sử hội tụ cuối cùng
    df_history = pd.DataFrame(history)
    df_history.to_csv('qaoa_history_p2.log', index=False)
    logging.info("Lịch sử hội tụ QAOA đã được lưu vào 'qaoa_history_p2.log'.")

