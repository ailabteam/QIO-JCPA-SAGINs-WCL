# ==============================================================================
# File: qaoa_solver.py
# Mô tả: Triển khai thuật toán QAOA p=2 trên ma trận QUBO 32x32.
# ==============================================================================
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import torch
import time

# --- 1. Cấu hình Hệ thống ---
N_QUBITS = 20
QAOA_P = 2  # Độ sâu của QAOA (p=2 là lựa chọn tối ưu cho Letter)
N_ITERATIONS = 500 # Số lần lặp tối ưu hóa
LEARNING_RATE = 0.01

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
    """Chạy quá trình tối ưu hóa tham số QAOA."""

    H_C, H_M = build_qaoa_hamiltonian(Q_matrix)

    # Khởi tạo tham số (ngẫu nhiên)
    # 2p tham số: p gamma (cost) và p beta (mixer)
    params_init = torch.rand(2 * QAOA_P, requires_grad=True)

    # Sử dụng PyTorch Optimizer (Adam)
    opt = torch.optim.Adam([params_init], lr=LEARNING_RATE)

    # Log dữ liệu
    history = {'iteration': [], 'cost': [], 'params': []}

    print(f"\n--- Bắt đầu Tối ưu hóa QAOA (p={QAOA_P}) ---")
    start_time = time.time()

    for step in range(1, N_ITERATIONS + 1):
        # Phân tách params thành gamma và beta
        gamma = params_init[:QAOA_P]
        beta = params_init[QAOA_P:]

        # Tính toán chi phí (Cost Function = Expectation Value of H_C)
        cost = qaoa_circuit(gamma, beta, H_C, H_M)

        # Tối ưu hóa
        opt.zero_grad()
        cost.backward()
        opt.step()

        # Lưu Log
        history['iteration'].append(step)
        history['cost'].append(cost.item())
        history['params'].append(params_init.detach().numpy())

        if step % 50 == 0 or step == 1:
            print(f"Step {step}/{N_ITERATIONS} | Chi phí (Cost): {cost.item():.6f}")

    end_time = time.time()
    print(f"\nOptimization Finished in {end_time - start_time:.2f} seconds.")

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
    # Load ma trận Q đã tính toán trước đó
    try:
        Q_matrix = np.load('qubo_matrix_Q_32x32.npy')
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy 'qubo_matrix_Q_32x32.npy'. Vui lòng chạy qubo_mapping.py trước.")
        exit()

    # Chạy Tối ưu hóa QAOA
    optimal_x_qaoa, history, H_C_constant = run_qaoa_optimization(Q_matrix, N_ITERATIONS)

    print("\n--- Kết quả QAOA ---")
    print(f"Vector Phân bổ Kênh Tối ưu (x): \n{optimal_x_qaoa}")

    # Tính chi phí thực tế (Minimum Energy)
    # H_C = x^T Q x + Constant
    # Chuyển x_i = (1 - Z_i)/2, Z_i = 1 - 2x_i
    # Chi phí QUBO = x^T Q x

    qubo_cost = optimal_x_qaoa @ Q_matrix @ optimal_x_qaoa.T

    print(f"Chi phí QUBO tối thiểu tìm được (x^T Q x): {qubo_cost:.2f}")

    # Lưu Log lịch sử hội tụ
    df_history = pd.DataFrame(history)
    df_history.to_csv('qaoa_history_p2.log', index=False)

    print("\nLịch sử hội tụ QAOA đã được lưu vào 'qaoa_history_p2.log'.")
