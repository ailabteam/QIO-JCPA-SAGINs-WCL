## QIO-JLSPA-SAGINs-WCL (Quantum-Inspired Optimization for Real-Time Co-Channel Link Selection in LEO-HAPS Networks)

This repository contains the simulation environment and numerical results supporting the paper:

**"Quantum-Inspired Optimization for Real-Time Co-Channel Link Selection and Power Control in Integrated LEO-HAPS Networks"**
*(Intended for submission to IEEE Wireless Communications Letters (WCL))*

---

## 1. Project Overview and Contribution

The primary challenge in Satellite-Air-Ground Integrated Networks (SAGINs) is the need for real-time optimization of wireless resources under dynamic channel conditions and severe co-channel interference. The **Joint Link Selection and Power Allocation (JLSPA)** problem is an NP-hard Mixed-Integer Non-Linear Program (MINLP), rendering classical iterative algorithms too slow for real-time operation (millisecond-scale decisions).

This work proposes the **Quantum-Inspired Link Selection and Power Allocation (QIO-JLSPA)** framework. We decouple the MINLP into two parts:
1. **Combinatorial Problem (Link Selection):** Solved by mapping to a **Quadratic Unconstrained Binary Optimization (QUBO)** model.
2. **Continuous Problem (Power Allocation):** Solved using classical Successive Convex Approximation (SCA).

We utilize the **Simulated Annealing (Neal)** algorithm, a leading Quantum-Inspired Heuristic (QIH), to achieve high-quality solutions for the QUBO model in a significantly reduced runtime, demonstrating the critical **quality-runtime trade-off** required for next-generation Non-Terrestrial Networks (NTN).

## 2. System Model and Scope

The simulation models a single snapshot of a two-tier SAGIN under co-channel interference ($C=1$).

*   **Network Configuration:** 4 Transmitters ($N_T=4$; 3 LEO Satellites + 1 HAPS) serving up to 5 User Equipment ($N_U=5$).
*   **Problem Size:** The Link Selection problem maps to a **20-Qubit QUBO** matrix.
*   **Key Finding:** The QIO-JLSPA approach achieves $\approx 94\%$ of the high-complexity classical optimum in $\mathbf{2.66 \times}$ less total time, validating the real-time feasibility of QIH techniques.

## 3. Implementation Details and Environment Setup

The code is primarily written in Python and uses specialized libraries for optimization and channel modeling.

### Prerequisites

We recommend setting up a dedicated Conda environment:

```bash
# Create and activate the environment
conda create -n wcl_qio python=3.9 -y
conda activate wcl_qio

# Install required libraries
pip install numpy scipy pandas matplotlib neal pulp
```

### File Structure

| File | Description |
| :--- | :--- |
| `qubo_mapping.py` | Generates the Channel Gain matrix ($G$) and constructs the QUBO matrix ($Q$) based on cost and penalty terms. |
| `classical_optimization.py` | Contains the core functions for classical analysis, including the **Successive Convex Approximation (SCA)**-based Power Allocation (Step 2) and Sum Rate calculation. |
| `figure_generator.py` | **Main execution file.** Loops through different system scales ($N_U=2$ to 5), runs all benchmarks, and generates final figures/tables. |
| `channel_gain_G_4x5.npy` | Saved array of the normalized Channel Gain matrix $G$. |
| `qubo_matrix_Q_20x20.npy` | Saved array of the final QUBO matrix $Q$ (for $N_U=5$). |
| `table1_summary_wcl.csv` | Final numerical results comparing Runtime and Quality for the WCL paper. |
| `figure1_quality_scale.pdf` | Figure showing Sum Rate vs. System Scale. |
| `figure2_tradeoff_runtime.pdf` | Figure showing Solution Quality vs. Total Runtime (log scale) - The core WCL argument. |

## 4. How to Reproduce Results

To reproduce the full set of numerical results and figures presented in the paper:

1. **Clone the Repository:** Ensure you have the latest code.
2. **Setup Environment:** Run the commands listed in Section 3.
3. **Run Data Acquisition:** Execute the main generator file. This will automatically run all benchmarks and save the results.

```bash
conda activate wcl_qio
python figure_generator.py
```

*(This process will take several minutes as it iterates through 20 total snapshots.)*

## 5. Benchmarks and Metrics

The results in this repository compare the following policies, normalized against the ICA (Iterative Classical Algorithm) as the 100% Quality Benchmark:

| Policy | Description | Quality Basis | Runtime Basis |
| :--- | :--- | :--- | :--- |
| **QIO-JLSPA (HPQ)** | Quantum-Inspired (Neal, High Reads). **The proposed solution.** | High Quality $R_{\text{QUBO}}$ | Measured Time $T_{\text{QIH}}$ |
| **ICA (Proxy)** | Iterative Classical Algorithm (SCA/AO). | Highest Observed Rate $R_{\text{ICA}}$ | **Assumed Total Convergence Time** $T_{\text{ICA}} \approx 400 \text{ ms}$ |
| **GRY (Baseline)** | Greedy Heuristic (Highest Channel Gain). | Low/Medium Quality | Measured Time $T_{\text{GRY}}$ |

---

## 6. Citation

*(To be updated upon publication)*

