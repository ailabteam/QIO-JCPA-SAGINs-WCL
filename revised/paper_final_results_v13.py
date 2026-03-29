import numpy as np
import pandas as pd
import neal, itertools, time, pygad
from qubo_mapping import run_snapshot_mapping, N_TRANS
from classical_optimization import calculate_fixed_sum_rate, joint_classical_ica

def ga_solver(Q_mat, n_users):
    def fitness_func(ga_inst, sol, idx):
        energy = np.dot(sol, np.dot(Q_mat, sol))
        penalty = 0 if np.all(np.sum(sol.reshape(N_TRANS, n_users), axis=0) == 1) else 1000
        return -(energy + penalty)
    ga = pygad.GA(num_generations=100, num_parents_mating=10, fitness_func=fitness_func,
                  sol_per_pop=40, num_genes=N_TRANS*n_users, gene_space=[0, 1], stop_criteria="reach_0")
    ga.run()
    return ga.best_solution()[0].tolist()

def qio_solver(Q_mat, n_users):
    bqm = {(i, j): Q_mat[i, j] for i in range(Q_mat.shape[0]) for j in range(i, Q_mat.shape[1])}
    res = neal.SimulatedAnnealingSampler().sample_qubo(bqm, num_reads=1000)
    for s in res.samples():
        if np.all(np.sum(np.array([s[k] for k in range(N_TRANS*n_users)]).reshape(N_TRANS, n_users), axis=0) == 1):
            return [s[k] for k in range(N_TRANS*n_users)]
    return [res.first.sample[k] for k in range(N_TRANS*n_users)]

if __name__ == "__main__":
    print(">>> RUNNING RIGOROUS V13 (FINAL BATTLE - 20 SEEDS)...")
    N_SCALES = [4, 8, 12, 16]
    final_data = []

    for n_u in N_SCALES:
        print(f"Processing {n_u} Users...")
        results = {'QIO':[], 'GA':[], 'GRY':[], 'Joint':[], 'Truth':[], 't_QIO':[], 't_GA':[]}
        
        for s in range(20): # 20 seeds
            G, Q = run_snapshot_mapping(n_u, seed=13000+s)
            
            # 1. Proposed QIO
            start = time.time()
            x_q = qio_solver(Q, n_u)
            results['t_QIO'].append((time.time()-start)*1000)
            results['QIO'].append(calculate_fixed_sum_rate(x_q, G, n_u))
            
            # 2. GA Baseline
            start = time.time()
            x_ga = ga_solver(Q, n_u)
            results['t_GA'].append((time.time()-start)*1000)
            results['GA'].append(calculate_fixed_sum_rate(x_ga, G, n_u))
            
            # 3. Joint ICA (Validation for Decoupling)
            _, r_joint = joint_classical_ica(G, n_u)
            results['Joint'].append(r_joint)
            
            # 4. Greedy
            x_g = np.zeros((N_TRANS, n_u))
            for u in range(n_u): x_g[np.argmax(G[:, u]), u] = 1
            results['GRY'].append(calculate_fixed_sum_rate(x_g.flatten(), G, n_u))
            
            # 5. Brute Force Truth (Small scale only)
            if n_u == 4:
                choices = list(itertools.product(range(N_TRANS), repeat=n_u))
                best_r = 0
                for c in choices:
                    x_t = np.zeros((N_TRANS, n_u))
                    for u, t in enumerate(c): x_t[t, u] = 1
                    best_r = max(best_r, calculate_fixed_sum_rate(x_t.flatten(), G, n_u))
                results['Truth'].append(best_r)

        final_data.append({
            'Users': n_u,
            'Rate_QIO': np.mean(results['QIO']),
            'Rate_GA': np.mean(results['GA']),
            'Rate_Joint': np.mean(results['Joint']),
            'Rate_Greedy': np.mean(results['GRY']),
            'Rate_Truth': np.mean(results['Truth']) if n_u==4 else 0,
            'Time_QIO': np.mean(results['t_QIO']),
            'Time_GA': np.mean(results['t_GA'])
        })

    df = pd.DataFrame(final_data)
    print("\n" + "="*80 + "\nFINAL RIGOROUS DATA FOR IEEE WCL\n" + "="*80)
    print(df)
    df.to_csv('final_v13_rigorous.csv', index=False)
