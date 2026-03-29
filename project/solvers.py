# =============================================================================
# File: solvers.py
# Description:
# Binary/QUBO solvers and helper utilities for the refactored QIO-JLSPA project.
#
# This module contains:
#   - QUBO objective evaluation
#   - Simulated annealing (Neal) wrapper
#   - Random feasible assignment generator
#   - Local search over binary link assignments
# =============================================================================

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import neal
import numpy as np

from config import ProjectConfig, get_default_config
from qubo_mapping import (
    assignment_matrix_to_vector,
    qubo_objective,
    vector_to_assignment_matrix,
)


# -------------------------------------------------------------------------
# QUBO / binary helpers
# -------------------------------------------------------------------------
def vector_to_matrix(
    x: np.ndarray,
    n_transmitters: int,
    n_users: int,
) -> np.ndarray:
    """
    Alias wrapper for consistency inside solver code.
    """
    return vector_to_assignment_matrix(x, n_transmitters, n_users)


def matrix_to_vector(X: np.ndarray) -> np.ndarray:
    """
    Alias wrapper for consistency inside solver code.
    """
    return assignment_matrix_to_vector(X)


def build_qubo_dict(Q: np.ndarray) -> Dict[Tuple[int, int], float]:
    """
    Convert a dense symmetric Q matrix to a dictionary format accepted by Neal.
    """
    n = Q.shape[0]
    qubo_dict = {}

    for i in range(n):
        for j in range(i, n):
            if Q[i, j] != 0.0:
                qubo_dict[(i, j)] = float(Q[i, j])

    return qubo_dict


# -------------------------------------------------------------------------
# Feasibility-preserving random assignment generator
# -------------------------------------------------------------------------
def generate_random_feasible_assignment(
    n_transmitters: int,
    n_users: int,
    candidate_mask: np.ndarray | None = None,
    config: ProjectConfig | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a random feasible assignment X[t, u] under:
      - at most one transmitter per user
      - optional max_links_per_tx

    If allow_unserved_users=True, a user may remain unassigned.
    """
    if config is None:
        config = get_default_config()

    if rng is None:
        rng = np.random.default_rng()

    sys_cfg = config.system

    if candidate_mask is None:
        candidate_mask = np.ones((n_transmitters, n_users), dtype=int)

    X = np.zeros((n_transmitters, n_users), dtype=int)
    tx_loads = np.zeros(n_transmitters, dtype=int)

    user_order = rng.permutation(n_users)

    for u in user_order:
        feasible_tx = [
            t for t in range(n_transmitters)
            if candidate_mask[t, u] == 1
            and (
                sys_cfg.max_links_per_tx is None
                or tx_loads[t] < sys_cfg.max_links_per_tx
            )
        ]

        # Optionally allow the user to remain unserved
        if sys_cfg.allow_unserved_users:
            choices = feasible_tx + [-1]
        else:
            choices = feasible_tx

        if len(choices) == 0:
            continue

        chosen = rng.choice(choices)
        if chosen != -1:
            X[chosen, u] = 1
            tx_loads[chosen] += 1

    return X


# -------------------------------------------------------------------------
# Neal / simulated annealing QUBO solver
# -------------------------------------------------------------------------
def solve_qubo_neal(
    Q: np.ndarray,
    num_reads: int = 100,
) -> Dict:
    """
    Solve the QUBO using D-Wave Neal simulated annealing.

    Returns
    -------
    result : dict
        {
            "x": best binary vector,
            "objective": best x^T Q x,
            "runtime_ms": runtime in milliseconds,
            "num_reads": num_reads,
        }
    """
    start = time.time()

    qubo_dict = build_qubo_dict(Q)
    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample_qubo(qubo_dict, num_reads=num_reads)

    best_sample = response.first.sample
    n = Q.shape[0]

    x = np.zeros(n, dtype=int)
    for i, val in best_sample.items():
        x[i] = int(val)

    objective = qubo_objective(x, Q)
    runtime_ms = (time.time() - start) * 1000.0

    return {
        "x": x,
        "objective": float(objective),
        "runtime_ms": float(runtime_ms),
        "num_reads": int(num_reads),
    }


# -------------------------------------------------------------------------
# Feasibility repair for binary assignments
# -------------------------------------------------------------------------
def repair_assignment(
    X: np.ndarray,
    candidate_mask: np.ndarray | None = None,
    config: ProjectConfig | None = None,
    score_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """
    Repair a binary assignment matrix to satisfy:
      - at most one transmitter per user
      - optional max_links_per_tx
      - candidate mask

    Repair heuristic:
      1) remove disallowed candidate links,
      2) for each user with multiple active links, keep the best scored one,
      3) if TX load exceeds cap, drop weakest assigned links.
    """
    if config is None:
        config = get_default_config()

    sys_cfg = config.system
    X = np.asarray(X, dtype=int).copy()

    n_transmitters, n_users = X.shape

    if candidate_mask is None:
        candidate_mask = np.ones_like(X, dtype=int)

    if score_matrix is None:
        score_matrix = np.zeros_like(X, dtype=float)

    # Step 1: remove disallowed links
    X = X * candidate_mask

    # Step 2: enforce at most one TX per user
    for u in range(n_users):
        active_tx = np.where(X[:, u] == 1)[0]
        if len(active_tx) > 1:
            best_t = active_tx[np.argmax(score_matrix[active_tx, u])]
            X[active_tx, u] = 0
            X[best_t, u] = 1

    # Step 3: enforce optional TX load cap
    if sys_cfg.max_links_per_tx is not None:
        for t in range(n_transmitters):
            active_users = np.where(X[t, :] == 1)[0]
            overflow = len(active_users) - sys_cfg.max_links_per_tx
            if overflow > 0:
                # Drop weakest assigned users first
                ranked_users = sorted(
                    active_users,
                    key=lambda u: score_matrix[t, u]
                )
                users_to_drop = ranked_users[:overflow]
                X[t, users_to_drop] = 0

    return X


# -------------------------------------------------------------------------
# Neighborhood generators for local search
# -------------------------------------------------------------------------
def generate_user_reassignment_neighbors(
    X: np.ndarray,
    candidate_mask: np.ndarray | None = None,
    config: ProjectConfig | None = None,
) -> List[np.ndarray]:
    """
    Generate neighbors by reassigning each user to another candidate transmitter
    (or optionally to 'unserved' if allowed).
    """
    if config is None:
        config = get_default_config()

    sys_cfg = config.system

    X = np.asarray(X, dtype=int)
    n_transmitters, n_users = X.shape

    if candidate_mask is None:
        candidate_mask = np.ones_like(X, dtype=int)

    neighbors = []

    for u in range(n_users):
        current_active = np.where(X[:, u] == 1)[0]
        current_t = current_active[0] if len(current_active) == 1 else None

        # Candidate transmitters for this user
        feasible_tx = [t for t in range(n_transmitters) if candidate_mask[t, u] == 1]

        move_choices = feasible_tx.copy()
        if sys_cfg.allow_unserved_users:
            move_choices.append(-1)

        for t_new in move_choices:
            if current_t == t_new:
                continue

            X_new = X.copy()

            # Clear current assignment of user u
            X_new[:, u] = 0

            # Assign to new transmitter if not unserved
            if t_new != -1:
                X_new[t_new, u] = 1

            neighbors.append(X_new)

    return neighbors


def evaluate_assignment_under_qubo(
    X: np.ndarray,
    Q: np.ndarray,
) -> float:
    """
    Evaluate an assignment matrix under x^T Q x.
    """
    x = matrix_to_vector(X)
    return qubo_objective(x, Q)


# -------------------------------------------------------------------------
# Local search over assignment matrices
# -------------------------------------------------------------------------
def solve_local_search_qubo(
    Q: np.ndarray,
    n_transmitters: int,
    n_users: int,
    candidate_mask: np.ndarray | None = None,
    config: ProjectConfig | None = None,
    score_matrix: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> Dict:
    """
    Multi-start local search on assignment matrices.

    Objective:
        minimize x^T Q x

    The search:
      - initializes from random feasible assignments,
      - explores user-reassignment neighbors,
      - keeps the best improving move until local optimum,
      - repeats over multiple restarts.

    Returns
    -------
    result : dict
        {
            "x": best binary vector,
            "X": best assignment matrix,
            "objective": best QUBO objective,
            "runtime_ms": total runtime,
            "num_restarts": ...
        }
    """
    if config is None:
        config = get_default_config()

    if rng is None:
        rng = np.random.default_rng()

    solver_cfg = config.solver

    if candidate_mask is None:
        candidate_mask = np.ones((n_transmitters, n_users), dtype=int)

    if score_matrix is None:
        score_matrix = np.zeros((n_transmitters, n_users), dtype=float)

    start = time.time()

    best_X = None
    best_obj = np.inf

    for _ in range(solver_cfg.local_search_num_restarts):
        X = generate_random_feasible_assignment(
            n_transmitters=n_transmitters,
            n_users=n_users,
            candidate_mask=candidate_mask,
            config=config,
            rng=rng,
        )
        X = repair_assignment(
            X,
            candidate_mask=candidate_mask,
            config=config,
            score_matrix=score_matrix,
        )

        current_obj = evaluate_assignment_under_qubo(X, Q)

        improved = True
        n_steps = 0

        while improved and n_steps < solver_cfg.local_search_max_iters:
            improved = False
            n_steps += 1

            neighbors = generate_user_reassignment_neighbors(
                X,
                candidate_mask=candidate_mask,
                config=config,
            )

            best_neighbor = None
            best_neighbor_obj = current_obj

            for Xn in neighbors:
                Xn = repair_assignment(
                    Xn,
                    candidate_mask=candidate_mask,
                    config=config,
                    score_matrix=score_matrix,
                )
                obj_n = evaluate_assignment_under_qubo(Xn, Q)

                if obj_n < best_neighbor_obj:
                    best_neighbor_obj = obj_n
                    best_neighbor = Xn

            if best_neighbor is not None:
                X = best_neighbor
                current_obj = best_neighbor_obj
                improved = True

        if current_obj < best_obj:
            best_obj = current_obj
            best_X = X.copy()

    runtime_ms = (time.time() - start) * 1000.0
    best_x = matrix_to_vector(best_X)

    return {
        "x": best_x,
        "X": best_X,
        "objective": float(best_obj),
        "runtime_ms": float(runtime_ms),
        "num_restarts": int(solver_cfg.local_search_num_restarts),
    }


# -------------------------------------------------------------------------
# Convenience wrappers
# -------------------------------------------------------------------------
def solve_qio_default(
    Q: np.ndarray,
    config: ProjectConfig | None = None,
) -> Dict:
    """
    Default QIO solver wrapper using Neal with the default number of reads.
    """
    if config is None:
        config = get_default_config()

    return solve_qubo_neal(
        Q=Q,
        num_reads=config.solver.neal_num_reads_default,
    )


def solve_qio_high_quality(
    Q: np.ndarray,
    config: ProjectConfig | None = None,
) -> Dict:
    """
    High-quality QIO wrapper using Neal with more reads.
    """
    if config is None:
        config = get_default_config()

    return solve_qubo_neal(
        Q=Q,
        num_reads=config.solver.neal_num_reads_high_quality,
    )


if __name__ == "__main__":
    from channel_model import generate_snapshot
    from qubo_mapping import build_qubo_matrix, compute_direct_reward_matrix

    cfg = get_default_config()
    snapshot = generate_snapshot(n_users=5, config=cfg, seed=42)

    G = snapshot["G"]
    M = snapshot["candidate_mask"]

    Q, meta = build_qubo_matrix(G=G, candidate_mask=M, config=cfg)
    alpha = compute_direct_reward_matrix(G=G, config=cfg)

    qio_result = solve_qio_default(Q=Q, config=cfg)
    print("QIO objective:", qio_result["objective"])
    print("QIO runtime (ms):", qio_result["runtime_ms"])

    ls_result = solve_local_search_qubo(
        Q=Q,
        n_transmitters=cfg.system.n_transmitters,
        n_users=snapshot["n_users"],
        candidate_mask=M,
        config=cfg,
        score_matrix=alpha,
    )
    print("Local search objective:", ls_result["objective"])
    print("Local search runtime (ms):", ls_result["runtime_ms"])
