# =============================================================================
# File: baselines.py
# Description:
# Baseline wrappers for binary link selection + classical power refinement.
#
# Supported baselines:
#   - Greedy
#   - QIO (default Neal)
#   - QIO high-quality (Neal high reads)
#   - Local search
#   - Exact-small enumeration
#
# All baselines are evaluated using the same downstream power refinement step.
# =============================================================================

from __future__ import annotations

import itertools
import time
from typing import Dict, List

import numpy as np

from config import ProjectConfig, get_default_config
from power_refinement import (
    evaluate_assignment_matrix,
    assignment_matrix_to_vector,
)
from qubo_mapping import (
    build_qubo_matrix,
    compute_direct_reward_matrix,
    vector_to_assignment_matrix,
)
from solvers import (
    solve_qio_default,
    solve_qio_high_quality,
    solve_local_search_qubo,
)


# -------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------
def _format_baseline_result(
    name: str,
    X: np.ndarray,
    binary_objective: float | None,
    binary_runtime_ms: float,
    evaluation_result: Dict,
) -> Dict:
    """
    Standardized result dictionary for all baselines.
    """
    total_runtime_ms = binary_runtime_ms

    power_info = evaluation_result.get("power_info", None)

    return {
        "policy": name,
        "assignment": X.copy(),
        "x": assignment_matrix_to_vector(X),
        "sum_rate": float(evaluation_result["sum_rate"]),
        "is_feasible": bool(evaluation_result["is_feasible"]),
        "power": evaluation_result["power"].copy(),
        "binary_objective": None if binary_objective is None else float(binary_objective),
        "runtime_ms": float(total_runtime_ms),
        "power_info": power_info,
        "feasibility": evaluation_result.get("feasibility", None),
    }


def _evaluate_end_to_end(
    name: str,
    X: np.ndarray,
    binary_objective: float | None,
    start_time: float,
    G: np.ndarray,
    config: ProjectConfig,
    refine_power: bool = True,
) -> Dict:
    """
    Run downstream evaluation and package the final result.
    """
    evaluation_result = evaluate_assignment_matrix(
        G=G,
        X=X,
        config=config,
        refine_power=refine_power,
    )

    total_runtime_ms = (time.time() - start_time) * 1000.0

    result = _format_baseline_result(
        name=name,
        X=X,
        binary_objective=binary_objective,
        binary_runtime_ms=total_runtime_ms,
        evaluation_result=evaluation_result,
    )
    result["runtime_ms"] = float(total_runtime_ms)
    return result


# -------------------------------------------------------------------------
# Greedy baseline
# -------------------------------------------------------------------------
def solve_greedy_assignment(
    G: np.ndarray,
    candidate_mask: np.ndarray | None = None,
    config: ProjectConfig | None = None,
) -> np.ndarray:
    """
    Greedy binary assignment:
      for each user, choose the transmitter with the strongest direct gain
      among candidate transmitters.

    If allow_unserved_users=True and no feasible candidate exists, the user
    remains unserved.
    """
    if config is None:
        config = get_default_config()

    n_transmitters, n_users = G.shape
    X = np.zeros((n_transmitters, n_users), dtype=int)

    if candidate_mask is None:
        candidate_mask = np.ones_like(G, dtype=int)

    max_links_per_tx = config.system.max_links_per_tx
    tx_loads = np.zeros(n_transmitters, dtype=int)

    for u in range(n_users):
        feasible_tx = [
            t for t in range(n_transmitters)
            if candidate_mask[t, u] == 1
            and (max_links_per_tx is None or tx_loads[t] < max_links_per_tx)
        ]

        if len(feasible_tx) == 0:
            if config.system.allow_unserved_users:
                continue
            raise ValueError(f"No feasible transmitter available for user {u}.")

        gains = np.array([G[t, u] for t in feasible_tx], dtype=float)
        best_t = feasible_tx[int(np.argmax(gains))]
        X[best_t, u] = 1
        tx_loads[best_t] += 1

    return X


def run_greedy_baseline(
    G: np.ndarray,
    candidate_mask: np.ndarray | None = None,
    config: ProjectConfig | None = None,
    refine_power: bool = True,
) -> Dict:
    """
    Run Greedy + downstream power refinement.
    """
    if config is None:
        config = get_default_config()

    start = time.time()

    X = solve_greedy_assignment(
        G=G,
        candidate_mask=candidate_mask,
        config=config,
    )

    return _evaluate_end_to_end(
        name="Greedy",
        X=X,
        binary_objective=None,
        start_time=start,
        G=G,
        config=config,
        refine_power=refine_power,
    )


# -------------------------------------------------------------------------
# QIO baselines
# -------------------------------------------------------------------------
def run_qio_default_baseline(
    G: np.ndarray,
    candidate_mask: np.ndarray | None = None,
    config: ProjectConfig | None = None,
    refine_power: bool = True,
) -> Dict:
    """
    Run default QIO (Neal with default reads) + power refinement.
    """
    if config is None:
        config = get_default_config()

    start = time.time()

    Q, _ = build_qubo_matrix(
        G=G,
        candidate_mask=candidate_mask,
        config=config,
    )

    result_bin = solve_qio_default(Q=Q, config=config)
    X = vector_to_assignment_matrix(
        result_bin["x"],
        n_transmitters=config.system.n_transmitters,
        n_users=G.shape[1],
    )

    return _evaluate_end_to_end(
        name="QIO-Default",
        X=X,
        binary_objective=result_bin["objective"],
        start_time=start,
        G=G,
        config=config,
        refine_power=refine_power,
    )


def run_qio_high_quality_baseline(
    G: np.ndarray,
    candidate_mask: np.ndarray | None = None,
    config: ProjectConfig | None = None,
    refine_power: bool = True,
) -> Dict:
    """
    Run high-quality QIO (Neal with higher reads) + power refinement.
    """
    if config is None:
        config = get_default_config()

    start = time.time()

    Q, _ = build_qubo_matrix(
        G=G,
        candidate_mask=candidate_mask,
        config=config,
    )

    result_bin = solve_qio_high_quality(Q=Q, config=config)
    X = vector_to_assignment_matrix(
        result_bin["x"],
        n_transmitters=config.system.n_transmitters,
        n_users=G.shape[1],
    )

    return _evaluate_end_to_end(
        name="QIO-HighQuality",
        X=X,
        binary_objective=result_bin["objective"],
        start_time=start,
        G=G,
        config=config,
        refine_power=refine_power,
    )


# -------------------------------------------------------------------------
# Local-search baseline
# -------------------------------------------------------------------------
def run_local_search_baseline(
    G: np.ndarray,
    candidate_mask: np.ndarray | None = None,
    config: ProjectConfig | None = None,
    refine_power: bool = True,
    seed: int | None = None,
) -> Dict:
    """
    Run local search on the QUBO objective + downstream power refinement.
    """
    if config is None:
        config = get_default_config()

    start = time.time()

    if candidate_mask is None:
        candidate_mask = np.ones_like(G, dtype=int)

    Q, _ = build_qubo_matrix(
        G=G,
        candidate_mask=candidate_mask,
        config=config,
    )
    alpha = compute_direct_reward_matrix(G=G, config=config)

    rng = np.random.default_rng(seed)

    result_bin = solve_local_search_qubo(
        Q=Q,
        n_transmitters=config.system.n_transmitters,
        n_users=G.shape[1],
        candidate_mask=candidate_mask,
        config=config,
        score_matrix=alpha,
        rng=rng,
    )

    X = result_bin["X"]

    return _evaluate_end_to_end(
        name="LocalSearch",
        X=X,
        binary_objective=result_bin["objective"],
        start_time=start,
        G=G,
        config=config,
        refine_power=refine_power,
    )


# -------------------------------------------------------------------------
# Exact-small enumeration
# -------------------------------------------------------------------------
def _enumerate_all_feasible_assignments(
    n_transmitters: int,
    n_users: int,
    candidate_mask: np.ndarray | None,
    config: ProjectConfig,
):
    """
    Enumerate all feasible assignments for small instances.

    For each user:
      - choose one transmitter among feasible candidates, or
      - choose "unserved" if allow_unserved_users=True

    This enumeration is exponential in n_users and intended only for small cases.
    """
    if candidate_mask is None:
        candidate_mask = np.ones((n_transmitters, n_users), dtype=int)

    sys_cfg = config.system

    per_user_choices: List[List[int]] = []

    for u in range(n_users):
        feasible_tx = [t for t in range(n_transmitters) if candidate_mask[t, u] == 1]
        if sys_cfg.allow_unserved_users:
            feasible_tx = feasible_tx + [-1]
        per_user_choices.append(feasible_tx)

    for u, choices in enumerate(per_user_choices):
        if len(choices) == 0:
            raise ValueError(
                f"User {u} has no feasible transmitter choices under current candidate_mask."
            )

    for choice_tuple in itertools.product(*per_user_choices):
        X = np.zeros((n_transmitters, n_users), dtype=int)

        valid = True
        tx_loads = np.zeros(n_transmitters, dtype=int)

        for u, t in enumerate(choice_tuple):
            if t != -1:
                X[t, u] = 1
                tx_loads[t] += 1

                if sys_cfg.max_links_per_tx is not None and tx_loads[t] > sys_cfg.max_links_per_tx:
                    valid = False
                    break

        if valid:
            yield X


def run_exact_small_baseline(
    G: np.ndarray,
    candidate_mask: np.ndarray | None = None,
    config: ProjectConfig | None = None,
    refine_power: bool = True,
) -> Dict:
    """
    Run exact enumeration over feasible assignments for small cases only.
    """
    if config is None:
        config = get_default_config()

    n_transmitters, n_users = G.shape

    if n_users > config.solver.exact_max_users:
        raise ValueError(
            f"Exact-small baseline is only allowed up to "
            f"{config.solver.exact_max_users} users; got {n_users}."
        )

    start = time.time()

    Q, _ = build_qubo_matrix(
        G=G,
        candidate_mask=candidate_mask,
        config=config,
    )

    best_result = None
    best_rate = -np.inf
    best_binary_obj = None
    num_evaluated = 0

    for X in _enumerate_all_feasible_assignments(
        n_transmitters=n_transmitters,
        n_users=n_users,
        candidate_mask=candidate_mask,
        config=config,
    ):
        evaluation_result = evaluate_assignment_matrix(
            G=G,
            X=X,
            config=config,
            refine_power=refine_power,
        )
        num_evaluated += 1

        if evaluation_result["sum_rate"] > best_rate:
            best_rate = evaluation_result["sum_rate"]
            best_result = evaluation_result
            x_vec = assignment_matrix_to_vector(X)
            best_binary_obj = float(x_vec @ Q @ x_vec)

    total_runtime_ms = (time.time() - start) * 1000.0

    result = {
        "policy": "ExactSmall",
        "assignment": best_result["assignment"].copy(),
        "x": assignment_matrix_to_vector(best_result["assignment"]),
        "sum_rate": float(best_result["sum_rate"]),
        "is_feasible": bool(best_result["is_feasible"]),
        "power": best_result["power"].copy(),
        "binary_objective": best_binary_obj,
        "runtime_ms": float(total_runtime_ms),
        "power_info": best_result.get("power_info", None),
        "feasibility": best_result.get("feasibility", None),
        "num_evaluated_assignments": int(num_evaluated),
    }
    return result


# -------------------------------------------------------------------------
# Baseline collection for a single snapshot
# -------------------------------------------------------------------------
def run_snapshot_baselines(
    G: np.ndarray,
    candidate_mask: np.ndarray | None = None,
    config: ProjectConfig | None = None,
    seed: int | None = None,
    refine_power: bool = True,
) -> List[Dict]:
    """
    Run all enabled baselines for a single snapshot and return a list of results.
    """
    if config is None:
        config = get_default_config()

    exp_cfg = config.experiment
    results: List[Dict] = []

    if exp_cfg.run_greedy:
        results.append(
            run_greedy_baseline(
                G=G,
                candidate_mask=candidate_mask,
                config=config,
                refine_power=refine_power,
            )
        )

    if exp_cfg.run_qio_neal:
        results.append(
            run_qio_default_baseline(
                G=G,
                candidate_mask=candidate_mask,
                config=config,
                refine_power=refine_power,
            )
        )

        results.append(
            run_qio_high_quality_baseline(
                G=G,
                candidate_mask=candidate_mask,
                config=config,
                refine_power=refine_power,
            )
        )

    if exp_cfg.run_local_search:
        results.append(
            run_local_search_baseline(
                G=G,
                candidate_mask=candidate_mask,
                config=config,
                refine_power=refine_power,
                seed=seed,
            )
        )

    if exp_cfg.run_exact_small and G.shape[1] <= config.solver.exact_max_users:
        results.append(
            run_exact_small_baseline(
                G=G,
                candidate_mask=candidate_mask,
                config=config,
                refine_power=refine_power,
            )
        )

    return results


if __name__ == "__main__":
    from channel_model import generate_snapshot

    cfg = get_default_config()
    snapshot = generate_snapshot(n_users=4, config=cfg, seed=42)

    G = snapshot["G"]
    M = snapshot["candidate_mask"]

    results = run_snapshot_baselines(
        G=G,
        candidate_mask=M,
        config=cfg,
        seed=42,
        refine_power=True,
    )

    for res in results:
        print(
            f"{res['policy']:>15s} | "
            f"Rate = {res['sum_rate']:.4f} | "
            f"Runtime = {res['runtime_ms']:.2f} ms | "
            f"Feasible = {res['is_feasible']}"
        )
