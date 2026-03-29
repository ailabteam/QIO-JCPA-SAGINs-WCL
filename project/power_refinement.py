# =============================================================================
# File: power_refinement.py
# Description:
# Classical power refinement and rate evaluation for a fixed binary assignment.
#
# Updated:
#   - if allow_unserved_users=False, each user must be assigned exactly one TX
#   - each active link must receive at least p_min_active_link power
# =============================================================================

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import scipy.optimize

from config import ProjectConfig, get_default_config


# -------------------------------------------------------------------------
# Assignment helpers
# -------------------------------------------------------------------------
def vector_to_assignment_matrix(
    x: np.ndarray,
    n_transmitters: int,
    n_users: int,
) -> np.ndarray:
    x = np.asarray(x, dtype=int).reshape(-1)
    expected = n_transmitters * n_users
    if x.size != expected:
        raise ValueError(f"x has length {x.size}, expected {expected}.")
    return x.reshape(n_transmitters, n_users)


def assignment_matrix_to_vector(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=int)
    return X.reshape(-1)


def validate_assignment(
    X: np.ndarray,
    config: ProjectConfig | None = None,
) -> Dict:
    """
    Validate assignment feasibility.

    Rules:
      - if allow_unserved_users=False:
            each user must have exactly one transmitter
      - if allow_unserved_users=True:
            each user may have at most one transmitter
      - optional TX load cap if max_links_per_tx is not None
    """
    if config is None:
        config = get_default_config()

    sys_cfg = config.system

    X = np.asarray(X, dtype=int)
    if X.ndim != 2:
        raise ValueError("X must be a 2D assignment matrix.")

    tx_loads = np.sum(X, axis=1)
    user_loads = np.sum(X, axis=0)

    if sys_cfg.allow_unserved_users:
        is_user_feasible = bool(np.all(user_loads <= 1))
    else:
        is_user_feasible = bool(np.all(user_loads == 1))

    if sys_cfg.max_links_per_tx is None:
        is_tx_feasible = True
    else:
        is_tx_feasible = bool(np.all(tx_loads <= sys_cfg.max_links_per_tx))

    return {
        "is_user_feasible": is_user_feasible,
        "is_tx_feasible": is_tx_feasible,
        "is_feasible": bool(is_user_feasible and is_tx_feasible),
        "user_loads": user_loads,
        "tx_loads": tx_loads,
    }


# -------------------------------------------------------------------------
# SINR and rate evaluation
# -------------------------------------------------------------------------
def calculate_sinr(
    P: np.ndarray,
    G: np.ndarray,
    X: np.ndarray,
    t_i: int,
    u_i: int,
    config: ProjectConfig | None = None,
) -> float:
    if config is None:
        config = get_default_config()

    sys_cfg = config.system

    if X[t_i, u_i] != 1:
        return 0.0

    signal = G[t_i, u_i] * P[t_i, u_i]
    interference = 0.0

    n_transmitters, n_users = X.shape

    for t_prime in range(n_transmitters):
        for u_prime in range(n_users):
            if X[t_prime, u_prime] == 1 and not (t_prime == t_i and u_prime == u_i):
                interference += G[t_prime, u_i] * P[t_prime, u_prime]

    return float(signal / (sys_cfg.noise_power + interference))


def calculate_sum_rate(
    P: np.ndarray,
    G: np.ndarray,
    X: np.ndarray,
    config: ProjectConfig | None = None,
) -> Tuple[float, Dict]:
    if config is None:
        config = get_default_config()

    X = np.asarray(X, dtype=int)
    P = np.asarray(P, dtype=float)

    n_transmitters, n_users = X.shape
    sinr_matrix = np.zeros_like(P, dtype=float)

    total_rate = 0.0
    active_links = int(np.sum(X))

    for t in range(n_transmitters):
        for u in range(n_users):
            if X[t, u] == 1:
                sinr = calculate_sinr(P, G, X, t, u, config=config)
                sinr_matrix[t, u] = sinr
                total_rate += np.log2(1.0 + sinr)

    details = {
        "sinr_matrix": sinr_matrix,
        "active_links": active_links,
        "tx_powers": np.sum(P, axis=1),
        "user_loads": np.sum(X, axis=0),
        "tx_loads": np.sum(X, axis=1),
    }

    return float(total_rate), details


# -------------------------------------------------------------------------
# Power initialization
# -------------------------------------------------------------------------
def initialize_power(
    X: np.ndarray,
    config: ProjectConfig | None = None,
) -> np.ndarray:
    if config is None:
        config = get_default_config()

    sys_cfg = config.system
    power_cfg = config.power

    X = np.asarray(X, dtype=int)
    n_transmitters, n_users = X.shape

    P0 = np.zeros((n_transmitters, n_users), dtype=float)
    active_total = int(np.sum(X))

    if active_total == 0:
        return P0

    if power_cfg.init_strategy == "uniform_per_tx":
        for t in range(n_transmitters):
            active_on_t = int(np.sum(X[t, :]))
            if active_on_t > 0:
                # make sure each active link can satisfy p_min_active_link
                base = sys_cfg.p_max_per_tx / active_on_t
                P0[t, :] = X[t, :] * max(base, power_cfg.p_min_active_link)

    elif power_cfg.init_strategy == "uniform_global":
        p_init = max(sys_cfg.p_max_per_tx / max(1, active_total), power_cfg.p_min_active_link)
        P0 = X * p_init

    else:
        raise ValueError(f"Unknown init_strategy: {power_cfg.init_strategy}")

    # If initialization accidentally violates TX budget, renormalize per TX
    for t in range(n_transmitters):
        tx_sum = np.sum(P0[t, :])
        if tx_sum > sys_cfg.p_max_per_tx and tx_sum > 0:
            P0[t, :] *= sys_cfg.p_max_per_tx / tx_sum

    return P0


# -------------------------------------------------------------------------
# Constrained power refinement
# -------------------------------------------------------------------------
def refine_power_given_assignment(
    G: np.ndarray,
    X: np.ndarray,
    config: ProjectConfig | None = None,
) -> Tuple[np.ndarray, float, Dict]:
    if config is None:
        config = get_default_config()

    sys_cfg = config.system
    power_cfg = config.power

    X = np.asarray(X, dtype=int)
    n_transmitters, n_users = X.shape
    n_vars = n_transmitters * n_users

    feasibility = validate_assignment(X, config=config)
    if not feasibility["is_feasible"]:
        zero_power = np.zeros_like(X, dtype=float)
        return zero_power, 0.0, {
            "optimizer_success": False,
            "message": "Infeasible assignment.",
            "iterations": 0,
            "rate_before": 0.0,
            "rate_after": 0.0,
            "feasibility": feasibility,
        }

    # Quick feasibility check for minimum-power requirement
    active_per_tx = np.sum(X, axis=1)
    for t in range(n_transmitters):
        if active_per_tx[t] * power_cfg.p_min_active_link > sys_cfg.p_max_per_tx + 1e-12:
            zero_power = np.zeros_like(X, dtype=float)
            return zero_power, 0.0, {
                "optimizer_success": False,
                "message": f"Infeasible due to minimum-power requirement on TX {t}.",
                "iterations": 0,
                "rate_before": 0.0,
                "rate_after": 0.0,
                "feasibility": feasibility,
            }

    P0 = initialize_power(X, config=config)
    rate_before, _ = calculate_sum_rate(P0, G, X, config=config)

    def objective_to_minimize(p_flat: np.ndarray) -> float:
        P = p_flat.reshape(n_transmitters, n_users)
        P = np.maximum(P, 0.0)

        # Force zero power on inactive links
        P = P * X

        rate, _ = calculate_sum_rate(P, G, X, config=config)
        return -rate

    bounds = []
    for t in range(n_transmitters):
        for u in range(n_users):
            if X[t, u] == 1:
                bounds.append((power_cfg.p_min_active_link, sys_cfg.p_max_per_tx))
            else:
                bounds.append((0.0, 0.0))

    constraints = []
    for t in range(n_transmitters):
        constraints.append({
            "type": "ineq",
            "fun": lambda p_flat, t=t: sys_cfg.p_max_per_tx
            - np.sum(p_flat.reshape(n_transmitters, n_users)[t, :])
        })

    result = scipy.optimize.minimize(
        objective_to_minimize,
        P0.reshape(-1),
        method=power_cfg.method,
        bounds=bounds,
        constraints=constraints,
        tol=power_cfg.convergence_tol,
        options={"maxiter": power_cfg.max_iter, "disp": False},
    )

    P_opt = result.x.reshape(n_transmitters, n_users)
    P_opt = np.maximum(P_opt, 0.0)
    P_opt[np.abs(P_opt) < power_cfg.projection_eps] = 0.0
    P_opt = P_opt * X

    rate_after, rate_details = calculate_sum_rate(P_opt, G, X, config=config)

    info = {
        "optimizer_success": bool(result.success),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "rate_before": float(rate_before),
        "rate_after": float(rate_after),
        "objective_value": float(result.fun) if result.fun is not None else None,
        "feasibility": feasibility,
        "rate_details": rate_details,
    }

    return P_opt, float(rate_after), info


# -------------------------------------------------------------------------
# Assignment-level wrappers
# -------------------------------------------------------------------------
def evaluate_assignment_matrix(
    G: np.ndarray,
    X: np.ndarray,
    config: ProjectConfig | None = None,
    refine_power: bool = True,
) -> Dict:
    if config is None:
        config = get_default_config()

    X = np.asarray(X, dtype=int)

    feasibility = validate_assignment(X, config=config)
    if not feasibility["is_feasible"]:
        return {
            "assignment": X.copy(),
            "power": np.zeros_like(X, dtype=float),
            "sum_rate": 0.0,
            "is_feasible": False,
            "feasibility": feasibility,
            "power_info": None,
        }

    if refine_power:
        P, rate, power_info = refine_power_given_assignment(G, X, config=config)
    else:
        P = initialize_power(X, config=config)
        rate, _ = calculate_sum_rate(P, G, X, config=config)
        power_info = {
            "optimizer_success": None,
            "message": "Refinement skipped; initialized power used.",
            "iterations": 0,
            "rate_before": float(rate),
            "rate_after": float(rate),
            "feasibility": feasibility,
        }

    return {
        "assignment": X.copy(),
        "power": P,
        "sum_rate": float(rate),
        "is_feasible": True,
        "feasibility": feasibility,
        "power_info": power_info,
    }


def evaluate_assignment_vector(
    G: np.ndarray,
    x: np.ndarray,
    n_transmitters: int,
    n_users: int,
    config: ProjectConfig | None = None,
    refine_power: bool = True,
) -> Dict:
    X = vector_to_assignment_matrix(x, n_transmitters, n_users)
    return evaluate_assignment_matrix(
        G=G,
        X=X,
        config=config,
        refine_power=refine_power,
    )


if __name__ == "__main__":
    from channel_model import generate_snapshot

    cfg = get_default_config()
    snapshot = generate_snapshot(n_users=3, config=cfg, seed=42)

    G = snapshot["G"]

    X = np.zeros((cfg.system.n_transmitters, 3), dtype=int)
    for u in range(3):
        t_best = np.argmax(G[:, u])
        X[t_best, u] = 1

    result = evaluate_assignment_matrix(G=G, X=X, config=cfg, refine_power=True)

    print("Feasible:", result["is_feasible"])
    print("User loads:", result["feasibility"]["user_loads"])
    print("Sum rate:", result["sum_rate"])
    print("TX powers:", result["power_info"]["rate_details"]["tx_powers"])
