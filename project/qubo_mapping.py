# =============================================================================
# File: qubo_mapping.py
# Description:
# Build an analytically motivated QUBO surrogate for binary link selection.
#
# Updated version:
#   - direct-link reward,
#   - pairwise mutual-interference penalty,
#   - EXACTLY-ONE-PER-USER penalty when allow_unserved_users=False,
#   - AT-MOST-ONE-PER-USER penalty when allow_unserved_users=True,
#   - optional transmitter-side penalty,
#   - candidate pruning support.
# =============================================================================

from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import numpy as np

from config import ProjectConfig, get_default_config


# -------------------------------------------------------------------------
# Index mapping utilities
# -------------------------------------------------------------------------
def vars_to_index(t: int, u: int, n_users: int, n_channels: int = 1) -> int:
    """
    Map (t, u, c=0) to flattened binary variable index.
    """
    if n_channels != 1:
        raise NotImplementedError("Current implementation assumes n_channels = 1.")
    return t * n_users + u


def index_to_vars(i: int, n_users: int, n_transmitters: int, n_channels: int = 1) -> Tuple[int, int, int]:
    """
    Map flattened binary index to (t, u, c).
    """
    if n_channels != 1:
        raise NotImplementedError("Current implementation assumes n_channels = 1.")

    n_qubits = n_transmitters * n_users * n_channels
    if i < 0 or i >= n_qubits:
        raise IndexError(f"Index {i} out of bounds for n_qubits={n_qubits}.")

    c = 0
    u = i % n_users
    t = i // n_users
    return t, u, c


# -------------------------------------------------------------------------
# Assignment conversion helpers
# -------------------------------------------------------------------------
def vector_to_assignment_matrix(
    x: np.ndarray,
    n_transmitters: int,
    n_users: int,
) -> np.ndarray:
    """
    Convert a flattened binary vector x into X[t, u].
    """
    x = np.asarray(x, dtype=int).reshape(-1)
    expected = n_transmitters * n_users
    if x.size != expected:
        raise ValueError(f"x has length {x.size}, expected {expected}.")
    return x.reshape(n_transmitters, n_users)


def assignment_matrix_to_vector(X: np.ndarray) -> np.ndarray:
    """
    Convert assignment matrix X[t, u] into flattened vector x.
    """
    X = np.asarray(X, dtype=int)
    return X.reshape(-1)


# -------------------------------------------------------------------------
# Surrogate coefficient builders
# -------------------------------------------------------------------------
def _equal_power_template(config: ProjectConfig, n_users: int) -> np.ndarray:
    """
    Return per-transmitter template power used to derive surrogate coefficients.

    If a TX load cap is set, use Pmax / L_t.
    Otherwise, use Pmax / n_users as a simple equal-split template.
    """
    sys_cfg = config.system
    p_bar = np.zeros(sys_cfg.n_transmitters, dtype=float)

    for t in range(sys_cfg.n_transmitters):
        if sys_cfg.max_links_per_tx is not None and sys_cfg.max_links_per_tx > 0:
            denom = float(sys_cfg.max_links_per_tx)
        else:
            denom = float(max(1, n_users))

        p_bar[t] = sys_cfg.p_max_per_tx / denom

    return p_bar


def compute_direct_reward_matrix(
    G: np.ndarray,
    config: ProjectConfig | None = None,
) -> np.ndarray:
    """
    Compute alpha[t, u], the direct-link reward matrix.

    alpha_{t,u} = w_direct * log(1 + p_bar[t] * G[t,u] / noise)
    """
    if config is None:
        config = get_default_config()

    sys_cfg = config.system
    qubo_cfg = config.qubo

    n_transmitters, n_users = G.shape
    if n_transmitters != sys_cfg.n_transmitters:
        raise ValueError(
            f"G has {n_transmitters} transmitters, expected {sys_cfg.n_transmitters}."
        )

    p_bar = _equal_power_template(config, n_users)
    alpha = np.zeros_like(G, dtype=float)

    for t in range(n_transmitters):
        for u in range(n_users):
            snr_proxy = (p_bar[t] * G[t, u]) / (sys_cfg.noise_power + qubo_cfg.eps)
            alpha[t, u] = qubo_cfg.w_direct_reward * np.log1p(snr_proxy)

    return alpha


def compute_pairwise_interference_penalty(
    G: np.ndarray,
    config: ProjectConfig | None = None,
) -> np.ndarray:
    """
    Compute beta[i, j] > 0 for i != j, where i and j are binary link variables.

    For links i=(t,u), j=(t',u'), with u != u':
        beta_ij = w_interference * (
                    p_bar[t'] * G[t', u] / noise
                  + p_bar[t]  * G[t,  u'] / noise
                 )
    """
    if config is None:
        config = get_default_config()

    sys_cfg = config.system
    qubo_cfg = config.qubo

    n_transmitters, n_users = G.shape
    n_vars = n_transmitters * n_users
    p_bar = _equal_power_template(config, n_users)

    beta = np.zeros((n_vars, n_vars), dtype=float)

    for i in range(n_vars):
        t_i, u_i, _ = index_to_vars(i, n_users, n_transmitters, sys_cfg.n_channels)

        for j in range(i + 1, n_vars):
            t_j, u_j, _ = index_to_vars(j, n_users, n_transmitters, sys_cfg.n_channels)

            # Same-user interactions are handled by user assignment penalties.
            if u_i == u_j:
                continue

            mutual_interference = (
                (p_bar[t_j] * G[t_j, u_i]) / (sys_cfg.noise_power + qubo_cfg.eps)
                + (p_bar[t_i] * G[t_i, u_j]) / (sys_cfg.noise_power + qubo_cfg.eps)
            )

            beta_ij = qubo_cfg.w_interference_penalty * mutual_interference
            beta[i, j] = beta_ij
            beta[j, i] = beta_ij

    return beta


# -------------------------------------------------------------------------
# QUBO construction
# -------------------------------------------------------------------------
def build_qubo_matrix(
    G: np.ndarray,
    candidate_mask: np.ndarray | None = None,
    config: ProjectConfig | None = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Build the QUBO matrix Q and return metadata.

    Updated logic:
      - direct-link reward,
      - pairwise mutual-interference penalty,
      - exactly-one-per-user penalty if allow_unserved_users=False,
      - at-most-one-per-user penalty if allow_unserved_users=True,
      - optional TX-side soft penalty,
      - candidate pruning by large diagonal disabling penalty.
    """
    if config is None:
        config = get_default_config()

    sys_cfg = config.system
    qubo_cfg = config.qubo

    n_transmitters, n_users = G.shape
    if n_transmitters != sys_cfg.n_transmitters:
        raise ValueError(
            f"G has {n_transmitters} transmitters, expected {sys_cfg.n_transmitters}."
        )

    if candidate_mask is None:
        candidate_mask = np.ones_like(G, dtype=int)

    if candidate_mask.shape != G.shape:
        raise ValueError("candidate_mask must have the same shape as G.")

    n_vars = n_transmitters * n_users
    Q = np.zeros((n_vars, n_vars), dtype=float)

    alpha = compute_direct_reward_matrix(G, config=config)
    beta = compute_pairwise_interference_penalty(G, config=config)

    # ---------------------------------------------------------------------
    # 1) Direct-link reward: -alpha_i x_i
    # ---------------------------------------------------------------------
    for t in range(n_transmitters):
        for u in range(n_users):
            i = vars_to_index(t, u, n_users, sys_cfg.n_channels)
            Q[i, i] += -alpha[t, u]

    # ---------------------------------------------------------------------
    # 2) Pairwise mutual-interference penalty: +beta_ij x_i x_j
    # ---------------------------------------------------------------------
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if beta[i, j] != 0.0:
                Q[i, j] += beta[i, j]
                Q[j, i] += beta[i, j]

    # ---------------------------------------------------------------------
    # 3) User-side assignment constraint
    #
    # If allow_unserved_users = False:
    #   enforce exactly one TX per user via
    #       lambda * (sum_t x_{t,u} - 1)^2
    #
    # Expansion:
    #   lambda * [ 2*sum_{t<t'} x_{t,u}x_{t',u} - sum_t x_{t,u} + 1 ]
    #
    # So:
    #   diagonal += -lambda
    #   off-diagonal same user += +2*lambda
    #
    # If allow_unserved_users = True:
    #   enforce only at-most-one via pairwise conflicts
    # ---------------------------------------------------------------------
    user_conflict_pairs: List[Tuple[int, int]] = []
    user_linear_terms: List[int] = []

    if not sys_cfg.allow_unserved_users:
        for u in range(n_users):
            # diagonal linear part: -lambda * x_{t,u}
            for t in range(n_transmitters):
                i = vars_to_index(t, u, n_users, sys_cfg.n_channels)
                Q[i, i] += -qubo_cfg.lambda_user_constraint
                user_linear_terms.append(i)

            # quadratic same-user conflicts: +2*lambda * x_{t,u}x_{t',u}
            for t1, t2 in itertools.combinations(range(n_transmitters), 2):
                i = vars_to_index(t1, u, n_users, sys_cfg.n_channels)
                j = vars_to_index(t2, u, n_users, sys_cfg.n_channels)

                Q[i, j] += 2.0 * qubo_cfg.lambda_user_constraint
                Q[j, i] += 2.0 * qubo_cfg.lambda_user_constraint
                user_conflict_pairs.append((i, j))
    else:
        for u in range(n_users):
            for t1, t2 in itertools.combinations(range(n_transmitters), 2):
                i = vars_to_index(t1, u, n_users, sys_cfg.n_channels)
                j = vars_to_index(t2, u, n_users, sys_cfg.n_channels)

                Q[i, j] += qubo_cfg.lambda_user_constraint
                Q[j, i] += qubo_cfg.lambda_user_constraint
                user_conflict_pairs.append((i, j))

    # ---------------------------------------------------------------------
    # 4) Optional transmitter-side soft penalty
    #
    # For now we keep a simple pairwise overload discouragement when
    # max_links_per_tx <= 1.
    # ---------------------------------------------------------------------
    tx_conflict_pairs: List[Tuple[int, int]] = []

    if sys_cfg.max_links_per_tx is not None and sys_cfg.max_links_per_tx <= 1:
        for t in range(n_transmitters):
            for u1, u2 in itertools.combinations(range(n_users), 2):
                i = vars_to_index(t, u1, n_users, sys_cfg.n_channels)
                j = vars_to_index(t, u2, n_users, sys_cfg.n_channels)

                Q[i, j] += qubo_cfg.lambda_tx_constraint
                Q[j, i] += qubo_cfg.lambda_tx_constraint
                tx_conflict_pairs.append((i, j))

    # ---------------------------------------------------------------------
    # 5) Candidate pruning / disabling
    # ---------------------------------------------------------------------
    disabled_variables: List[int] = []
    large_disable_penalty = 10.0 * max(
        qubo_cfg.lambda_user_constraint,
        qubo_cfg.lambda_tx_constraint,
        qubo_cfg.w_interference_penalty + qubo_cfg.w_direct_reward + 1.0,
    )

    for t in range(n_transmitters):
        for u in range(n_users):
            if int(candidate_mask[t, u]) == 0:
                i = vars_to_index(t, u, n_users, sys_cfg.n_channels)
                Q[i, i] += large_disable_penalty
                disabled_variables.append(i)

    metadata = {
        "n_users": n_users,
        "n_transmitters": n_transmitters,
        "n_variables": n_vars,
        "alpha_matrix": alpha,
        "beta_matrix": beta,
        "candidate_mask": candidate_mask.copy(),
        "disabled_variables": disabled_variables,
        "user_conflict_pairs": user_conflict_pairs,
        "user_linear_terms": user_linear_terms,
        "tx_conflict_pairs": tx_conflict_pairs,
        "qubo_symmetry_error": float(np.max(np.abs(Q - Q.T))),
        "constraint_mode": "exactly_one_per_user" if not sys_cfg.allow_unserved_users else "at_most_one_per_user",
    }

    return Q, metadata


# -------------------------------------------------------------------------
# Objective / diagnostics helpers
# -------------------------------------------------------------------------
def qubo_objective(x: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute x^T Q x for a binary vector x.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    return float(x @ Q @ x)


def summarize_qubo_metadata(metadata: Dict) -> Dict:
    """
    Produce a lightweight summary dictionary for logging/debugging.
    """
    alpha = metadata["alpha_matrix"]
    beta = metadata["beta_matrix"]

    off_diag = beta[np.triu_indices_from(beta, k=1)]

    summary = {
        "n_users": metadata["n_users"],
        "n_transmitters": metadata["n_transmitters"],
        "n_variables": metadata["n_variables"],
        "num_disabled_variables": len(metadata["disabled_variables"]),
        "num_user_conflict_pairs": len(metadata["user_conflict_pairs"]),
        "num_tx_conflict_pairs": len(metadata["tx_conflict_pairs"]),
        "alpha_min": float(np.min(alpha)),
        "alpha_max": float(np.max(alpha)),
        "alpha_mean": float(np.mean(alpha)),
        "beta_nonzero_mean": float(np.mean(off_diag[off_diag > 0])) if np.any(off_diag > 0) else 0.0,
        "beta_nonzero_max": float(np.max(off_diag)) if off_diag.size > 0 else 0.0,
        "qubo_symmetry_error": metadata["qubo_symmetry_error"],
        "constraint_mode": metadata["constraint_mode"],
    }
    return summary


if __name__ == "__main__":
    from channel_model import generate_snapshot

    cfg = get_default_config()
    snapshot = generate_snapshot(n_users=5, config=cfg, seed=42)

    G = snapshot["G"]
    M = snapshot["candidate_mask"]

    Q, meta = build_qubo_matrix(G=G, candidate_mask=M, config=cfg)

    print("Q shape:", Q.shape)
    print("Q symmetry error:", meta["qubo_symmetry_error"])
    print("Q summary:", summarize_qubo_metadata(meta))
