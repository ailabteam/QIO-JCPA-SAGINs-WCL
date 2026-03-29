# =============================================================================
# File: channel_model.py
# Description:
# Channel generation utilities for integrated LEO-HAPS snapshots.
# The module provides synthetic channel gain generation consistent with the
# original project logic, while exposing a cleaner interface through config.py.
# =============================================================================

from __future__ import annotations

import numpy as np

from config import ProjectConfig, get_default_config


def path_loss_db(distance_km: float, frequency_ghz: float) -> float:
    """
    Simple synthetic path-loss model in dB.

    This is intentionally kept close to the original code logic so that the
    refactored project remains comparable to prior experiments.
    """
    return 20.0 * np.log10(distance_km) + 20.0 * np.log10(frequency_ghz) + 20.0


def rician_power_gain(rng: np.random.Generator, k_factor: float) -> float:
    """
    Generate a Rician fading power gain |h|^2.
    """
    sigma = 1.0 / np.sqrt(k_factor + 1.0)
    mu = np.sqrt(k_factor / (k_factor + 1.0))

    rayleigh = rng.normal(0.0, sigma) + 1j * rng.normal(0.0, sigma)
    los = mu + 0j
    h = los + rayleigh
    return float(np.abs(h) ** 2)


def generate_channel_gains(
    n_users: int,
    config: ProjectConfig | None = None,
    seed: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Generate channel gains G[t, u] for a snapshot.

    Parameters
    ----------
    n_users : int
        Number of users in the snapshot.
    config : ProjectConfig, optional
        Global project configuration.
    seed : int, optional
        Random seed for reproducibility.
    normalize : bool
        Whether to normalize G by its maximum element.

    Returns
    -------
    G : np.ndarray
        Channel gain matrix of shape (n_transmitters, n_users).
    """
    if config is None:
        config = get_default_config()

    if n_users <= 0:
        raise ValueError("n_users must be positive.")

    sys_cfg = config.system
    rng = np.random.default_rng(seed)

    n_transmitters = sys_cfg.n_transmitters
    G = np.zeros((n_transmitters, n_users), dtype=float)

    for t in range(n_transmitters):
        is_haps = (t == n_transmitters - 1)

        for u in range(n_users):
            if not is_haps:
                # LEO-like link distance variation
                distance_km = sys_cfg.d_leo_km * rng.uniform(0.9, 1.1)
            else:
                # HAPS-like link distance variation
                distance_km = sys_cfg.d_haps_km * rng.uniform(0.8, 1.2)

            pl_db = path_loss_db(distance_km, sys_cfg.frequency_ghz)
            pl_linear = 10.0 ** (-pl_db / 10.0)

            fading_gain = rician_power_gain(rng, sys_cfg.rician_k)

            G[t, u] = pl_linear * fading_gain

    if normalize:
        g_max = np.max(G)
        if g_max > 0:
            G = G / g_max

    return G


def prune_candidate_links(
    G: np.ndarray,
    top_k_tx_per_user: int | None,
) -> np.ndarray:
    """
    Build a binary candidate mask M[t, u] indicating which TX-user links are
    retained for binary optimization.

    Parameters
    ----------
    G : np.ndarray
        Channel gain matrix, shape (n_transmitters, n_users).
    top_k_tx_per_user : int or None
        If None, keep all links. Otherwise, for each user keep only the top-k
        transmitters ranked by direct channel gain.

    Returns
    -------
    mask : np.ndarray
        Binary mask of shape (n_transmitters, n_users), with values in {0, 1}.
    """
    n_transmitters, n_users = G.shape

    if top_k_tx_per_user is None or top_k_tx_per_user >= n_transmitters:
        return np.ones_like(G, dtype=int)

    if top_k_tx_per_user <= 0:
        raise ValueError("top_k_tx_per_user must be positive or None.")

    mask = np.zeros_like(G, dtype=int)

    for u in range(n_users):
        ranked_tx = np.argsort(-G[:, u])  # descending by direct gain
        selected = ranked_tx[:top_k_tx_per_user]
        mask[selected, u] = 1

    return mask


def generate_snapshot(
    n_users: int,
    config: ProjectConfig | None = None,
    seed: int | None = None,
) -> dict:
    """
    Generate a complete snapshot dictionary containing channel matrix and
    optional candidate mask.

    Returns
    -------
    snapshot : dict
        {
            "G": channel gain matrix,
            "candidate_mask": binary candidate mask,
            "n_users": int,
            "n_transmitters": int,
            "seed": int or None,
        }
    """
    if config is None:
        config = get_default_config()

    G = generate_channel_gains(
        n_users=n_users,
        config=config,
        seed=seed,
        normalize=True,
    )

    candidate_mask = prune_candidate_links(
        G=G,
        top_k_tx_per_user=config.qubo.top_k_tx_per_user,
    )

    return {
        "G": G,
        "candidate_mask": candidate_mask,
        "n_users": n_users,
        "n_transmitters": config.system.n_transmitters,
        "seed": seed,
    }


if __name__ == "__main__":
    cfg = get_default_config()
    snapshot = generate_snapshot(n_users=5, config=cfg, seed=42)

    G = snapshot["G"]
    M = snapshot["candidate_mask"]

    print("Channel matrix G shape:", G.shape)
    print("Candidate mask shape:", M.shape)
    print("G =\n", G)
    print("Candidate mask =\n", M)
