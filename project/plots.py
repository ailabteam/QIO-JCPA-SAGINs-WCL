# =============================================================================
# File: plots.py
# Description:
# Plotting utilities for the refactored QIO-JLSPA project.
#
# This module generates:
#   1) Quality vs. system scale
#   2) Quality vs. runtime trade-off
#   3) Runtime scaling
#
# Inputs are expected to come from experiments.py, especially df_summary.
# =============================================================================

from __future__ import annotations

from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ProjectConfig, get_default_config


# -------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------
def _check_summary_df(df_summary: pd.DataFrame) -> None:
    if df_summary is None or df_summary.empty:
        raise ValueError("df_summary is empty. Run experiments first.")

    required_cols = {"n_users", "policy", "mean_sum_rate", "mean_runtime_ms"}
    missing = required_cols - set(df_summary.columns)
    if missing:
        raise ValueError(f"df_summary is missing required columns: {missing}")


def _select_policies(
    df_summary: pd.DataFrame,
    policies: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    if policies is None:
        return df_summary.copy()

    policies = list(policies)
    return df_summary[df_summary["policy"].isin(policies)].copy()


def _get_reference_quality_column(df_summary: pd.DataFrame) -> str:
    """
    Prefer exact-reference quality if available; otherwise fall back to
    best-available-reference quality.
    """
    if "mean_quality_pct_exact" in df_summary.columns and df_summary["mean_quality_pct_exact"].notna().any():
        return "mean_quality_pct_exact"
    if "mean_quality_pct_best_available" in df_summary.columns:
        return "mean_quality_pct_best_available"
    raise ValueError("No quality percentage column found in df_summary.")


def _safe_errorbar(series: pd.Series) -> np.ndarray:
    """
    Replace NaN std values with zeros for plotting.
    """
    arr = series.to_numpy(dtype=float)
    arr = np.nan_to_num(arr, nan=0.0)
    return arr


def _policy_sort_key(policy: str) -> tuple:
    """
    Stable display order for key policies.
    """
    order = {
        "ExactSmall": 0,
        "QIO-HighQuality": 1,
        "QIO-Default": 2,
        "LocalSearch": 3,
        "Greedy": 4,
    }
    return (order.get(policy, 999), policy)


# -------------------------------------------------------------------------
# Plot 1: Quality vs. system scale
# -------------------------------------------------------------------------
def plot_quality_vs_scale(
    df_summary: pd.DataFrame,
    config: ProjectConfig | None = None,
    policies: Optional[Iterable[str]] = None,
    save_path: Optional[str] = None,
    show_errorbars: bool = True,
) -> None:
    """
    Plot mean achieved sum-rate versus number of users.
    """
    if config is None:
        config = get_default_config()

    _check_summary_df(df_summary)
    df = _select_policies(df_summary, policies)

    if df.empty:
        raise ValueError("No data left after policy filtering.")

    plt.figure(figsize=(config.plot.figure_width, config.plot.figure_height))

    plot_policies = sorted(df["policy"].unique(), key=_policy_sort_key)

    for policy in plot_policies:
        d = df[df["policy"] == policy].sort_values("n_users")

        x = d["n_users"].to_numpy()
        y = d["mean_sum_rate"].to_numpy()

        if show_errorbars and "std_sum_rate" in d.columns:
            yerr = _safe_errorbar(d["std_sum_rate"])
            plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=policy)
        else:
            plt.plot(x, y, marker="o", label=policy)

    plt.xlabel("Number of Users")
    plt.ylabel("Mean Sum Rate")
    plt.title("Solution Quality vs. System Scale")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    if save_path is None:
        save_path = config.plot.quality_scale_filename
    plt.savefig(save_path, dpi=config.plot.dpi)
    plt.close()


# -------------------------------------------------------------------------
# Plot 2: Quality vs. runtime trade-off
# -------------------------------------------------------------------------
def plot_quality_runtime_tradeoff(
    df_summary: pd.DataFrame,
    target_n_users: int,
    config: ProjectConfig | None = None,
    policies: Optional[Iterable[str]] = None,
    save_path: Optional[str] = None,
    annotate_points: bool = True,
) -> None:
    """
    Plot quality (%) versus runtime (ms) for a chosen system scale.

    Quality is measured relative to:
      - ExactSmall if available,
      - otherwise best available policy in the snapshot summaries.
    """
    if config is None:
        config = get_default_config()

    _check_summary_df(df_summary)

    df = df_summary[df_summary["n_users"] == target_n_users].copy()
    df = _select_policies(df, policies)

    if df.empty:
        raise ValueError(f"No summary data found for n_users={target_n_users}.")

    quality_col = _get_reference_quality_column(df)

    plt.figure(figsize=(config.plot.figure_width, config.plot.figure_height))

    x = df["mean_runtime_ms"].to_numpy(dtype=float)
    y = df[quality_col].to_numpy(dtype=float)

    plt.scatter(x, y)

    if annotate_points:
        for _, row in df.iterrows():
            label = row["policy"]
            xx = float(row["mean_runtime_ms"])
            yy = float(row[quality_col])
            plt.annotate(label, (xx, yy), textcoords="offset points", xytext=(5, 5))

    # Sort by runtime so a trade-off curve is easier to read
    df_curve = df.sort_values("mean_runtime_ms")
    plt.plot(
        df_curve["mean_runtime_ms"].to_numpy(dtype=float),
        df_curve[quality_col].to_numpy(dtype=float),
    )

    plt.xscale("log")
    plt.xlabel("Mean Runtime (ms) [log scale]")
    plt.ylabel("Quality (%)")
    plt.title(f"Quality vs. Runtime Trade-off (n_users={target_n_users})")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path is None:
        save_path = config.plot.runtime_tradeoff_filename
    plt.savefig(save_path, dpi=config.plot.dpi)
    plt.close()


# -------------------------------------------------------------------------
# Plot 3: Runtime scaling
# -------------------------------------------------------------------------
def plot_runtime_scaling(
    df_summary: pd.DataFrame,
    config: ProjectConfig | None = None,
    policies: Optional[Iterable[str]] = None,
    save_path: Optional[str] = None,
    show_errorbars: bool = True,
) -> None:
    """
    Plot mean runtime versus number of users.
    """
    if config is None:
        config = get_default_config()

    _check_summary_df(df_summary)
    df = _select_policies(df_summary, policies)

    if df.empty:
        raise ValueError("No data left after policy filtering.")

    plt.figure(figsize=(config.plot.figure_width, config.plot.figure_height))

    plot_policies = sorted(df["policy"].unique(), key=_policy_sort_key)

    for policy in plot_policies:
        d = df[df["policy"] == policy].sort_values("n_users")

        x = d["n_users"].to_numpy()
        y = d["mean_runtime_ms"].to_numpy()

        if show_errorbars and "std_runtime_ms" in d.columns:
            yerr = _safe_errorbar(d["std_runtime_ms"])
            plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=policy)
        else:
            plt.plot(x, y, marker="o", label=policy)

    plt.xlabel("Number of Users")
    plt.ylabel("Mean Runtime (ms)")
    plt.title("Runtime Scaling vs. System Scale")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    if save_path is None:
        save_path = config.plot.scaling_runtime_filename
    plt.savefig(save_path, dpi=config.plot.dpi)
    plt.close()


# -------------------------------------------------------------------------
# Convenience entry point
# -------------------------------------------------------------------------
def generate_all_plots(
    df_summary: pd.DataFrame,
    config: ProjectConfig | None = None,
    target_n_users: Optional[int] = None,
    policies: Optional[Iterable[str]] = None,
) -> None:
    """
    Generate all standard plots from the summary DataFrame.
    """
    if config is None:
        config = get_default_config()

    _check_summary_df(df_summary)

    if target_n_users is None:
        target_n_users = int(df_summary["n_users"].max())

    plot_quality_vs_scale(
        df_summary=df_summary,
        config=config,
        policies=policies,
        save_path=config.plot.quality_scale_filename,
    )

    plot_quality_runtime_tradeoff(
        df_summary=df_summary,
        target_n_users=target_n_users,
        config=config,
        policies=policies,
        save_path=config.plot.runtime_tradeoff_filename,
    )

    plot_runtime_scaling(
        df_summary=df_summary,
        config=config,
        policies=policies,
        save_path=config.plot.scaling_runtime_filename,
    )


if __name__ == "__main__":
    cfg = get_default_config()

    # Example usage:
    #   python experiments.py
    # then load the generated CSV summary here if desired.
    summary_path = "results_summary.csv"

    try:
        df_summary = pd.read_csv(summary_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find '{summary_path}'. Run experiments.py first."
        )

    generate_all_plots(
        df_summary=df_summary,
        config=cfg,
        target_n_users=max(cfg.experiment.n_users_scales),
    )

    print("All plots generated successfully.")
