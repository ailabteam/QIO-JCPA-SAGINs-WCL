# =============================================================================
# File: experiments.py
# Description:
# Experiment orchestration for the refactored QIO-JLSPA project.
#
# This module:
#   - generates snapshots across scales and seeds,
#   - runs enabled baselines on each snapshot,
#   - aggregates raw and summary results,
#   - computes quality ratios against the best available reference,
#   - computes exact-gap statistics whenever ExactSmall is available.
# =============================================================================

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from baselines import run_snapshot_baselines
from channel_model import generate_snapshot
from config import ProjectConfig, get_default_config


# -------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------
def _snapshot_seed(config: ProjectConfig, iteration_index: int) -> int:
    """
    Compute the seed used for a given experiment iteration.
    """
    exp_cfg = config.experiment
    return exp_cfg.seed_start + iteration_index * exp_cfg.seed_stride


def _records_from_snapshot_results(
    snapshot_results: List[Dict],
    n_users: int,
    seed: int,
) -> List[Dict]:
    """
    Convert a list of baseline result dictionaries into flat records.
    """
    records = []

    for res in snapshot_results:
        record = {
            "policy": res["policy"],
            "n_users": int(n_users),
            "seed": int(seed),
            "sum_rate": float(res["sum_rate"]),
            "runtime_ms": float(res["runtime_ms"]),
            "is_feasible": bool(res["is_feasible"]),
            "binary_objective": (
                np.nan if res["binary_objective"] is None else float(res["binary_objective"])
            ),
        }

        if "num_evaluated_assignments" in res:
            record["num_evaluated_assignments"] = int(res["num_evaluated_assignments"])
        else:
            record["num_evaluated_assignments"] = np.nan

        # Optional optimizer info
        power_info = res.get("power_info", None)
        if isinstance(power_info, dict):
            record["power_optimizer_success"] = power_info.get("optimizer_success", np.nan)
            record["power_optimizer_iterations"] = power_info.get("iterations", np.nan)
            record["rate_before_refinement"] = power_info.get("rate_before", np.nan)
            record["rate_after_refinement"] = power_info.get("rate_after", np.nan)
        else:
            record["power_optimizer_success"] = np.nan
            record["power_optimizer_iterations"] = np.nan
            record["rate_before_refinement"] = np.nan
            record["rate_after_refinement"] = np.nan

        records.append(record)

    return records


def _attach_reference_metrics(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    For each (n_users, seed), compute:
      - best_available_rate
      - quality_pct_best_available
      - exact_rate (if ExactSmall exists)
      - quality_pct_exact
      - exact_gap_pct

    The "best available reference" is the max sum-rate among all policies for
    the same snapshot.
    """
    if df_raw.empty:
        return df_raw.copy()

    df = df_raw.copy()

    # Best available reference within each snapshot
    best_available = (
        df.groupby(["n_users", "seed"])["sum_rate"]
        .max()
        .rename("best_available_rate")
        .reset_index()
    )
    df = df.merge(best_available, on=["n_users", "seed"], how="left")
    df["quality_pct_best_available"] = 100.0 * df["sum_rate"] / df["best_available_rate"]

    # Exact reference if ExactSmall exists for that snapshot
    exact_df = (
        df[df["policy"] == "ExactSmall"][["n_users", "seed", "sum_rate"]]
        .rename(columns={"sum_rate": "exact_rate"})
        .drop_duplicates()
    )

    df = df.merge(exact_df, on=["n_users", "seed"], how="left")

    df["quality_pct_exact"] = np.where(
        df["exact_rate"].notna(),
        100.0 * df["sum_rate"] / df["exact_rate"],
        np.nan,
    )

    df["exact_gap_pct"] = np.where(
        df["exact_rate"].notna(),
        100.0 * (df["exact_rate"] - df["sum_rate"]) / df["exact_rate"],
        np.nan,
    )

    return df


def _aggregate_summary(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw records into per-(n_users, policy) summary statistics.
    """
    if df_raw.empty:
        return pd.DataFrame()

    agg = (
        df_raw.groupby(["n_users", "policy"])
        .agg(
            mean_sum_rate=("sum_rate", "mean"),
            std_sum_rate=("sum_rate", "std"),
            mean_runtime_ms=("runtime_ms", "mean"),
            std_runtime_ms=("runtime_ms", "std"),
            mean_quality_pct_best_available=("quality_pct_best_available", "mean"),
            std_quality_pct_best_available=("quality_pct_best_available", "std"),
            mean_quality_pct_exact=("quality_pct_exact", "mean"),
            std_quality_pct_exact=("quality_pct_exact", "std"),
            mean_exact_gap_pct=("exact_gap_pct", "mean"),
            std_exact_gap_pct=("exact_gap_pct", "std"),
            feasibility_rate=("is_feasible", "mean"),
            mean_binary_objective=("binary_objective", "mean"),
            mean_optimizer_iterations=("power_optimizer_iterations", "mean"),
        )
        .reset_index()
        .sort_values(["n_users", "policy"])
    )

    return agg


# -------------------------------------------------------------------------
# Main experiment runners
# -------------------------------------------------------------------------
def run_multi_scale_experiments(
    config: ProjectConfig | None = None,
    refine_power: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run enabled baselines for all configured user scales and random seeds.

    Returns
    -------
    df_raw : pd.DataFrame
        One row per (snapshot, policy).
    df_summary : pd.DataFrame
        Aggregated summary by (n_users, policy).
    """
    if config is None:
        config = get_default_config()

    exp_cfg = config.experiment

    all_records: List[Dict] = []

    print("\n=== Running multi-scale experiments ===")

    for n_users in exp_cfg.n_users_scales:
        print(f"\n--- n_users = {n_users} ---")

        for it in range(exp_cfg.n_random_seeds):
            seed = _snapshot_seed(config, it)

            snapshot = generate_snapshot(
                n_users=n_users,
                config=config,
                seed=seed,
            )

            G = snapshot["G"]
            candidate_mask = snapshot["candidate_mask"]

            try:
                snapshot_results = run_snapshot_baselines(
                    G=G,
                    candidate_mask=candidate_mask,
                    config=config,
                    seed=seed,
                    refine_power=refine_power,
                )
            except Exception as exc:
                print(f"[Warning] Failed at n_users={n_users}, seed={seed}: {exc}")
                continue

            records = _records_from_snapshot_results(
                snapshot_results=snapshot_results,
                n_users=n_users,
                seed=seed,
            )
            all_records.extend(records)

            # Lightweight console log
            log_parts = []
            for rec in records:
                log_parts.append(
                    f"{rec['policy']}: R={rec['sum_rate']:.4f}, T={rec['runtime_ms']:.2f} ms"
                )
            print(f"seed={seed} | " + " | ".join(log_parts))

    df_raw = pd.DataFrame(all_records)
    if df_raw.empty:
        return df_raw, pd.DataFrame()

    df_raw = _attach_reference_metrics(df_raw)
    df_summary = _aggregate_summary(df_raw)

    return df_raw, df_summary


def build_snapshot_reference_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build a snapshot-level table with best available and exact references.

    Useful for debugging and writing reviewer responses.
    """
    if df_raw.empty:
        return pd.DataFrame()

    cols = [
        "n_users",
        "seed",
        "policy",
        "sum_rate",
        "runtime_ms",
        "best_available_rate",
        "quality_pct_best_available",
        "exact_rate",
        "quality_pct_exact",
        "exact_gap_pct",
    ]

    existing_cols = [c for c in cols if c in df_raw.columns]
    return df_raw[existing_cols].sort_values(["n_users", "seed", "policy"]).reset_index(drop=True)


def build_letter_table(df_summary: pd.DataFrame, target_n_users: int) -> pd.DataFrame:
    """
    Build a compact table for a chosen system scale suitable for paper reporting.

    Example use:
        target_n_users = 5
    """
    if df_summary.empty:
        return pd.DataFrame()

    df = df_summary[df_summary["n_users"] == target_n_users].copy()
    if df.empty:
        return df

    # Select compact columns for paper use
    cols = [
        "policy",
        "mean_sum_rate",
        "std_sum_rate",
        "mean_runtime_ms",
        "std_runtime_ms",
        "mean_quality_pct_best_available",
        "mean_quality_pct_exact",
        "mean_exact_gap_pct",
    ]
    existing_cols = [c for c in cols if c in df.columns]

    df = df[existing_cols].copy()
    return df.reset_index(drop=True)


def save_experiment_outputs(
    df_raw: pd.DataFrame,
    df_summary: pd.DataFrame,
    prefix: str = "results",
) -> None:
    """
    Save raw and summary experiment outputs to CSV.
    """
    if df_raw is not None and not df_raw.empty:
        df_raw.to_csv(f"{prefix}_raw.csv", index=False)

    if df_summary is not None and not df_summary.empty:
        df_summary.to_csv(f"{prefix}_summary.csv", index=False)


# -------------------------------------------------------------------------
# Script entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = get_default_config()

    df_raw, df_summary = run_multi_scale_experiments(
        config=cfg,
        refine_power=True,
    )

    if not df_raw.empty:
        print("\n=== Raw results preview ===")
        print(df_raw.head())

    if not df_summary.empty:
        print("\n=== Summary results preview ===")
        print(df_summary.head())

    save_experiment_outputs(df_raw, df_summary, prefix="results")

    target_n_users = max(cfg.experiment.n_users_scales)
    df_letter = build_letter_table(df_summary, target_n_users=target_n_users)

    if not df_letter.empty:
        df_letter.to_csv(f"letter_table_nusers_{target_n_users}.csv", index=False)
        print(f"\nSaved compact letter table for n_users={target_n_users}.")
