#!/usr/bin/env python3
"""Corrected formation validation for the Gryzzly intensive-game study.

Inputs
------
formation_candidate_scores.csv with columns:
project_code, year, user_code, is_observed_member, q, eb_resource, prior_projects.

Corrections relative to Stage G v01
-----------------------------------
1. Enumerate all feasible same-size coalitions when C(N,k) <= --exact-limit.
2. Use deterministic Monte Carlo otherwise.
3. Compare floating-point coalition means with a scale-aware tolerance.
4. Treat an observed team as exactly optimal when it is any member of the
   possibly non-unique set of top-k optimal coalitions.
5. Report maximum Jaccard similarity over all tied optimal coalitions.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def tolerance(values: np.ndarray, reference: float) -> float:
    scale = max(1.0, abs(float(reference)), float(np.max(np.abs(values))))
    return 1e-12 * scale


def exact_or_mc_means(values: np.ndarray, k: int, project_code: int,
                      exact_limit: int, mc_draws: int) -> tuple[np.ndarray, str, int]:
    n = len(values)
    n_combinations = math.comb(n, k)
    if n_combinations <= exact_limit:
        means = np.fromiter(
            (float(np.mean(values[list(ix)])) for ix in itertools.combinations(range(n), k)),
            dtype=float,
            count=n_combinations,
        )
        return means, "exact", n_combinations

    rng = np.random.default_rng(1_000_003 + int(project_code))
    means = np.empty(mc_draws, dtype=float)
    for draw in range(mc_draws):
        means[draw] = float(values[rng.choice(n, size=k, replace=False)].mean())
    return means, "monte_carlo", n_combinations


def tie_aware_optimum(values: np.ndarray, observed_mask: np.ndarray, k: int) -> dict:
    sorted_values = np.sort(values)[::-1]
    cutoff = float(sorted_values[k - 1])
    tol = tolerance(values, cutoff)

    mandatory = values > cutoff + tol
    tied = np.abs(values - cutoff) <= tol
    mandatory_count = int(mandatory.sum())
    remaining_slots = k - mandatory_count

    observed_mandatory = int(np.sum(observed_mask & mandatory))
    observed_tied = int(np.sum(observed_mask & tied))
    max_overlap = observed_mandatory + min(remaining_slots, observed_tied)
    max_jaccard = max_overlap / (2 * k - max_overlap)

    return {
        "optimal_mean_q": float(sorted_values[:k].mean()),
        "optimal_cutoff_q": cutoff,
        "boundary_tie": bool(int(tied.sum()) > remaining_slots),
        "observed_is_any_optimal": bool(max_overlap == k),
        "max_overlap_with_optimal": int(max_overlap),
        "max_jaccard_with_optimal": float(max_jaccard),
    }


def bootstrap_workspace_means(projects: pd.DataFrame, seed: int, draws: int) -> dict:
    grouped = projects.groupby("team_id", sort=False).agg(
        n=("project_code", "size"),
        sum_gap=("observed_minus_random_expectation", "sum"),
        sum_optimal_gap=("optimal_gap", "sum"),
        sum_percentile=("observed_random_percentile", "sum"),
    ).reset_index()
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(grouped), size=(draws, len(grouped)))
    denominators = grouped["n"].to_numpy(float)[ix].sum(axis=1)
    gap = grouped["sum_gap"].to_numpy(float)[ix].sum(axis=1) / denominators
    optimal_gap = grouped["sum_optimal_gap"].to_numpy(float)[ix].sum(axis=1) / denominators
    percentile = grouped["sum_percentile"].to_numpy(float)[ix].sum(axis=1) / denominators
    return {
        "ci_observed_minus_random": [float(x) for x in np.quantile(gap, [0.025, 0.975])],
        "ci_optimal_gap": [float(x) for x in np.quantile(optimal_gap, [0.025, 0.975])],
        "ci_mean_random_percentile": [float(x) for x in np.quantile(percentile, [0.025, 0.975])],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-scores", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--exact-limit", type=int, default=2000)
    parser.add_argument("--mc-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    args = parser.parse_args()

    if args.out_dir is None or args.candidate_scores is None:
        root = Path(__file__).resolve().parents[1]
        stageg = Path(__import__("os").environ.get("GRYZZLY_STAGEG_DIR", root / "work" / "stageG"))
        args.out_dir = args.out_dir or stageg
        args.candidate_scores = args.candidate_scores or (stageg / "formation_candidate_scores.csv")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidate = pd.read_csv(args.candidate_scores)
    required = {
        "project_code", "year", "user_code", "is_observed_member", "q"
    }
    missing = required.difference(candidate.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows: list[dict] = []
    for project_code, group in candidate.groupby("project_code", sort=True):
        values = group["q"].to_numpy(float)
        observed_mask = group["is_observed_member"].to_numpy(int) == 1
        k = int(observed_mask.sum())
        n = int(len(group))
        if k < 2 or n < k:
            continue

        observed_mean = float(values[observed_mask].mean())
        pool_mean = float(values.mean())
        tol = tolerance(values, observed_mean)
        means, method, n_combinations = exact_or_mc_means(
            values, k, int(project_code), args.exact_limit, args.mc_draws
        )
        percentile = float(np.mean(means <= observed_mean + tol))
        if method == "exact":
            p_upper = float(np.mean(means >= observed_mean - tol))
        else:
            p_upper = float((1 + np.sum(means >= observed_mean - tol)) / (len(means) + 1))

        optimum = tie_aware_optimum(values, observed_mask, k)
        rows.append({
            "project_code": int(project_code),
            "year": int(group["year"].iloc[0]),
            "team_id": group["team_id"].iloc[0] if "team_id" in group.columns else "",
            "observed_n": k,
            "pool_n": n,
            "n_feasible_coalitions": int(n_combinations),
            "percentile_method": method,
            "observed_mean_q": observed_mean,
            "pool_mean_q": pool_mean,
            "optimal_mean_q": optimum["optimal_mean_q"],
            "observed_minus_random_expectation": observed_mean - pool_mean,
            "optimal_gap": optimum["optimal_mean_q"] - observed_mean,
            "observed_random_percentile": percentile,
            "random_p_upper": p_upper,
            "optimal_cutoff_q": optimum["optimal_cutoff_q"],
            "boundary_tie": optimum["boundary_tie"],
            "observed_is_any_optimal": optimum["observed_is_any_optimal"],
            "max_overlap_with_optimal": optimum["max_overlap_with_optimal"],
            "max_jaccard_with_optimal": optimum["max_jaccard_with_optimal"],
        })

    projects = pd.DataFrame(rows).sort_values(["year", "project_code"]).reset_index(drop=True)
    # Restore team_id from an optional mapping when it was not carried in candidate scores.
    # Current canonical candidate-scores file lacks team_id, so merge from the previous
    # project-level file if supplied alongside it.
    if (projects["team_id"] == "").all():
        legacy = args.candidate_scores.parent / "formation_validation_projects.csv"
        if legacy.exists():
            mapping = pd.read_csv(legacy, usecols=["project_code", "team_id"]).drop_duplicates()
            projects = projects.drop(columns="team_id").merge(mapping, on="project_code", how="left", validate="one_to_one")
        else:
            raise FileNotFoundError(
                "team_id is absent from candidate scores and formation_validation_projects.csv "
                "was not found beside the input; workspace-cluster bootstrap cannot be computed."
            )

    boot = bootstrap_workspace_means(projects, args.bootstrap_seed, args.bootstrap_draws)
    summary = {
        "projects": int(len(projects)),
        "workspaces": int(projects["team_id"].nunique()),
        "exact_enumeration_projects": int((projects["percentile_method"] == "exact").sum()),
        "monte_carlo_projects": int((projects["percentile_method"] == "monte_carlo").sum()),
        "share_observed_above_pool_mean": float((projects["observed_minus_random_expectation"] > 0).mean()),
        "mean_observed_minus_random": float(projects["observed_minus_random_expectation"].mean()),
        "median_observed_random_percentile": float(projects["observed_random_percentile"].median()),
        "share_observed_top_quartile_random": float((projects["observed_random_percentile"] >= 0.75).mean()),
        "share_observed_exact_any_optimal": float(projects["observed_is_any_optimal"].mean()),
        "projects_with_boundary_ties": int(projects["boundary_tie"].sum()),
        "median_max_jaccard_optimal": float(projects["max_jaccard_with_optimal"].median()),
        "mean_optimal_gap": float(projects["optimal_gap"].mean()),
        "median_optimal_gap": float(projects["optimal_gap"].median()),
        **boot,
    }

    percentile_points = projects["observed_random_percentile"].to_numpy(float) * 100
    bin_index = np.floor((percentile_points + 1e-12) / 5).astype(int)
    bin_index = np.clip(bin_index, 0, 19)
    bin_counts = np.bincount(bin_index, minlength=20)
    bins = pd.DataFrame({
        "lower_percentile": list(range(0, 100, 5)),
        "upper_percentile": list(range(5, 105, 5)),
        "interval": [f"{lo}\u2013{lo+5}" for lo in range(0, 100, 5)],
        "project_count": bin_counts.astype(int),
    })

    projects.to_csv(args.out_dir / "formation_validation_projects_corrected.csv", index=False)
    bins.to_csv(args.out_dir / "figure2_bins_corrected.csv", index=False)
    with open(args.out_dir / "formation_validation_summary_corrected.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    # Canonical corrected aliases and Stage G summary update.
    projects.to_csv(args.out_dir / "formation_validation_projects.csv", index=False)
    with open(args.out_dir / "formation_validation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    stageg_summary_path = args.out_dir / "stageG_summary.json"
    if stageg_summary_path.exists():
        stageg_summary = json.loads(stageg_summary_path.read_text(encoding="utf-8"))
        stageg_summary["formation"] = summary
        stageg_summary_path.write_text(json.dumps(stageg_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nFigure 2 bins:\n", bins.to_string(index=False))


if __name__ == "__main__":
    main()
