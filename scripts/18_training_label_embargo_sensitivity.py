#!/usr/bin/env python3
"""Rolling-origin sensitivity with a 90-day declaration-maturity embargo on training labels.

The script is a sensitivity analysis, not a replacement of the primary specification.
A project's final static target is admitted to a training sample only when
max(declaration availability) + 90 days precedes the relevant model-training cutoff.
Inner TimeSeriesSplit folds are purged by the same rule at each validation-block start.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000]
Y = "outcome_log_B_over_C"
MODELS = {
    "B1_card": ["log_planned_h", "log_candidate_pool", "log1p_initial_leaf_tasks", "observed_team_size", "pair_count"],
    "I1_exp": ["log_planned_h", "log_candidate_pool", "log1p_initial_leaf_tasks", "observed_team_size", "pair_count", "mean_log_prior_projects"],
    "I2_exp_collab": ["log_planned_h", "log_candidate_pool", "log1p_initial_leaf_tasks", "observed_team_size", "pair_count", "mean_log_prior_projects", "mean_log_prior_collaborators"],
    "I4_exp_collab_EB": ["log_planned_h", "log_candidate_pool", "log1p_initial_leaf_tasks", "observed_team_size", "pair_count", "mean_log_prior_projects", "mean_log_prior_collaborators", "mean_eb_resource_tau5"],
    "IG_main": ["log_planned_h", "log_candidate_pool", "log1p_initial_leaf_tasks", "observed_team_size", "mean_log_prior_projects", "mean_eb_resource_tau5"],
}


def fit_ridge(data: pd.DataFrame, features: list[str], alpha: float) -> Pipeline:
    model = Pipeline([("standardize", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    model.fit(data[features], data[Y])
    return model


def select_alpha_purged(train: pd.DataFrame, features: list[str]) -> tuple[float, float, list[dict]]:
    train = train.sort_values("project_created_dt").reset_index(drop=True)
    splits = list(TimeSeriesSplit(4).split(train))
    best: tuple[float, float, list[dict]] | None = None
    for alpha in ALPHAS:
        fold_rows = []
        fold_mae = []
        for fold, (train_index, validation_index) in enumerate(splits, start=1):
            validation = train.iloc[validation_index]
            cutoff = validation["project_created_dt"].min()
            candidate_train = train.iloc[train_index]
            purged_train = candidate_train[candidate_train["outcome_known_dt"] < cutoff]
            if len(purged_train) < 20:
                raise RuntimeError(f"Too few purged training rows in fold {fold}: {len(purged_train)}")
            model = fit_ridge(purged_train, features, alpha)
            prediction = model.predict(validation[features])
            mae = float(mean_absolute_error(validation[Y], prediction))
            fold_mae.append(mae)
            fold_rows.append({
                "fold": fold,
                "alpha": alpha,
                "validation_start": cutoff.isoformat(),
                "n_train_before_purge": int(len(candidate_train)),
                "n_train_after_purge": int(len(purged_train)),
                "n_validation": int(len(validation)),
                "mae": mae,
            })
        score = float(np.mean(fold_mae))
        if best is None or score < best[0]:
            best = (score, alpha, fold_rows)
    assert best is not None
    return best[1], best[0], best[2]


def cluster_bootstrap(predictions: pd.DataFrame, model: str, base: str,
                      seed: int, draws: int) -> dict:
    selected = predictions[predictions["model"] == model][["project_code", "team_id", Y, "prediction"]].rename(columns={"prediction": "model_prediction"})
    reference = predictions[predictions["model"] == base][["project_code", "prediction"]].rename(columns={"prediction": "base_prediction"})
    paired = selected.merge(reference, on="project_code", validate="one_to_one")
    paired["model_abs_error"] = (paired[Y] - paired["model_prediction"]).abs()
    paired["base_abs_error"] = (paired[Y] - paired["base_prediction"]).abs()
    grouped = paired.groupby("team_id", sort=False).agg(
        n=("project_code", "size"),
        model_error=("model_abs_error", "sum"),
        base_error=("base_abs_error", "sum"),
    ).reset_index()
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(grouped), size=(draws, len(grouped)))
    n = grouped["n"].to_numpy(float)[ix].sum(axis=1)
    delta = (
        grouped["model_error"].to_numpy(float)[ix].sum(axis=1)
        - grouped["base_error"].to_numpy(float)[ix].sum(axis=1)
    ) / n
    observed = float(paired["model_abs_error"].mean() - paired["base_abs_error"].mean())
    return {
        "model": model,
        "base": base,
        "delta_mae": observed,
        "ci2.5": float(np.quantile(delta, 0.025)),
        "median": float(np.median(delta)),
        "ci97.5": float(np.quantile(delta, 0.975)),
        "p_improve": float(np.mean(delta < 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-features", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--embargo-days", type=int, default=90)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    args = parser.parse_args()
    if args.out_dir is None or args.project_features is None:
        root = Path(__file__).resolve().parents[1]
        import os
        stagef = Path(os.environ.get("GRYZZLY_STAGEF_DIR", root / "work" / "stageF"))
        stageg = Path(os.environ.get("GRYZZLY_STAGEG_DIR", root / "work" / "stageG"))
        args.project_features = args.project_features or (stagef / "intensive_eb_project_features.pkl")
        args.out_dir = args.out_dir or stageg
    args.out_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_pickle(args.project_features).copy()
    required = {"project_code", "team_id", "project_created_dt", "max_avail_ns", Y}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    features["year"] = features["project_created_dt"].dt.year
    embargo_ns = int(pd.Timedelta(days=args.embargo_days).value)
    features["outcome_known_dt"] = pd.to_datetime(
        features["max_avail_ns"].astype("int64") + embargo_ns, utc=True
    )

    result_rows = []
    prediction_rows = []
    inner_rows = []
    for year in [2022, 2023, 2024]:
        cutoff = pd.Timestamp(f"{year}-01-01", tz="UTC")
        chronological_train = features[features["project_created_dt"] < cutoff]
        train = chronological_train[chronological_train["outcome_known_dt"] < cutoff].sort_values("project_created_dt")
        test = features[features["year"] == year].sort_values("project_created_dt")
        for model_name, model_features in MODELS.items():
            alpha, inner_mae, folds = select_alpha_purged(train, model_features)
            model = fit_ridge(train, model_features, alpha)
            prediction = model.predict(test[model_features])
            result_rows.append({
                "year": year,
                "model": model_name,
                "alpha": alpha,
                "inner_cv_mae": inner_mae,
                "n_train_chronological": int(len(chronological_train)),
                "n_train_after_embargo": int(len(train)),
                "n_excluded_by_embargo": int(len(chronological_train) - len(train)),
                "n_test": int(len(test)),
                "mae": float(mean_absolute_error(test[Y], prediction)),
                "rmse": float(mean_squared_error(test[Y], prediction) ** 0.5),
                "r2": float(r2_score(test[Y], prediction)),
            })
            for fold_row in folds:
                inner_rows.append({"year": year, "model": model_name, **fold_row})
            output = test[["project_code", "team_id", "project_created_dt", Y]].copy()
            output["year"] = year
            output["model"] = model_name
            output["prediction"] = prediction
            prediction_rows.append(output)

    results = pd.DataFrame(result_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    aggregate_rows = []
    for model_name, group in predictions.groupby("model"):
        aggregate_rows.append({
            "model": model_name,
            "n_oof": int(len(group)),
            "mae": float(mean_absolute_error(group[Y], group["prediction"])),
            "rmse": float(mean_squared_error(group[Y], group["prediction"]) ** 0.5),
            "r2": float(r2_score(group[Y], group["prediction"])),
        })
    aggregate = pd.DataFrame(aggregate_rows).sort_values("mae")

    contrasts = [
        ("I1_exp", "B1_card"),
        ("I4_exp_collab_EB", "I2_exp_collab"),
        ("IG_main", "B1_card"),
        ("IG_main", "I2_exp_collab"),
    ]
    bootstrap = pd.DataFrame([
        cluster_bootstrap(predictions, model, base, args.bootstrap_seed + i, args.bootstrap_draws)
        for i, (model, base) in enumerate(contrasts)
    ])

    results.to_csv(args.out_dir / "training_label_embargo_folds.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(args.out_dir / "training_label_embargo_inner_folds.csv", index=False)
    aggregate.to_csv(args.out_dir / "training_label_embargo_aggregate.csv", index=False)
    bootstrap.to_csv(args.out_dir / "training_label_embargo_bootstrap.csv", index=False)
    predictions.to_csv(args.out_dir / "training_label_embargo_predictions.csv", index=False)

    summary = {
        "embargo_days": args.embargo_days,
        "aggregate": aggregate.to_dict("records"),
        "contrasts": bootstrap.to_dict("records"),
        "outer_training_counts": results[["year", "n_train_chronological", "n_train_after_embargo", "n_excluded_by_embargo", "n_test"]].drop_duplicates().to_dict("records"),
    }
    with open(args.out_dir / "training_label_embargo_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(aggregate.to_string(index=False))
    print("\nContrasts\n", bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
