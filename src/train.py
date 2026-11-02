"""Train, compare, tune and persist the car price models.

Workflow (all model-selection decisions use the training split only):

1. Load + clean the data, hold out a stratified 20% test set.
2. For each algorithm, compare three target scales (raw price, log price,
   share of new price retained) with repeated K-fold CV; keep the best.
3. Tune the tree-based models (Linear Regression has no hyperparameters to tune)
   with GridSearchCV / RandomizedSearchCV.
4. Re-score every final candidate with repeated K-fold CV; pick the lowest
   mean CV RMSE. The test set is only used afterwards to *report* performance.
5. Save the full pipeline (feature engineering + preprocessing + model) with
   joblib, plus metrics, comparison table and evaluation figures.

Usage:
    python -m src.train            # full run
    python -m src.train --fast     # fewer search iterations (CI / smoke test)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    RepeatedKFold,
    cross_validate,
)
from sklearn.tree import DecisionTreeRegressor

from src import config
from src.data_preprocessing import load_clean_data, train_test_split_stratified
from src.evaluate import (
    compute_permutation_importance,
    model_based_importance,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_model_comparison,
    plot_residuals,
    regression_metrics,
)
from src.model import TARGET_MODES, CarPriceRegressor, build_model

logger = logging.getLogger(__name__)

SCORING = {
    "RMSE": "neg_root_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
    "MSE": "neg_mean_squared_error",
    "R2": "r2",
}


@dataclass
class ModelSpec:
    """An algorithm plus (optionally) how to tune it."""

    name: str
    estimator: BaseEstimator
    param_space: dict[str, list] = field(default_factory=dict)
    search: str = "none"  # "grid", "random" or "none"
    n_iter: int = 0


def get_model_specs(random_state: int = config.RANDOM_STATE, fast: bool = False) -> list[ModelSpec]:
    """Candidate algorithms and their hyperparameter search spaces.

    Search spaces deliberately include strong regularisation (shallow trees,
    larger leaves, subsampling) because ~160 training rows overfit easily.
    """
    p = "pipeline__model__"  # parameter prefix inside CarPriceRegressor
    return [
        ModelSpec("Linear Regression", LinearRegression()),
        ModelSpec(
            "Decision Tree",
            DecisionTreeRegressor(random_state=random_state),
            {
                f"{p}max_depth": [3, 4, 5, 6, 8, None],
                f"{p}min_samples_leaf": [1, 2, 4, 8],
                f"{p}min_samples_split": [2, 5, 10],
            },
            search="grid",
        ),
        ModelSpec(
            "Random Forest",
            RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1),
            {
                f"{p}n_estimators": [200, 300, 500],
                f"{p}max_depth": [None, 6, 10, 16],
                f"{p}min_samples_leaf": [1, 2, 3, 5],
                f"{p}max_features": [1.0, 0.7, 0.5, "sqrt"],
            },
            search="random",
            n_iter=8 if fast else 30,
        ),
        ModelSpec(
            "Gradient Boosting",
            GradientBoostingRegressor(random_state=random_state),
            {
                f"{p}n_estimators": [100, 200, 400, 800],
                f"{p}learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
                f"{p}max_depth": [2, 3, 4, 5],
                f"{p}subsample": [0.6, 0.8, 1.0],
                f"{p}min_samples_leaf": [1, 3, 5, 10],
            },
            search="random",
            n_iter=10 if fast else 40,
        ),
    ]


def repeated_cv_scores(
    model: CarPriceRegressor, X: pd.DataFrame, y: pd.Series, random_state: int,
    n_repeats: int = config.CV_REPEATS,
) -> dict[str, float]:
    """Mean/std of each metric over repeated K-fold CV (all in lakh INR)."""
    cv = RepeatedKFold(n_splits=config.CV_FOLDS, n_repeats=n_repeats,
                       random_state=random_state)
    res = cross_validate(model, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    out: dict[str, float] = {}
    for metric in SCORING:
        values = res[f"test_{metric}"]
        values = values if metric == "R2" else -values  # sklearn scorers are "higher is better"
        out[f"cv_{metric}_mean"] = float(np.mean(values))
        out[f"cv_{metric}_std"] = float(np.std(values))
    return out


def choose_target_mode(
    spec: ModelSpec, X: pd.DataFrame, y: pd.Series, random_state: int
) -> tuple[str, list[dict[str, Any]]]:
    """CV-compare every target scale for one algorithm (default hyperparameters)."""
    rows = []
    for mode in TARGET_MODES:
        scores = repeated_cv_scores(build_model(clone(spec.estimator), mode), X, y, random_state)
        rows.append({"model": spec.name, "target_mode": mode, **scores})
    best = min(rows, key=lambda r: r["cv_RMSE_mean"])
    return str(best["target_mode"]), rows


def tune(spec: ModelSpec, model: CarPriceRegressor, X: pd.DataFrame, y: pd.Series,
         random_state: int) -> tuple[CarPriceRegressor, dict[str, Any]]:
    """Hyperparameter search; returns the refitted best model and its params."""
    if spec.search == "none":
        return model.fit(X, y), {}
    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=random_state)
    common = {"scoring": "neg_root_mean_squared_error", "cv": cv, "n_jobs": -1, "refit": True}
    if spec.search == "grid":
        search = GridSearchCV(model, spec.param_space, **common)
    else:
        search = RandomizedSearchCV(model, spec.param_space, n_iter=spec.n_iter,
                                    random_state=random_state, **common)
    search.fit(X, y)
    params = {k.replace("pipeline__model__", ""): v for k, v in search.best_params_.items()}
    logger.info("%s best params: %s (search CV RMSE %.3f)", spec.name, params,
                -search.best_score_)
    return search.best_estimator_, params


def feature_schema(X_train: pd.DataFrame, model: CarPriceRegressor) -> dict[str, Any]:
    """What the model saw during training - used to validate inputs at inference."""
    return {
        "numeric": {
            col: {"min": float(X_train[col].min()), "max": float(X_train[col].max())}
            for col in config.NUMERIC_INPUTS
        },
        "categorical": {
            col: sorted(X_train[col].astype(str).unique().tolist())
            for col in config.CATEGORICAL_INPUTS
        },
        "reference_year": int(model.pipeline_.named_steps["features"].reference_year_),
    }


def save_model(bundle: dict[str, Any], path: Path = config.MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    logger.info("Saved model bundle to %s", path)
    return path


def run_training(data_path: Path = config.DATA_PATH, fast: bool = False,
                 random_state: int = config.RANDOM_STATE) -> dict[str, Any]:
    start = time.perf_counter()
    df, cleaning = load_clean_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        df, random_state=random_state
    )
    logger.info("Train rows: %d, test rows: %d", len(X_train), len(X_test))

    # Different seeds for search CV and selection CV reduce the optimism of
    # choosing hyperparameters and the winner on exactly the same folds.
    selection_seed = random_state + 1

    ablation_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    fitted: dict[str, CarPriceRegressor] = {}

    for spec in get_model_specs(random_state, fast=fast):
        logger.info("=== %s ===", spec.name)
        target_mode, rows = choose_target_mode(spec, X_train, y_train, selection_seed)
        ablation_rows.extend(rows)
        model = build_model(clone(spec.estimator), target_mode=target_mode)
        best_model, params = tune(spec, model, X_train, y_train, random_state)

        cv_scores = repeated_cv_scores(clone(best_model), X_train, y_train, selection_seed)
        train_metrics = regression_metrics(y_train, best_model.predict(X_train))
        test_metrics = regression_metrics(y_test, best_model.predict(X_test))
        fitted[spec.name] = best_model
        comparison_rows.append({
            "model": spec.name,
            "target_mode": target_mode,
            **cv_scores,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
            "best_params": json.dumps(params, default=str),
        })
        logger.info("%s: CV RMSE %.3f +/- %.3f | test RMSE %.3f | test R2 %.3f",
                    spec.name, cv_scores["cv_RMSE_mean"], cv_scores["cv_RMSE_std"],
                    test_metrics["RMSE"], test_metrics["R2"])

    comparison = pd.DataFrame(comparison_rows).sort_values("cv_RMSE_mean").reset_index(drop=True)
    best_name = str(comparison.loc[0, "model"])  # selected on CV, not on the test set
    best_model = fitted[best_name]
    best_row = comparison.loc[0].to_dict()
    logger.info("Selected model: %s", best_name)

    # ---- artifacts --------------------------------------------------------
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(config.COMPARISON_PATH, index=False)
    pd.DataFrame(ablation_rows).to_csv(config.ARTIFACTS_DIR / "target_transform_ablation.csv",
                                       index=False)

    y_pred_test = best_model.predict(X_test)
    plot_actual_vs_predicted(y_test, y_pred_test, best_name)
    plot_residuals(y_test, y_pred_test, best_name)
    plot_model_comparison(comparison)
    perm = compute_permutation_importance(best_model, X_test, y_test)
    perm.to_csv(config.ARTIFACTS_DIR / "permutation_importance.csv", index=False)
    plot_feature_importance(perm, model_based_importance(best_model), best_name)

    metrics = {
        "selected_model": best_name,
        "selection_criterion": "lowest mean RMSE over 5x3 repeated K-fold CV on the training set",
        "target_mode": str(best_row["target_mode"]),
        "best_params": json.loads(best_row["best_params"]),
        "cv": {m: {"mean": best_row[f"cv_{m}_mean"], "std": best_row[f"cv_{m}_std"]}
               for m in SCORING},
        "train": {m: best_row[f"train_{m}"] for m in SCORING},
        "test": {m: best_row[f"test_{m}"] for m in SCORING},
        "units": "lakh INR (1 lakh = 100,000 INR); MSE in lakh^2",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "cleaning": cleaning.to_dict(),
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    bundle = {
        "model": best_model,
        "metadata": {
            "model_name": best_name,
            "trained_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__,
            "input_features": config.INPUT_FEATURES,
            "target": config.TARGET,
            "target_unit": "lakh INR",
            "target_mode": str(best_row["target_mode"]),
            "feature_schema": feature_schema(X_train, best_model),
            "metrics": {"cv": metrics["cv"], "test": metrics["test"]},
            "random_state": random_state,
        },
    }
    save_model(bundle)
    logger.info("Training finished in %.1fs", time.perf_counter() - start)
    return {"metrics": metrics, "comparison": comparison}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train car price regression models.")
    parser.add_argument("--data", type=Path, default=config.DATA_PATH,
                        help="Path to the CarDekho 'car data.csv' file.")
    parser.add_argument("--fast", action="store_true",
                        help="Use fewer hyperparameter-search iterations.")
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    args = parse_args()
    result = run_training(args.data, fast=args.fast, random_state=args.seed)
    cols = ["model", "target_mode", "cv_RMSE_mean", "cv_RMSE_std", "cv_R2_mean",
            "train_RMSE", "test_MAE", "test_RMSE", "test_R2"]
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(result["comparison"][cols].to_string(index=False))
    print(f"\nSelected model: {result['metrics']['selected_model']}")


if __name__ == "__main__":
    main()
