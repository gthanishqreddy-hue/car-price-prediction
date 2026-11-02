"""Regression metrics and evaluation plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: works in CI and without a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config


def regression_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, MSE, RMSE and R2. Units: lakh INR (MSE in lakh^2)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_actual_vs_predicted(
    y_true: pd.Series, y_pred: np.ndarray, model_name: str,
    out_dir: Path = config.FIGURES_DIR,
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.75, edgecolor="white", s=50)
    lims = [0, max(float(np.max(y_true)), float(np.max(y_pred))) * 1.05]
    ax.plot(lims, lims, linestyle="--", color="grey", label="Perfect prediction")
    ax.set(xlim=lims, ylim=lims, xlabel="Actual selling price (lakh INR)",
           ylabel="Predicted selling price (lakh INR)",
           title=f"Actual vs predicted - {model_name} (test set)")
    ax.legend()
    return _save(fig, out_dir / "actual_vs_predicted.png")


def plot_residuals(
    y_true: pd.Series, y_pred: np.ndarray, model_name: str,
    out_dir: Path = config.FIGURES_DIR,
) -> Path:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(residuals, kde=True, ax=axes[0])
    axes[0].axvline(0, color="grey", linestyle="--")
    axes[0].set(title="Residual distribution (actual - predicted)",
                xlabel="Residual (lakh INR)")
    axes[1].scatter(y_pred, residuals, alpha=0.75, edgecolor="white", s=50)
    axes[1].axhline(0, color="grey", linestyle="--")
    axes[1].set(title="Residuals vs predicted", xlabel="Predicted price (lakh INR)",
                ylabel="Residual (lakh INR)")
    fig.suptitle(model_name)
    return _save(fig, out_dir / "residuals.png")


def plot_model_comparison(comparison: pd.DataFrame, out_dir: Path = config.FIGURES_DIR) -> Path:
    """Bar chart of CV RMSE (with std error bars) next to test RMSE per model."""
    df = comparison.sort_values("cv_RMSE_mean")
    x = np.arange(len(df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, df["cv_RMSE_mean"], width, yerr=df["cv_RMSE_std"],
           capsize=4, label="Cross-validation RMSE (train set)")
    ax.bar(x + width / 2, df["test_RMSE"], width, label="Hold-out test RMSE")
    ax.set_xticks(x, df["model"], rotation=10)
    ax.set(ylabel="RMSE (lakh INR, lower is better)", title="Model comparison")
    ax.legend()
    return _save(fig, out_dir / "model_comparison.png")


def model_based_importance(model: BaseEstimator) -> pd.Series | None:
    """Importances from the fitted model itself, in the *transformed* feature space.

    Tree models: impurity-based importances. Linear model: absolute coefficients
    (comparable because inputs are standardised). Returns None if unsupported.
    """
    pipeline = model.pipeline_
    feature_names = pipeline[:-1].get_feature_names_out()
    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values, kind = estimator.feature_importances_, "impurity"
    elif hasattr(estimator, "coef_"):
        values, kind = np.abs(np.ravel(estimator.coef_)), "coef"
    else:
        return None
    importance = pd.Series(values, index=feature_names).sort_values(ascending=False)
    importance.attrs["kind"] = kind
    return importance


def compute_permutation_importance(
    model: BaseEstimator, X: pd.DataFrame, y: pd.Series, n_repeats: int = 30,
) -> pd.DataFrame:
    """Model-agnostic importance on the *raw* input features.

    Shuffles one raw column at a time and measures how much test RMSE gets worse.
    Unlike impurity importance it is not biased towards high-cardinality
    features and works for every model type.
    """
    result = permutation_importance(
        model, X, y, scoring="neg_root_mean_squared_error",
        n_repeats=n_repeats, random_state=config.RANDOM_STATE,
    )
    return (
        pd.DataFrame(
            {"feature": X.columns, "importance_mean": result.importances_mean,
             "importance_std": result.importances_std}
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def plot_feature_importance(
    permutation_df: pd.DataFrame, model_importance: pd.Series | None, model_name: str,
    out_dir: Path = config.FIGURES_DIR,
) -> Path:
    n_panels = 2 if model_importance is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 4.5), squeeze=False)
    ax = axes[0, 0]
    df = permutation_df.sort_values("importance_mean")
    ax.barh(df["feature"], df["importance_mean"], xerr=df["importance_std"], capsize=3)
    ax.set(title="Permutation importance (test set, raw features)",
           xlabel="Increase in RMSE when shuffled (lakh INR)")
    if model_importance is not None:
        ax = axes[0, 1]
        mi = model_importance.sort_values()
        ax.barh(mi.index, mi.values, color="tab:orange")
        xlabel = ("|coefficient| (standardised inputs)"
                  if model_importance.attrs.get("kind") == "coef" else "Impurity importance")
        ax.set(title="Model-based importance (transformed features)", xlabel=xlabel)
    fig.suptitle(model_name)
    return _save(fig, out_dir / "feature_importance.png")
