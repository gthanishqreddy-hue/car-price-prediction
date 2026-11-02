"""Exploratory data analysis: data-quality report and figures.

EDA is run on the *training split only*. Looking at test-set relationships
before modelling is a subtle form of leakage: decisions made after seeing them
(which features to engineer, which transforms to apply) would make the final
test score optimistic.

Usage:
    python -m src.eda
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.data_preprocessing import (
    CarFeatureEngineer,
    clean_data,
    iqr_outlier_summary,
    load_raw_data,
    train_test_split_stratified,
)

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def build_eda_frame(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Training features + engineered features + target, for plotting."""
    engineered = CarFeatureEngineer().fit(X_train).transform(X_train)
    engineered["Year"] = X_train["Year"]
    engineered[config.TARGET] = y_train
    return engineered


def plot_price_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df[config.TARGET], kde=True, ax=axes[0])
    axes[0].set(title="Selling price distribution", xlabel="Selling price (lakh INR)")
    sns.histplot(np.log1p(df[config.TARGET]), kde=True, ax=axes[1], color="tab:green")
    axes[1].set(title="log(1 + selling price)", xlabel="log1p(selling price)")
    return _save(fig, out_dir, "price_distribution.png")


def plot_age_vs_price(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=df, x="Car_Age", y=config.TARGET, ax=ax, color="tab:blue")
    ax.set(title="Vehicle age vs selling price", xlabel="Vehicle age (years)",
           ylabel="Selling price (lakh INR)", yscale="log")
    return _save(fig, out_dir, "age_vs_price.png")


def plot_kms_vs_price(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(data=df, x="Kms_Driven", y=config.TARGET, hue="Car_Age",
                    palette="viridis", ax=ax)
    ax.set(title="Kilometres driven vs selling price", xlabel="Kilometres driven",
           ylabel="Selling price (lakh INR)", yscale="log")
    return _save(fig, out_dir, "kms_vs_price.png")


def plot_present_vs_selling(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(data=df, x="Present_Price", y=config.TARGET, hue="Car_Age",
                    palette="viridis", ax=ax)
    ax.set(title="Current ex-showroom price vs selling price",
           xlabel="Present (ex-showroom) price (lakh INR)",
           ylabel="Selling price (lakh INR)", xscale="log", yscale="log")
    return _save(fig, out_dir, "present_vs_selling_price.png")


def plot_fuel_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    order = df["Fuel_Type"].value_counts().index
    sns.countplot(data=df, x="Fuel_Type", order=order, ax=axes[0])
    axes[0].set(title="Fuel type distribution", xlabel="Fuel type", ylabel="Number of cars")
    sns.boxplot(data=df, x="Fuel_Type", y=config.TARGET, order=order, ax=axes[1])
    axes[1].set(title="Selling price by fuel type", xlabel="Fuel type",
                ylabel="Selling price (lakh INR)", yscale="log")
    return _save(fig, out_dir, "fuel_type_distribution.png")


def plot_transmission_comparison(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.boxplot(data=df, x="Transmission", y=config.TARGET, ax=axes[0])
    axes[0].set(title="Selling price by transmission", ylabel="Selling price (lakh INR)",
                yscale="log")
    # Retained value = selling / present price: isolates depreciation from car segment.
    retained = df.assign(Retained=df[config.TARGET] / df["Present_Price"])
    sns.boxplot(data=retained, x="Transmission", y="Retained", ax=axes[1])
    axes[1].set(title="Share of new price retained", ylabel="Selling price / present price")
    return _save(fig, out_dir, "transmission_comparison.png")


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    numeric = df[["Present_Price", "Kms_Driven", "Car_Age", "Kms_Per_Year", "Owner",
                  config.TARGET]]
    # Spearman (rank) correlation: robust to the skew and outliers in prices/kms.
    corr = numeric.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Spearman correlation (training set)")
    return _save(fig, out_dir, "correlation_heatmap.png")


def summarise(raw: pd.DataFrame, clean: pd.DataFrame, cleaning_report: dict,
              eda_df: pd.DataFrame) -> dict:
    """Machine-readable data-quality summary written to artifacts/."""
    return {
        "raw_shape": list(raw.shape),
        "raw_missing_values": {k: int(v) for k, v in raw.isna().sum().items()},
        "raw_duplicate_rows": int(raw.duplicated().sum()),
        "raw_dtypes": {k: str(v) for k, v in raw.dtypes.items()},
        "cleaning": cleaning_report,
        "clean_shape": list(clean.shape),
        "category_counts_clean": {
            col: {str(k): int(v) for k, v in clean[col].value_counts().items()}
            for col in [*config.CATEGORICAL_INPUTS, "Owner"]
        },
        "numeric_summary_clean": json.loads(
            clean[[config.TARGET, *config.NUMERIC_INPUTS]].describe().round(3).to_json()
        ),
        "iqr_outliers_train": iqr_outlier_summary(
            eda_df, [config.TARGET, "Present_Price", "Kms_Driven", "Car_Age"]
        ),
        "spearman_with_target_train": {
            k: round(float(v), 3)
            for k, v in eda_df[["Present_Price", "Kms_Driven", "Car_Age", "Kms_Per_Year",
                                "Owner", config.TARGET]]
            .corr(method="spearman")[config.TARGET].drop(config.TARGET).items()
        },
    }


def run_eda(data_path: Path = config.DATA_PATH, out_dir: Path = config.FIGURES_DIR) -> dict:
    raw = load_raw_data(data_path)
    clean, report = clean_data(raw)
    X_train, _, y_train, _ = train_test_split_stratified(clean)
    eda_df = build_eda_frame(X_train, y_train)

    figures = [
        plot_price_distribution(eda_df, out_dir),
        plot_age_vs_price(eda_df, out_dir),
        plot_kms_vs_price(eda_df, out_dir),
        plot_present_vs_selling(eda_df, out_dir),
        plot_fuel_distribution(eda_df, out_dir),
        plot_transmission_comparison(eda_df, out_dir),
        plot_correlation_heatmap(eda_df, out_dir),
    ]
    summary = summarise(raw, clean, report.to_dict(), eda_df)
    summary["figures"] = [p.name for p in figures]

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_QUALITY_PATH.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %d figures to %s", len(figures), out_dir)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    summary = run_eda()
    print(json.dumps({k: summary[k] for k in ("cleaning", "spearman_with_target_train",
                                              "iqr_outliers_train", "figures")}, indent=2))


if __name__ == "__main__":
    main()
