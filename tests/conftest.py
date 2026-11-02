"""Shared fixtures.

Unit tests run on a small *synthetic* frame with the real schema so they are
fast and do not depend on downloading the dataset. Integration tests that need
the real data or trained model are skipped automatically when those files are
absent (e.g. on a fresh clone).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src import config
from src.data_preprocessing import split_features_target
from src.model import build_model
from src.predict import ModelBundle
from src.train import feature_schema


def make_synthetic_cars(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Synthetic listings that follow a simple depreciation rule (test fixture only)."""
    rng = np.random.default_rng(seed)
    year = rng.integers(2005, 2019, n)
    present = rng.uniform(3, 30, n).round(2)
    kms = rng.integers(5_000, 150_000, n)
    age = 2019 - year
    price = (present * 0.9 ** age * rng.uniform(0.9, 1.1, n)).round(2)
    return pd.DataFrame({
        "Car_Name": [f"car{i}" for i in range(n)],
        "Year": year,
        "Selling_Price": price,
        "Present_Price": present,
        "Kms_Driven": kms,
        "Fuel_Type": rng.choice(["Petrol", "Diesel", "CNG"], n, p=[0.6, 0.35, 0.05]),
        "Seller_Type": rng.choice(["Dealer", "Individual"], n),
        "Transmission": rng.choice(["Manual", "Automatic"], n),
        "Owner": rng.choice([0, 1, 3], n, p=[0.85, 0.1, 0.05]),
    })


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_cars()


@pytest.fixture
def trained_bundle(synthetic_df: pd.DataFrame) -> ModelBundle:
    X, y = split_features_target(synthetic_df)
    model = build_model(RandomForestRegressor(n_estimators=30, random_state=0), "ratio").fit(X, y)
    metadata = {
        "model_name": "Random Forest (test)",
        "target_mode": "ratio",
        "sklearn_version": __import__("sklearn").__version__,
        "feature_schema": feature_schema(X, model),
        "metrics": {"test": {"MAE": 0.5, "RMSE": 0.7, "R2": 0.9, "MSE": 0.49},
                    "cv": {"RMSE": {"mean": 0.8, "std": 0.1}}},
        "trained_at_utc": "2026-01-01T00:00:00+00:00",
    }
    return ModelBundle(model=model, metadata=metadata)


@pytest.fixture
def saved_model_path(trained_bundle: ModelBundle, tmp_path: Path) -> Path:
    path = tmp_path / "model.joblib"
    joblib.dump({"model": trained_bundle.model, "metadata": trained_bundle.metadata}, path)
    return path


@pytest.fixture
def valid_vehicle() -> dict:
    return {"Year": 2015, "Present_Price": 9.5, "Kms_Driven": 40_000, "Fuel_Type": "Petrol",
            "Seller_Type": "Dealer", "Transmission": "Manual", "Owner": 0}


requires_real_data = pytest.mark.skipif(
    not config.DATA_PATH.exists(), reason="dataset not downloaded (python -m src.download_data)"
)
requires_trained_model = pytest.mark.skipif(
    not config.MODEL_PATH.exists(), reason="model not trained (python -m src.train)"
)
