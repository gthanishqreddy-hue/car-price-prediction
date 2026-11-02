"""Central configuration: paths, column names and reproducibility settings.

All paths are resolved relative to the project root so the code runs the same
way on any machine (no hardcoded user-specific paths).
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# The dataset location can be overridden with an environment variable,
# e.g. CAR_DATA_PATH=/some/where/car_data.csv python -m src.train
DATA_PATH: Path = Path(
    os.environ.get("CAR_DATA_PATH", PROJECT_ROOT / "data" / "raw" / "car_data.csv")
)

MODELS_DIR: Path = PROJECT_ROOT / "models"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
FIGURES_DIR: Path = ARTIFACTS_DIR / "figures"
MODEL_PATH: Path = MODELS_DIR / "car_price_model.joblib"
METRICS_PATH: Path = ARTIFACTS_DIR / "metrics.json"
COMPARISON_PATH: Path = ARTIFACTS_DIR / "model_comparison.csv"
DATA_QUALITY_PATH: Path = ARTIFACTS_DIR / "data_quality_report.json"

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5
CV_REPEATS: int = 3

# ---------------------------------------------------------------------------
# Dataset schema (CarDekho "car data.csv")
# ---------------------------------------------------------------------------
TARGET: str = "Selling_Price"  # used-car selling price, in lakh INR (1 lakh = 100,000)
NAME_COLUMN: str = "Car_Name"  # used only for data cleaning, never as a model feature

NUMERIC_INPUTS: list[str] = ["Year", "Present_Price", "Kms_Driven", "Owner"]
CATEGORICAL_INPUTS: list[str] = ["Fuel_Type", "Seller_Type", "Transmission"]
# Raw columns a caller must supply to get a prediction.
INPUT_FEATURES: list[str] = NUMERIC_INPUTS + CATEGORICAL_INPUTS

EXPECTED_COLUMNS: list[str] = [NAME_COLUMN, TARGET, *INPUT_FEATURES]

LAKH: int = 100_000
