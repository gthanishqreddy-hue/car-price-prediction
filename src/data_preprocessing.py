"""Data loading, validation, cleaning, feature engineering and preprocessing.

Design rules followed in this module:

* Row-level cleaning (schema checks, duplicate removal, scoping to cars) uses
  no statistics computed from the data, so it is safe to run before the
  train/test split.
* Everything that *learns* from data (reference year, scaling, encoding) lives
  inside an sklearn ``Pipeline`` and is therefore fitted on the training fold
  only - this is what prevents data leakage during cross-validation and testing.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config

logger = logging.getLogger(__name__)

# First word of Car_Name for rows that are motorcycles/scooters, not cars.
# The raw CarDekho file mixes 101 two-wheeler listings (rows 100-200) into the
# car data; a car-price model should not learn from them.
TWO_WHEELER_BRANDS: frozenset[str] = frozenset(
    {"activa", "bajaj", "hero", "honda", "hyosung", "ktm", "mahindra",
     "royal", "suzuki", "tvs", "um", "yamaha"}
)

# Engineered columns produced by CarFeatureEngineer.
LOG_SCALED_FEATURES: list[str] = ["Present_Price", "Kms_Driven", "Kms_Per_Year"]
SCALED_FEATURES: list[str] = ["Car_Age", "Owner"]
MODEL_CATEGORICAL_FEATURES: list[str] = list(config.CATEGORICAL_INPUTS)


class DataValidationError(ValueError):
    """Raised when a dataset does not match the expected schema."""


@dataclass
class CleaningReport:
    """Counts of what the cleaning step changed, for logging and reporting."""

    rows_in: int = 0
    duplicates_removed: int = 0
    two_wheelers_removed: int = 0
    rows_out: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------
def load_raw_data(path: Path | str = config.DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and validate its schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Download it first with "
            "`python -m src.download_data` (see data/README.md)."
        )
    df = pd.read_csv(path)
    return validate_schema(df)


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Check required columns exist and coerce them to the expected dtypes.

    Raises:
        DataValidationError: if columns are missing or values cannot be parsed.
    """
    missing = [c for c in config.EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Dataset is missing required columns: {missing}")

    df = df[config.EXPECTED_COLUMNS].copy()
    numeric_cols = [config.TARGET, *config.NUMERIC_INPUTS]
    for col in numeric_cols:
        converted = pd.to_numeric(df[col], errors="coerce")
        bad = converted.isna() & df[col].notna()
        if bad.any():
            raise DataValidationError(
                f"Column '{col}' has {int(bad.sum())} non-numeric value(s), "
                f"e.g. {df.loc[bad, col].iloc[0]!r}"
            )
        df[col] = converted
    for col in [config.NAME_COLUMN, *config.CATEGORICAL_INPUTS]:
        df[col] = df[col].astype("string")
    return df


def is_two_wheeler(car_names: pd.Series) -> pd.Series:
    """Return a boolean mask that is True for motorcycle/scooter listings."""
    first_word = car_names.astype("string").str.strip().str.split().str[0].str.lower()
    return first_word.isin(TWO_WHEELER_BRANDS).fillna(False).astype(bool)


def clean_data(df: pd.DataFrame, cars_only: bool = True) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply deterministic row-level cleaning.

    Steps: normalise string whitespace, drop rows with missing values, drop exact
    duplicates (a duplicate split across train and test would leak the answer),
    and optionally drop two-wheeler listings.
    """
    report = CleaningReport(rows_in=len(df))
    df = df.copy()

    for col in [config.NAME_COLUMN, *config.CATEGORICAL_INPUTS]:
        df[col] = df[col].str.strip().str.replace(r"\s+", " ", regex=True)

    n_missing_rows = int(df.isna().any(axis=1).sum())
    if n_missing_rows:
        report.notes.append(f"Dropped {n_missing_rows} row(s) with missing values.")
        df = df.dropna()

    before = len(df)
    df = df.drop_duplicates()
    report.duplicates_removed = before - len(df)

    if cars_only:
        mask = is_two_wheeler(df[config.NAME_COLUMN])
        report.two_wheelers_removed = int(mask.sum())
        df = df.loc[~mask]

    invalid = (df[config.TARGET] <= 0) | (df["Present_Price"] <= 0) | (df["Kms_Driven"] < 0)
    if invalid.any():
        report.notes.append(f"Dropped {int(invalid.sum())} row(s) with impossible values.")
        df = df.loc[~invalid]

    df = df.reset_index(drop=True)
    report.rows_out = len(df)
    logger.info("Cleaning: %s", report.to_dict())
    return df, report


def load_clean_data(
    path: Path | str = config.DATA_PATH, cars_only: bool = True
) -> tuple[pd.DataFrame, CleaningReport]:
    """Convenience wrapper: load, validate and clean in one call."""
    return clean_data(load_raw_data(path), cars_only=cars_only)


def iqr_outlier_summary(df: pd.DataFrame, columns: list[str]) -> dict[str, dict]:
    """Count values outside the 1.5*IQR whiskers for each numeric column.

    Used for *investigation* only. Outliers are not removed automatically: in
    this dataset the extreme values are real vehicles (e.g. a Land Cruiser),
    and with ~200 rows every genuine example matters.
    """
    summary: dict[str, dict] = {}
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = df[(df[col] < low) | (df[col] > high)]
        summary[col] = {
            "lower_bound": float(low),
            "upper_bound": float(high),
            "n_outliers": len(outliers),
            "max_value": float(df[col].max()),
        }
    return summary


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
class CarFeatureEngineer(BaseEstimator, TransformerMixin):
    """Derive vehicle age and usage intensity from the raw inputs.

    * ``Car_Age`` = reference_year - Year. The reference year is *learned in
      fit()* (latest model year in the training data + 1) instead of using
      today's date, so predictions do not silently change as the calendar moves
      and no information from the test set is used.
    * ``Kms_Per_Year`` = Kms_Driven / max(Car_Age, 1): separates a 3-year-old
      car with 90k km (heavy use) from a 10-year-old car with 90k km.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> CarFeatureEngineer:
        self._check_columns(X)
        self.reference_year_ = int(X["Year"].max()) + 1
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "reference_year_"):
            raise RuntimeError("CarFeatureEngineer must be fitted before transform().")
        self._check_columns(X)
        out = X.copy()
        out["Car_Age"] = (self.reference_year_ - out["Year"]).clip(lower=0)
        out["Kms_Per_Year"] = out["Kms_Driven"] / out["Car_Age"].clip(lower=1)
        return out.drop(columns=["Year"])

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        names = [c for c in self.feature_names_in_ if c != "Year"]
        return np.asarray([*names, "Car_Age", "Kms_Per_Year"], dtype=object)

    @staticmethod
    def _check_columns(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CarFeatureEngineer expects a pandas DataFrame.")
        missing = [c for c in ("Year", "Kms_Driven") if c not in X.columns]
        if missing:
            raise DataValidationError(f"Missing columns for feature engineering: {missing}")


def build_preprocessor() -> ColumnTransformer:
    """Column-wise preprocessing applied after feature engineering.

    * Skewed positive quantities (prices, distances) -> log1p -> standardise.
    * Age and owner count -> standardise.
    * Categoricals -> one-hot; unknown categories at inference become all-zeros
      instead of crashing (the prediction layer also rejects them explicitly).

    Scaling/log transforms matter for Linear Regression; tree models are
    invariant to monotonic transforms, so sharing one preprocessor is safe.
    """
    log_then_scale = Pipeline(
        steps=[
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("log_num", log_then_scale, LOG_SCALED_FEATURES),
            ("num", StandardScaler(), SCALED_FEATURES),
            (
                "cat",
                OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
                MODEL_CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_feature_pipeline(regressor: BaseEstimator) -> Pipeline:
    """Feature engineering + preprocessing + regressor as one estimator, so the
    exact same transformations are fitted in training and replayed at inference."""
    return Pipeline(
        steps=[
            ("features", CarFeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("model", regressor),
        ]
    )


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model inputs from the target. Car_Name is intentionally excluded:
    98 unique names in ~200 rows would let the model memorise rather than learn."""
    X = df[config.INPUT_FEATURES].copy()
    for col in config.CATEGORICAL_INPUTS:
        X[col] = X[col].astype(object)
    y = df[config.TARGET].astype(float)
    return X, y


def train_test_split_stratified(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
    n_bins: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Hold out a test set whose price distribution mirrors the training set.

    With only ~200 rows a purely random split can put most expensive cars on
    one side; stratifying on price quantile bins keeps both sets representative.
    The bins are used only to choose rows, never as a feature.
    """
    X, y = split_features_target(df)
    price_bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=price_bins
    )
