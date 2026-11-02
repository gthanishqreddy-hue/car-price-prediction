"""Model wrapper that learns price on a transformed target scale.

Why a custom wrapper instead of sklearn's ``TransformedTargetRegressor``?
The most useful target here - the *share of the new-car price that is retained*
(Selling_Price / Present_Price) - depends on an input column, and
``TransformedTargetRegressor`` can only transform ``y`` on its own.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.data_preprocessing import build_feature_pipeline

TargetMode = Literal["raw", "log", "ratio"]
TARGET_MODES: tuple[str, ...] = ("raw", "log", "ratio")


class CarPriceRegressor(RegressorMixin, BaseEstimator):
    """Fit ``pipeline`` on an encoded target; always predict in lakh INR.

    target_mode:
        * ``"raw"``   - learn the selling price directly.
        * ``"log"``   - learn log1p(price). Depreciation is multiplicative, so
          log(price) is close to linear in log(new price) and age.
        * ``"ratio"`` - learn Selling_Price / Present_Price (value retained,
          roughly 0-1). This normalises away the car's segment, so tree models
          no longer need to extrapolate to prices they never saw in training.

    Present_Price is a model *input* known at prediction time, so using it to
    rescale the target is not leakage.
    """

    def __init__(self, pipeline: Pipeline, target_mode: TargetMode = "raw"):
        self.pipeline = pipeline
        self.target_mode = target_mode

    def _encode(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        if self.target_mode == "raw":
            return y
        if self.target_mode == "log":
            return np.log1p(y)
        return y / X["Present_Price"].to_numpy(dtype=float)

    def _decode(self, X: pd.DataFrame, z: np.ndarray) -> np.ndarray:
        if self.target_mode == "raw":
            price = z
        elif self.target_mode == "log":
            price = np.expm1(z)
        else:
            price = z * X["Present_Price"].to_numpy(dtype=float)
        # A selling price cannot be negative; raw-target linear models can
        # produce small negative values for very old, cheap cars.
        return np.clip(price, 0.0, None)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> CarPriceRegressor:
        if self.target_mode not in TARGET_MODES:
            raise ValueError(f"target_mode must be one of {TARGET_MODES}, got {self.target_mode!r}")
        y = np.asarray(y, dtype=float)
        self.pipeline_ = clone(self.pipeline).fit(X, self._encode(X, y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "pipeline_")
        return self._decode(X, self.pipeline_.predict(X))


def build_model(regressor: BaseEstimator, target_mode: TargetMode = "raw") -> CarPriceRegressor:
    """Full estimator: feature engineering -> preprocessing -> regressor,
    trained on the chosen target scale. Hyperparameters are addressed as
    ``pipeline__model__<param>``."""
    return CarPriceRegressor(build_feature_pipeline(regressor), target_mode=target_mode)
