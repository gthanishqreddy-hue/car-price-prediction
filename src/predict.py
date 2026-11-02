"""Inference: load the saved model, validate inputs, predict selling price.

The saved artifact contains the *entire* fitted estimator (feature engineering,
preprocessing and regressor), so inference replays exactly the transformations
learned during training - nothing is re-fitted here.

Usage:
    python -m src.predict --year 2015 --present-price 9.5 --kms 40000 \
        --fuel Petrol --seller Dealer --transmission Manual --owner 0
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn

from src import config

logger = logging.getLogger(__name__)

# Hard limits for values that are physically impossible or clearly typos.
# Values that are possible but outside the training range only produce a warning.
MAX_KMS_DRIVEN = 1_000_000
MAX_OWNERS = 10
MIN_YEAR = 1980


class InvalidInputError(ValueError):
    """Raised when vehicle inputs fail validation. ``errors`` lists every problem."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


class ModelLoadError(RuntimeError):
    """Raised when the model file is missing or not a valid model bundle."""


@dataclass
class ModelBundle:
    model: Any
    metadata: dict[str, Any]

    @property
    def schema(self) -> dict[str, Any]:
        return self.metadata["feature_schema"]


@dataclass
class PredictionResult:
    price_lakh: float
    warnings: list[str] = field(default_factory=list)

    @property
    def price_inr(self) -> float:
        return self.price_lakh * config.LAKH


def load_model(path: Path | str = config.MODEL_PATH) -> ModelBundle:
    """Load the joblib bundle written by ``src.train``.

    Note: only load model files you created yourself - unpickling runs code.
    """
    path = Path(path)
    if not path.exists():
        raise ModelLoadError(
            f"No trained model at '{path}'. Train one with `python -m src.train`."
        )
    try:
        bundle = joblib.load(path)
    except Exception as exc:  # corrupted file, incompatible library versions, ...
        raise ModelLoadError(f"Could not load model from '{path}': {exc}") from exc
    if not isinstance(bundle, dict) or not {"model", "metadata"} <= bundle.keys():
        raise ModelLoadError(f"'{path}' is not a model bundle produced by src.train.")

    trained_with = bundle["metadata"].get("sklearn_version")
    if trained_with and trained_with != sklearn.__version__:
        logger.warning("Model trained with scikit-learn %s but running %s; "
                       "retrain if predictions look wrong.", trained_with, sklearn.__version__)
    return ModelBundle(model=bundle["model"], metadata=bundle["metadata"])


def _to_number(name: str, value: Any, errors: list[str], integer: bool = False) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        errors.append(f"'{name}' is required.")
        return None
    if isinstance(value, bool):
        errors.append(f"'{name}' must be a number, got a boolean.")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"'{name}' must be a number, got {value!r}.")
        return None
    if math.isnan(number) or math.isinf(number):
        errors.append(f"'{name}' must be a finite number.")
        return None
    if integer and not number.is_integer():
        errors.append(f"'{name}' must be a whole number, got {value!r}.")
        return None
    return number


def validate_vehicle(record: Mapping[str, Any], schema: Mapping[str, Any]) -> tuple[dict, list[str]]:
    """Validate and normalise one vehicle's raw features.

    Returns ``(clean_record, warnings)``. Raises ``InvalidInputError`` listing
    *all* problems at once, so a UI can show them together.
    """
    errors: list[str] = []
    warnings: list[str] = []
    unknown = sorted(set(record) - set(config.INPUT_FEATURES))
    if unknown:
        warnings.append(f"Ignored unexpected field(s): {unknown}")

    year = _to_number("Year", record.get("Year"), errors, integer=True)
    present = _to_number("Present_Price", record.get("Present_Price"), errors)
    kms = _to_number("Kms_Driven", record.get("Kms_Driven"), errors)
    owner = _to_number("Owner", record.get("Owner"), errors, integer=True)

    reference_year = int(schema["reference_year"])
    if year is not None and not MIN_YEAR <= year <= reference_year:
        errors.append(f"'Year' must be between {MIN_YEAR} and {reference_year} "
                      f"(the model's data snapshot year), got {int(year)}.")
    if present is not None and present <= 0:
        errors.append("'Present_Price' (ex-showroom price, lakh INR) must be greater than 0.")
    if kms is not None and not 0 <= kms <= MAX_KMS_DRIVEN:
        errors.append(f"'Kms_Driven' must be between 0 and {MAX_KMS_DRIVEN:,}.")
    if owner is not None and not 0 <= owner <= MAX_OWNERS:
        errors.append(f"'Owner' (number of previous owners) must be between 0 and {MAX_OWNERS}.")

    categoricals: dict[str, str] = {}
    for col in config.CATEGORICAL_INPUTS:
        allowed = schema["categorical"][col]
        value = record.get(col)
        if value is None or not str(value).strip():
            errors.append(f"'{col}' is required. Allowed values: {allowed}.")
            continue
        # Accept case/whitespace variations such as " petrol ".
        match = next((a for a in allowed if a.lower() == str(value).strip().lower()), None)
        if match is None:
            errors.append(f"'{col}' must be one of {allowed}, got {value!r}.")
        else:
            categoricals[col] = match

    if errors:
        raise InvalidInputError(errors)

    # Possible but outside what the model has seen -> predict, but warn.
    for col, value in (("Year", year), ("Present_Price", present), ("Kms_Driven", kms),
                       ("Owner", owner)):
        lo, hi = schema["numeric"][col]["min"], schema["numeric"][col]["max"]
        if not lo <= value <= hi:
            warnings.append(f"{col}={value:g} is outside the training range "
                            f"[{lo:g}, {hi:g}]; the estimate is less reliable.")

    clean = {"Year": int(year), "Present_Price": float(present), "Kms_Driven": float(kms),
             "Owner": int(owner), **categoricals}
    return {k: clean[k] for k in config.INPUT_FEATURES}, warnings


def predict_price(record: Mapping[str, Any], bundle: ModelBundle) -> PredictionResult:
    """Validate one vehicle and return its estimated selling price."""
    clean, warnings = validate_vehicle(record, bundle.schema)
    X = pd.DataFrame([clean], columns=config.INPUT_FEATURES)
    price = float(bundle.model.predict(X)[0])
    return PredictionResult(price_lakh=price, warnings=warnings)


def predict_batch(df: pd.DataFrame, bundle: ModelBundle) -> pd.Series:
    """Predict for many vehicles; every row is validated first."""
    rows = [validate_vehicle(row, bundle.schema)[0] for row in df.to_dict(orient="records")]
    X = pd.DataFrame(rows, columns=config.INPUT_FEATURES)
    return pd.Series(bundle.model.predict(X), index=df.index, name="predicted_price_lakh")


def format_inr(amount: float) -> str:
    """Format rupees with Indian digit grouping: 525000 -> '₹5,25,000'."""
    rupees = round(amount)
    sign = "-" if rupees < 0 else ""
    digits = str(abs(rupees))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join(groups) + "," + tail
    return f"{sign}₹{digits}"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a used car's selling price.")
    parser.add_argument("--year", required=True, help="Year of manufacture, e.g. 2015")
    parser.add_argument("--present-price", required=True,
                        help="Current ex-showroom price of the model, in lakh INR")
    parser.add_argument("--kms", required=True, help="Kilometres driven")
    parser.add_argument("--fuel", required=True, help="Petrol / Diesel / CNG")
    parser.add_argument("--seller", required=True, help="Dealer / Individual")
    parser.add_argument("--transmission", required=True, help="Manual / Automatic")
    parser.add_argument("--owner", default=0, help="Number of previous owners")
    parser.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = _parse_args(argv)
    record = {"Year": args.year, "Present_Price": args.present_price, "Kms_Driven": args.kms,
              "Fuel_Type": args.fuel, "Seller_Type": args.seller,
              "Transmission": args.transmission, "Owner": args.owner}
    try:
        bundle = load_model(args.model_path)
        result = predict_price(record, bundle)
    except InvalidInputError as exc:
        print("Invalid input:", *[f"  - {e}" for e in exc.errors], sep="\n", file=sys.stderr)
        return 2
    except ModelLoadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    for warning in result.warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    print(f"Estimated selling price: {format_inr(result.price_inr)} "
          f"({result.price_lakh:.2f} lakh) [{bundle.metadata['model_name']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
