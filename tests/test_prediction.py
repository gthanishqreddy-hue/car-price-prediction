"""Tests for the model wrapper, persistence, input validation and inference."""

from __future__ import annotations

import math

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from src import config
from src.data_preprocessing import split_features_target
from src.evaluate import regression_metrics
from src.model import build_model
from src.predict import (
    InvalidInputError,
    ModelLoadError,
    format_inr,
    load_model,
    predict_batch,
    predict_price,
    validate_vehicle,
)
from src.predict import (
    main as predict_cli,
)
from tests.conftest import requires_trained_model


# ------------------------------------------------------- model wrapper ----
@pytest.mark.parametrize("mode", ["raw", "log", "ratio"])
def test_every_target_mode_predicts_in_lakh(synthetic_df, mode):
    X, y = split_features_target(synthetic_df)
    model = build_model(LinearRegression(), mode).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (len(X),)
    assert (pred >= 0).all()
    # Predictions must be on the original price scale, not log/ratio scale.
    assert regression_metrics(y, pred)["R2"] > 0.7


def test_ratio_mode_decodes_with_present_price(synthetic_df):
    """price = predicted retained-value ratio x the car's ex-showroom price."""
    X, y = split_features_target(synthetic_df)
    model = build_model(DecisionTreeRegressor(random_state=0), "ratio").fit(X, y)
    row = X.iloc[[0]]
    ratio = model.pipeline_.predict(row)[0]
    assert 0 < ratio < 1.5  # learned on the retained-value scale, not in lakh
    assert model.predict(row)[0] == pytest.approx(ratio * row["Present_Price"].iloc[0])


def test_invalid_target_mode_raises(synthetic_df):
    X, y = split_features_target(synthetic_df)
    with pytest.raises(ValueError, match="target_mode"):
        build_model(LinearRegression(), "sqrt").fit(X, y)


def test_wrapper_supports_clone_and_hyperparameter_search(synthetic_df):
    X, y = split_features_target(synthetic_df)
    model = build_model(DecisionTreeRegressor(random_state=0), "ratio")
    assert clone(model).get_params()["target_mode"] == "ratio"
    search = GridSearchCV(model, {"pipeline__model__max_depth": [2, 4]}, cv=3).fit(X, y)
    assert search.best_params_["pipeline__model__max_depth"] in (2, 4)


def test_regression_metrics_known_values():
    m = regression_metrics([1.0, 2.0, 3.0], np.array([1.0, 2.0, 5.0]))
    assert m["MAE"] == pytest.approx(2 / 3)
    assert m["MSE"] == pytest.approx(4 / 3)
    assert m["RMSE"] == pytest.approx(math.sqrt(4 / 3))
    assert m["R2"] == pytest.approx(1 - 4 / 2)


# --------------------------------------------------------- persistence ----
def test_load_model_missing_file(tmp_path):
    with pytest.raises(ModelLoadError, match="src.train"):
        load_model(tmp_path / "missing.joblib")


def test_load_model_rejects_corrupt_or_foreign_files(tmp_path):
    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"not a pickle")
    with pytest.raises(ModelLoadError):
        load_model(corrupt)
    foreign = tmp_path / "foreign.joblib"
    joblib.dump({"something": "else"}, foreign)
    with pytest.raises(ModelLoadError, match="not a model bundle"):
        load_model(foreign)


def test_saved_and_loaded_model_give_identical_predictions(trained_bundle, saved_model_path,
                                                           synthetic_df):
    X, _ = split_features_target(synthetic_df)
    loaded = load_model(saved_model_path)
    np.testing.assert_array_equal(trained_bundle.model.predict(X), loaded.model.predict(X))
    assert loaded.metadata["feature_schema"] == trained_bundle.metadata["feature_schema"]


# ---------------------------------------------------------- validation ----
def test_validate_vehicle_normalises_types_and_case(trained_bundle, valid_vehicle):
    record = {**valid_vehicle, "Year": "2015", "Kms_Driven": "40000", "Fuel_Type": " petrol ",
              "Transmission": "MANUAL"}
    clean, warnings = validate_vehicle(record, trained_bundle.schema)
    assert clean["Year"] == 2015 and isinstance(clean["Year"], int)
    assert clean["Kms_Driven"] == 40_000.0
    assert clean["Fuel_Type"] == "Petrol" and clean["Transmission"] == "Manual"
    assert list(clean) == config.INPUT_FEATURES
    assert warnings == []


@pytest.mark.parametrize("field", config.INPUT_FEATURES)
def test_each_missing_field_is_reported(trained_bundle, valid_vehicle, field):
    record = {k: v for k, v in valid_vehicle.items() if k != field}
    with pytest.raises(InvalidInputError) as exc:
        validate_vehicle(record, trained_bundle.schema)
    assert any(field in e for e in exc.value.errors)


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("Year", 2035, "between"),
        ("Year", 2015.5, "whole number"),
        ("Present_Price", 0, "greater than 0"),
        ("Present_Price", float("nan"), "finite"),
        ("Kms_Driven", -10, "between 0"),
        ("Kms_Driven", "lots", "must be a number"),
        ("Kms_Driven", True, "boolean"),
        ("Owner", -1, "between 0"),
        ("Fuel_Type", "Electric", "must be one of"),
        ("Seller_Type", "", "required"),
    ],
)
def test_invalid_values_are_rejected(trained_bundle, valid_vehicle, field, value, fragment):
    with pytest.raises(InvalidInputError) as exc:
        validate_vehicle({**valid_vehicle, field: value}, trained_bundle.schema)
    assert any(fragment in e for e in exc.value.errors)


def test_all_errors_are_reported_together(trained_bundle, valid_vehicle):
    bad = {**valid_vehicle, "Year": 2050, "Kms_Driven": -1, "Fuel_Type": "Hydrogen"}
    with pytest.raises(InvalidInputError) as exc:
        validate_vehicle(bad, trained_bundle.schema)
    assert len(exc.value.errors) == 3


def test_out_of_training_range_values_warn_but_predict(trained_bundle, valid_vehicle):
    result = predict_price({**valid_vehicle, "Present_Price": 500}, trained_bundle)
    assert result.price_lakh > 0
    assert any("outside the training range" in w for w in result.warnings)


# ----------------------------------------------------------- inference ----
def test_predict_price_returns_positive_float(trained_bundle, valid_vehicle):
    result = predict_price(valid_vehicle, trained_bundle)
    assert isinstance(result.price_lakh, float)
    assert 0 < result.price_lakh < valid_vehicle["Present_Price"] * 1.5
    assert result.price_inr == pytest.approx(result.price_lakh * config.LAKH)


def test_newer_car_is_not_cheaper_than_identical_older_car(trained_bundle, valid_vehicle):
    old = predict_price({**valid_vehicle, "Year": 2007}, trained_bundle).price_lakh
    new = predict_price({**valid_vehicle, "Year": 2017}, trained_bundle).price_lakh
    assert new > old


def test_batch_and_single_predictions_match(trained_bundle, valid_vehicle):
    rows = pd.DataFrame([valid_vehicle, {**valid_vehicle, "Year": 2010, "Fuel_Type": "Diesel"}])
    batch = predict_batch(rows, trained_bundle)
    singles = [predict_price(r, trained_bundle).price_lakh for r in rows.to_dict("records")]
    np.testing.assert_allclose(batch.to_numpy(), singles)


@pytest.mark.parametrize(
    ("amount", "expected"),
    [(0, "₹0"), (999, "₹999"), (1_000, "₹1,000"), (525_000, "₹5,25,000"),
     (12_345_678, "₹1,23,45,678"), (-150_000, "-₹1,50,000"), (99.6, "₹100")],
)
def test_format_inr_uses_indian_grouping(amount, expected):
    assert format_inr(amount) == expected


def test_cli_exit_codes(saved_model_path, capsys):
    base = ["--year", "2015", "--present-price", "9.5", "--kms", "40000", "--fuel", "Petrol",
            "--seller", "Dealer", "--transmission", "Manual", "--model-path",
            str(saved_model_path)]
    assert predict_cli(base) == 0
    assert "Estimated selling price" in capsys.readouterr().out
    bad = base.copy()
    bad[bad.index("Petrol")] = "Electric"
    assert predict_cli(bad) == 2
    assert "Fuel_Type" in capsys.readouterr().err


# ------------------------------------------------- real trained model ----
@requires_trained_model
def test_real_model_bundle_is_consistent(valid_vehicle):
    bundle = load_model()
    meta = bundle.metadata
    assert meta["input_features"] == config.INPUT_FEATURES
    assert set(meta["feature_schema"]["categorical"]["Fuel_Type"]) == {"CNG", "Diesel", "Petrol"}
    result = predict_price(valid_vehicle, bundle)
    # A 2015 car that costs 9.5 lakh new should resell for less than new.
    assert 0 < result.price_lakh < valid_vehicle["Present_Price"]
