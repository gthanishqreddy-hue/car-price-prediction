"""Tests for loading, cleaning, feature engineering and leakage safety."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_preprocessing import (
    CarFeatureEngineer,
    DataValidationError,
    build_feature_pipeline,
    build_preprocessor,
    clean_data,
    is_two_wheeler,
    load_clean_data,
    load_raw_data,
    split_features_target,
    train_test_split_stratified,
    validate_schema,
)
from tests.conftest import requires_real_data


# ---------------------------------------------------------------- schema ----
def test_validate_schema_rejects_missing_columns(synthetic_df):
    with pytest.raises(DataValidationError, match="Kms_Driven"):
        validate_schema(synthetic_df.drop(columns=["Kms_Driven"]))


def test_validate_schema_rejects_non_numeric_values(synthetic_df):
    df = synthetic_df.astype({"Present_Price": object})
    df.loc[3, "Present_Price"] = "ten lakh"
    with pytest.raises(DataValidationError, match="Present_Price"):
        validate_schema(df)


def test_load_raw_data_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_raw_data(tmp_path / "nope.csv")


# -------------------------------------------------------------- cleaning ----
def test_clean_data_removes_duplicates_two_wheelers_and_whitespace(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "Fuel_Type"] = "  Petrol "
    bike = df.iloc[[1]].assign(Car_Name="Royal Enfield Classic 350")
    df = pd.concat([df, df.iloc[[5]], bike], ignore_index=True)  # 1 duplicate + 1 bike

    cleaned, report = clean_data(validate_schema(df))

    assert report.duplicates_removed == 1
    assert report.two_wheelers_removed == 1
    assert report.rows_out == len(synthetic_df)
    assert not cleaned.duplicated().any()
    assert set(cleaned["Fuel_Type"]) <= {"Petrol", "Diesel", "CNG"}


def test_clean_data_drops_rows_with_missing_values(synthetic_df):
    df = synthetic_df.copy()
    df.loc[2, "Kms_Driven"] = np.nan
    cleaned, report = clean_data(validate_schema(df))
    assert len(cleaned) == len(synthetic_df) - 1
    assert cleaned.isna().sum().sum() == 0
    assert any("missing" in note for note in report.notes)


def test_is_two_wheeler_flags_bikes_not_cars():
    names = pd.Series(["Bajaj Pulsar 150", "Activa 3g", "KTM RC390", "city", "fortuner",
                       "wagon r", " Hero  Splender iSmart"])
    assert is_two_wheeler(names).tolist() == [True, True, True, False, False, False, True]


# --------------------------------------------------- feature engineering ----
def test_feature_engineer_learns_reference_year_from_fit_data_only():
    train = pd.DataFrame({"Year": [2010, 2015], "Kms_Driven": [50_000, 10_000]})
    later = pd.DataFrame({"Year": [2018], "Kms_Driven": [5_000]})
    fe = CarFeatureEngineer().fit(train)
    assert fe.reference_year_ == 2016  # max(train Year) + 1, never from `later`
    out = fe.transform(later)
    assert out["Car_Age"].tolist() == [0]  # clipped: no negative ages
    assert out["Kms_Per_Year"].tolist() == [5_000]  # age clipped to 1 in denominator


def test_feature_engineer_computes_age_and_usage():
    X = pd.DataFrame({"Year": [2010, 2016], "Kms_Driven": [60_000, 30_000]})
    out = CarFeatureEngineer().fit(X).transform(X)
    assert "Year" not in out.columns
    assert out["Car_Age"].tolist() == [7, 1]
    assert out["Kms_Per_Year"].round(2).tolist() == [round(60_000 / 7, 2), 30_000]


def test_feature_engineer_requires_fit_and_columns():
    with pytest.raises(RuntimeError):
        CarFeatureEngineer().transform(pd.DataFrame({"Year": [2010], "Kms_Driven": [1]}))
    with pytest.raises(DataValidationError):
        CarFeatureEngineer().fit(pd.DataFrame({"Year": [2010]}))


def test_preprocessor_output_is_numeric_and_complete(synthetic_df):
    X, _ = split_features_target(synthetic_df)
    fe = CarFeatureEngineer().fit(X)
    pre = build_preprocessor().fit(fe.transform(X))
    out = pre.transform(fe.transform(X))
    names = list(pre.get_feature_names_out())
    assert out.shape == (len(X), len(names))
    assert np.isfinite(out).all()
    assert {"Present_Price", "Car_Age", "Kms_Per_Year", "Transmission_Manual"} <= set(names)


def test_preprocessor_ignores_unseen_category(synthetic_df):
    X, _ = split_features_target(synthetic_df)
    fe = CarFeatureEngineer().fit(X)
    pre = build_preprocessor().fit(fe.transform(X))
    unseen = X.iloc[[0]].assign(Fuel_Type="Electric")
    with pytest.warns(UserWarning, match="unknown categories"):
        out = pre.transform(fe.transform(unseen))
    assert np.isfinite(out).all()


# ------------------------------------------------------ split / leakage ----
def test_stratified_split_is_disjoint_and_reproducible(synthetic_df):
    X_tr, X_te, _, _ = train_test_split_stratified(synthetic_df)
    _, X_te2, _, _ = train_test_split_stratified(synthetic_df)
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert len(X_te) == round(len(synthetic_df) * config.TEST_SIZE)
    assert X_te.index.equals(X_te2.index)  # same seed -> same split
    assert "Car_Name" not in X_tr.columns and config.TARGET not in X_tr.columns


def test_pipeline_statistics_come_from_training_data_only(synthetic_df):
    """Scaler means must equal the *training* means: test rows never influence fit."""
    from sklearn.linear_model import LinearRegression

    X_tr, _, y_tr, _ = train_test_split_stratified(synthetic_df)
    pipe = build_feature_pipeline(LinearRegression()).fit(X_tr, y_tr)
    scaler = pipe.named_steps["preprocess"].named_transformers_["num"]
    engineered = pipe.named_steps["features"].transform(X_tr)
    np.testing.assert_allclose(scaler.mean_, engineered[["Car_Age", "Owner"]].mean().to_numpy())
    assert pipe.named_steps["features"].reference_year_ == X_tr["Year"].max() + 1


# ------------------------------------------------------- real dataset ----
@requires_real_data
def test_real_dataset_loads_and_cleans_as_documented():
    raw = load_raw_data()
    assert raw.shape == (301, 9)
    clean, report = load_clean_data()
    assert report.duplicates_removed == 2
    assert report.two_wheelers_removed == 101
    assert len(clean) == 198
    assert clean[config.INPUT_FEATURES].isna().sum().sum() == 0
