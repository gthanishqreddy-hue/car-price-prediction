"""Streamlit front-end for the car price model.

Run from the project root:
    streamlit run app/streamlit_app.py

The app only *loads* the model produced by `python -m src.train`; it never
retrains. The model is cached across reruns with st.cache_resource.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make `src` importable when Streamlit runs this file as a script. The pickled
# model references src.* classes, so this must happen before loading it.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.predict import (
    InvalidInputError,
    ModelBundle,
    ModelLoadError,
    format_inr,
    load_model,
    predict_price,
)

st.set_page_config(page_title="Used Car Price Estimator", page_icon="🚗", layout="wide")

st.markdown(
    """
    <style>
      .price-card {padding: 1.4rem 1.6rem; border-radius: 14px;
                   background: linear-gradient(135deg, #0f766e 0%, #155e75 100%);
                   color: #ffffff; margin-bottom: 0.8rem;}
      .price-card .label {font-size: 0.95rem; opacity: 0.85;}
      .price-card .value {font-size: 2.4rem; font-weight: 700; line-height: 1.2;}
      .price-card .sub {font-size: 0.95rem; opacity: 0.85;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model...")
def get_model() -> ModelBundle:
    return load_model(config.MODEL_PATH)


def render_sidebar(bundle: ModelBundle) -> None:
    meta = bundle.metadata
    test, cv = meta["metrics"]["test"], meta["metrics"]["cv"]
    with st.sidebar:
        st.header("Model")
        st.write(f"**{meta['model_name']}** (target: `{meta.get('target_mode', 'raw')}`)")
        c1, c2 = st.columns(2)
        c1.metric("Test R²", f"{test['R2']:.3f}")
        c2.metric("Test MAE", f"{test['MAE']:.2f} L")
        c1.metric("Test RMSE", f"{test['RMSE']:.2f} L")
        c2.metric("CV RMSE", f"{cv['RMSE']['mean']:.2f} L")
        st.caption("L = lakh INR (₹1,00,000). Metrics are computed on a held-out "
                   "test set the model never saw during training or selection.")
        st.caption(f"Trained {meta['trained_at_utc']} · scikit-learn {meta['sklearn_version']}")
        st.divider()
        st.caption("Data: CarDekho used-car listings (Kaggle). Prices reflect the "
                   f"Indian market up to ~{meta['feature_schema']['reference_year']}.")


def render_form(bundle: ModelBundle) -> dict | None:
    schema = bundle.schema
    num, cats = schema["numeric"], schema["categorical"]
    ref_year = int(schema["reference_year"])

    with st.form("vehicle"):
        st.subheader("Vehicle details")
        c1, c2, c3 = st.columns(3)
        year = c1.selectbox("Year of manufacture",
                            options=list(range(ref_year, int(num["Year"]["min"]) - 1, -1)),
                            index=ref_year - 2015 if ref_year >= 2015 else 0)
        present_price = c2.number_input(
            "Current ex-showroom price (lakh ₹)", min_value=0.0, value=8.0, step=0.25,
            help="Today's price of the same model new, e.g. 8.0 = ₹8,00,000. "
                 "This anchors the estimate to the car's segment.")
        kms = c3.number_input("Kilometres driven", min_value=0, value=40_000, step=1_000)

        c4, c5, c6, c7 = st.columns(4)
        fuel = c4.selectbox("Fuel type", cats["Fuel_Type"],
                            index=cats["Fuel_Type"].index("Petrol")
                            if "Petrol" in cats["Fuel_Type"] else 0)
        seller = c5.selectbox("Seller type", cats["Seller_Type"])
        transmission = c6.selectbox("Transmission", cats["Transmission"],
                                    index=cats["Transmission"].index("Manual")
                                    if "Manual" in cats["Transmission"] else 0)
        owner = c7.number_input("Previous owners", min_value=0, max_value=10, value=0, step=1)

        submitted = st.form_submit_button("Predict Selling Price", type="primary",
                                          width="stretch")
    if not submitted:
        return None
    return {"Year": year, "Present_Price": present_price, "Kms_Driven": kms,
            "Fuel_Type": fuel, "Seller_Type": seller, "Transmission": transmission,
            "Owner": owner}


def render_prediction(record: dict, bundle: ModelBundle) -> None:
    try:
        result = predict_price(record, bundle)
    except InvalidInputError as exc:
        st.error("Please fix the following:\n\n" + "\n".join(f"- {e}" for e in exc.errors))
        return

    retained = result.price_lakh / record["Present_Price"]
    mae = bundle.metadata["metrics"]["test"]["MAE"]
    low = max(result.price_lakh - mae, 0.0) * config.LAKH
    high = (result.price_lakh + mae) * config.LAKH
    st.markdown(
        f"""<div class="price-card">
              <div class="label">Estimated selling price</div>
              <div class="value">{format_inr(result.price_inr)}</div>
              <div class="sub">{result.price_lakh:.2f} lakh · retains {retained:.0%} of the
              new-car price</div>
            </div>""",
        unsafe_allow_html=True,
    )
    st.caption(f"Typical error on unseen cars is about ±{mae:.2f} lakh "
               f"(test MAE), i.e. roughly {format_inr(low)} – {format_inr(high)}.")
    for warning in result.warnings:
        st.warning(warning)


def render_insights() -> None:
    figures = [
        ("actual_vs_predicted.png", "Actual vs predicted (test set)"),
        ("feature_importance.png", "Feature importance"),
        ("model_comparison.png", "Model comparison"),
        ("residuals.png", "Residuals"),
    ]
    available = [(config.FIGURES_DIR / f, c) for f, c in figures if (config.FIGURES_DIR / f).exists()]
    if config.COMPARISON_PATH.exists():
        df = pd.read_csv(config.COMPARISON_PATH)
        cols = {"model": "Model", "target_mode": "Target", "cv_RMSE_mean": "CV RMSE",
                "cv_RMSE_std": "CV RMSE std", "test_MAE": "Test MAE",
                "test_RMSE": "Test RMSE", "test_R2": "Test R²"}
        st.dataframe(df[list(cols)].rename(columns=cols).round(3), hide_index=True,
                     width="stretch")
    if not available:
        st.info("Run `python -m src.train` to generate evaluation figures.")
    for i in range(0, len(available), 2):
        cols = st.columns(2)
        for col, (path, caption) in zip(cols, available[i:i + 2]):
            col.image(str(path), caption=caption, width="stretch")


def main() -> None:
    st.title("🚗 Used Car Price Estimator")
    st.write("Estimate the resale value of a used car in India from its age, mileage, "
             "fuel, transmission and current ex-showroom price.")
    try:
        bundle = get_model()
    except ModelLoadError as exc:
        st.error(str(exc))
        st.code("python -m src.download_data\npython -m src.train", language="bash")
        st.stop()

    render_sidebar(bundle)
    tab_predict, tab_model = st.tabs(["Predict", "Model performance"])
    with tab_predict:
        record = render_form(bundle)
        if record is not None:
            render_prediction(record, bundle)
    with tab_model:
        render_insights()


main()
