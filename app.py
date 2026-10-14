"""
Streamlit demo — Titanic Survival Predictor.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from titanic_predictor.pipeline import train_final_model
from titanic_predictor.preprocessing import prepare_raw_xy

MODEL_CANDIDATES = [
    ROOT / "artifacts" / "calibrated_model.joblib",
    ROOT / "artifacts" / "tuned_model.joblib",
    ROOT / "artifacts" / "best_model.joblib",
]


@st.cache_resource(show_spinner="Loading survival model…")
def load_or_train_model():
    for path in MODEL_CANDIDATES:
        if path.exists():
            return joblib.load(path), f"loaded:{path.name}"

    model, _, _ = train_final_model(
        ROOT / "train.csv",
        model_name="gradient_boosting",
        age_strategy="knn",
    )
    return model, "trained:gradient_boosting"


def passenger_frame(
    *,
    pclass: int,
    name: str,
    sex: str,
    age: float,
    sibsp: int,
    parch: int,
    fare: float,
    cabin: str,
    embarked: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Pclass": pclass,
                "Name": name,
                "Sex": sex,
                "Age": age if age > 0 else None,
                "SibSp": sibsp,
                "Parch": parch,
                "Fare": fare,
                "Cabin": cabin if cabin.strip() else None,
                "Embarked": embarked,
            }
        ]
    )


def main() -> None:
    st.set_page_config(
        page_title="Titanic Survival Predictor",
        page_icon="🚢",
        layout="centered",
    )
    st.title("Titanic Survival Predictor")
    st.caption(
        "Enter passenger details to estimate survival probability. "
        "Model uses the leak-safe feature pipeline from this repository."
    )

    model, source = load_or_train_model()
    st.sidebar.markdown(f"**Model source:** `{source}`")
    threshold = st.sidebar.slider(
        "Decision threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.01,
        help="Predict Survived=1 when probability ≥ threshold. "
        "Tune this with scripts/calibrate_and_threshold.py.",
    )

    with st.form("passenger_form"):
        col1, col2 = st.columns(2)
        with col1:
            pclass = st.selectbox("Passenger class", [1, 2, 3], index=2)
            sex = st.selectbox("Sex", ["male", "female"])
            age = st.number_input("Age", min_value=0.0, max_value=80.0, value=29.0)
            fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)
        with col2:
            sibsp = st.number_input("Siblings/spouses aboard", min_value=0, max_value=10, value=0)
            parch = st.number_input("Parents/children aboard", min_value=0, max_value=10, value=0)
            embarked = st.selectbox("Embarked", ["S", "C", "Q"], index=0)
            cabin = st.text_input("Cabin (optional)", value="")

        title = "Mr" if sex == "male" else "Miss"
        name = st.text_input(
            "Name (used for title extraction)",
            value=f"Demo, {title}. Passenger",
        )
        submitted = st.form_submit_button("Predict survival", type="primary")

    if not submitted:
        st.info("Fill in the form and click **Predict survival**.")
        return

    frame = passenger_frame(
        pclass=int(pclass),
        name=name,
        sex=sex,
        age=float(age),
        sibsp=int(sibsp),
        parch=int(parch),
        fare=float(fare),
        cabin=cabin,
        embarked=embarked,
    )
    X, _ = prepare_raw_xy(frame)
    proba = float(model.predict_proba(X)[0, 1])
    survived = int(proba >= threshold)

    st.subheader("Prediction")
    metric_col, gauge_col = st.columns([1, 1])
    with metric_col:
        st.metric(
            "Survived?",
            "Yes" if survived else "No",
            help=f"Threshold = {threshold:.2f}",
        )
        st.metric("Survival probability", f"{proba:.1%}")
    with gauge_col:
        st.progress(min(max(proba, 0.0), 1.0))
        st.caption(f"P(Survived=1) = {proba:.3f}")

    with st.expander("Input row sent to the model"):
        st.dataframe(frame, use_container_width=True)

    st.markdown(
        "Docs: [Methodology](docs/METHODOLOGY.md) · "
        "[Interpretation](docs/INTERPRETATION.md)"
    )


if __name__ == "__main__":
    main()
