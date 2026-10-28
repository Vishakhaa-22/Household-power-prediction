"""
Streamlit app - Next-hour household electricity forecasting
Models: Linear Regression, Decision Tree, Random Forest
Run from the project root with:   python -m streamlit run app.py
"""
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Electricity Forecast", page_icon="⚡", layout="wide")

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "models.pkl"
COMPARISON_PATH = ROOT / "models" / "model_comparison.csv"
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"


# ---------- loading ----------
@st.cache_resource
def load_models():
    bundle = joblib.load(MODEL_PATH)
    return bundle["models"], bundle["features"], bundle["best"]


@st.cache_data
def load_features():
    return pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)


for p in (MODEL_PATH, FEATURES_PATH):
    if not p.exists():
        st.error(f"Missing file: {p}. Run Step 9 of the notebook first (see README).")
        st.stop()

MODELS, FEATURES, BEST = load_models()
data = load_features()
split_idx = int(len(data) * 0.8)                   # same 80/20 split as the notebook
split_time = data.index[split_idx]
AVG_USAGE = float(data["y"].mean())

# ---------- header ----------
st.title("⚡ Household Electricity Forecast")
st.caption("Enter recent usage and a date/hour - the three models predict the household's "
           "average power (kW) for that hour.")

tab_predict, tab_explore, tab_compare, tab_about = st.tabs(
    ["🔮 Predict", "📈 Test-period check", "🏆 Model comparison", "ℹ️ About"]
)

# ---------- tab 1: predict from user input ----------
DEFAULTS = {
    "in_date": data.index.max().date(),
    "in_hour": 18,
    "in_lag1": 1.50,
    "in_lag2": 1.40,
    "in_lag24": 1.60,
    "in_mean24": 1.10,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def load_example():
    """Fill the inputs with a real hour from the test period."""
    row = data.iloc[split_idx:].sample(1).iloc[0]
    ts = row.name
    st.session_state["in_date"] = ts.date()
    st.session_state["in_hour"] = int(ts.hour)
    st.session_state["in_lag1"] = float(round(row["lag_1"], 2))
    st.session_state["in_lag2"] = float(round(row["lag_2"], 2))
    st.session_state["in_lag24"] = float(round(row["lag_24"], 2))
    st.session_state["in_mean24"] = float(round(row["roll_mean_24"], 2))
    st.session_state["example"] = (ts.date(), int(ts.hour), float(row["y"]))


with tab_predict:
    st.subheader("Enter the details")
    st.write("Fill in the hour you want to predict and the recent electricity usage. "
             "All usage values are in **kW** (average power during that hour). "
             f"For this household the overall average is about **{AVG_USAGE:.2f} kW**.")

    st.button("🎲 Fill with a real example", on_click=load_example)

    left, right = st.columns(2)
    with left:
        st.markdown("**When do you want the prediction for?**")
        in_date = st.date_input("Date", min_value=date(2006, 1, 1), max_value=date(2035, 12, 31), key="in_date")
        in_hour = st.slider("Hour of day (0 = midnight, 18 = 6 pm)", 0, 23, key="in_hour")
    with right:
        st.markdown("**Recent usage (kW)**")
        in_lag1 = st.number_input("Usage 1 hour before", 0.0, 15.0, step=0.05, format="%.2f", key="in_lag1")
        in_lag2 = st.number_input("Usage 2 hours before", 0.0, 15.0, step=0.05, format="%.2f", key="in_lag2")
        in_lag24 = st.number_input("Usage at the same hour yesterday", 0.0, 15.0, step=0.05, format="%.2f", key="in_lag24")
        in_mean24 = st.number_input("Average usage over the last 24 hours", 0.0, 15.0, step=0.05, format="%.2f", key="in_mean24")

    # Build the feature row exactly like the notebook did
    d = pd.Timestamp(in_date)
    row = pd.DataFrame([{
        "lag_1": in_lag1,
        "lag_2": in_lag2,
        "lag_24": in_lag24,
        "roll_mean_24": in_mean24,
        "hour_sin": np.sin(2 * np.pi * in_hour / 24),
        "hour_cos": np.cos(2 * np.pi * in_hour / 24),
        "day_of_week": d.dayofweek,
        "is_weekend": int(d.dayofweek >= 5),
        "month": d.month,
    }])[FEATURES]

    st.divider()
    st.subheader(f"Predicted usage for {in_date:%d %b %Y}, {in_hour:02d}:00 - {in_hour:02d}:59")

    preds = {name: float(m.predict(row)[0]) for name, m in MODELS.items()}
    cols = st.columns(len(preds))
    for col, (name, value) in zip(cols, preds.items()):
        label = f"{name} ⭐" if name == BEST else name
        col.metric(label, f"{value:.3f} kW")
    st.caption(f"⭐ = best model on the test data. A value of 1 kW sustained for the hour means about 1 kWh of energy.")

    main = preds[BEST]
    level = "above" if main > AVG_USAGE * 1.15 else "below" if main < AVG_USAGE * 0.85 else "close to"
    st.info(f"**{BEST}** predicts **{main:.2f} kW** - {level} this household's average ({AVG_USAGE:.2f} kW).")

    st.bar_chart(pd.Series(preds, name="Predicted kW"))

    # If the user loaded a real example and did not change it, show the real value
    ex = st.session_state.get("example")
    if ex and ex[0] == in_date and ex[1] == in_hour:
        st.success(f"In the real data, the actual usage for this hour was **{ex[2]:.3f} kW**.")

    if max(in_lag1, in_lag2, in_lag24, in_mean24) > 8:
        st.warning("One of the values is much higher than what this household normally uses "
                   "(maximum in the data is about 6.5 kW), so the prediction may be unreliable.")

# ---------- tab 2: test-period check ----------
with tab_explore:
    st.subheader("Actual vs predicted on real data")
    model_name = st.selectbox("Model", list(MODELS.keys()),
                              index=list(MODELS.keys()).index(BEST))

    test_start, max_d = split_time.date(), data.index.max().date()
    default_start = max(test_start, (data.index.max() - pd.Timedelta(days=7)).date())
    c1, c2 = st.columns(2)
    start = c1.date_input("From", default_start, min_value=data.index.min().date(), max_value=max_d, key="ex_from")
    end = c2.date_input("To", max_d, min_value=data.index.min().date(), max_value=max_d, key="ex_to")

    if start > end:
        st.warning("'From' date must be before 'To' date.")
    else:
        part = data.loc[str(start): str(end) + " 23:59"]
        if part.empty:
            st.warning("No data in this range.")
        else:
            pred = MODELS[model_name].predict(part[FEATURES])
            actual = part["y"]
            if part.index.min() < split_time:
                st.info(f"Part of this range is before {split_time.date()} - that data was used for training, "
                        "so the scores look better than they really are. Pick later dates for an honest test.")
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE (kW)", f"{mean_absolute_error(actual, pred):.3f}")
            m2.metric("RMSE (kW)", f"{np.sqrt(mean_squared_error(actual, pred)):.3f}")
            m3.metric("R²", f"{r2_score(actual, pred):.3f}" if len(part) > 1 else "n/a")
            st.line_chart(pd.DataFrame({"Actual": actual, "Predicted": pred}, index=part.index))

# ---------- tab 3: model comparison ----------
with tab_compare:
    st.subheader("All models on the held-out test period")
    if COMPARISON_PATH.exists():
        comp = pd.read_csv(COMPARISON_PATH, index_col=0)
        st.dataframe(comp.round(4))
        st.bar_chart(comp["RMSE"])
        st.caption("RMSE: lower is better. The 'Naive baseline' simply repeats the last hour's usage; "
                   "every real model should beat it.")
    else:
        st.info("models/model_comparison.csv not found.")

# ---------- tab 4: about ----------
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
**Dataset:** UCI Individual Household Electric Power Consumption (minute-level readings, Dec 2006 - Nov 2010).

**Task:** forecast the next hour's `Global_active_power`.

**Models:** Linear Regression, Decision Tree, Random Forest.

**Method:**
1. Resample minute data to hourly averages; fill short gaps only.
2. Features from past data only: usage 1h, 2h and 24h earlier, the 24-hour average, and calendar features (hour, weekday, weekend, month).
3. Chronological 80/20 split, a naive baseline, and time-series cross-validation.

**Note:** an earlier version predicted power from same-minute current and scored R² = 0.998, but that was
data leakage (power ≈ voltage × current). This version is a genuine forecast.
        """
    )
