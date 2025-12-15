# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import requests

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="ANFDR –AI Based Nano Fertilizers Dosage Regulator",
    layout="wide"
)

# --------------------------------------------------
# AESTHETIC TITLE
# --------------------------------------------------
st.markdown("""
<h1 style='text-align:center'>
🌱🌿 NanoDose AI 🌿🌱<br>
<small>AI-Based Nano Fertilizer Dosage Regulator</small>
</h1>
""", unsafe_allow_html=True)

# --------------------------------------------------
# WEATHER DISPLAY (TOP PANEL)
# --------------------------------------------------
with st.container():
    st.markdown("""
    <div style='background-color:#d9f1ff;padding:15px;border-radius:12px'>
    <h3>🌦 Local Weather (Live)</h3>
    </div>
    """, unsafe_allow_html=True)

    city = st.text_input("City", "Delhi")

    try:
        geo = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1",
            timeout=5
        ).json()

        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]

        weather = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,wind_speed_10m",
            timeout=5
        ).json()["current"]

        st.success(
            f"🌡 {weather['temperature_2m']} °C | "
            f"💧 {weather['relative_humidity_2m']} % | "
            f"🌬 {weather['wind_speed_10m']} km/h"
        )
    except:
        st.warning("Weather unavailable")

st.divider()

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------
MODEL_PATH = "nano_model.joblib"

CROPS = ["fenugreek", "mint", "coriander", "spinach", "tomato"]

SOIL_TYPES = ["sandy", "loamy", "clayey"]

SAFE_MAX_BY_CROP = {
    "fenugreek": 2.0,
    "mint": 1.8,
    "coriander": 2.0,
    "spinach": 1.5,
    "tomato": 2.5
}

# --------------------------------------------------
# SYNTHETIC DATA GENERATION
# --------------------------------------------------
def generate_synthetic_data(n=5000, seed=42):
    rng = np.random.RandomState(seed)

    df = pd.DataFrame({
        "crop": rng.choice(CROPS, n),
        "soil_type": rng.choice(SOIL_TYPES, n),
        "sunlight_hrs": rng.uniform(3, 10, n),
        "temp_c": rng.uniform(15, 38, n),
        "soil_ph": rng.uniform(4.8, 7.8, n),
        "soil_moisture_pct": rng.uniform(15, 60, n),
        "plant_age_days": rng.randint(7, 70, n),
        "np_kg_ha": rng.uniform(20, 120, n),
        "deficiency_score": rng.randint(0, 4, n),
    })

    # ---- SINGLE CONSERVATIVE DOSE LOGIC ----
    dose = (
        0.25
        + 0.45 * (df["deficiency_score"] / 3)
        + 0.0012 * (80 - df["np_kg_ha"]).clip(0)
        + 0.06 * (6.5 - df["soil_ph"]).clip(0)
        + 0.04 * (30 - df["soil_moisture_pct"]).clip(0) / 30
        - 0.002 * df["plant_age_days"]
    )

    crop_factor = {
        "fenugreek": 0.95,
        "mint": 0.9,
        "coriander": 1.0,
        "spinach": 0.85,
        "tomato": 1.15,
    }

    soil_factor = {
        "sandy": 1.1,
        "loamy": 1.0,
        "clayey": 0.9,
    }

    dose *= df["crop"].map(crop_factor)
    dose *= df["soil_type"].map(soil_factor)

    df["nano_amount_ml"] = dose.clip(0.05, 3.0).round(3)
    return df

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------
def train_and_save_model(df, model_path=MODEL_PATH):
    NUM_COLS = [
        "sunlight_hrs",
        "temp_c",
        "soil_ph",
        "soil_moisture_pct",
        "plant_age_days",
        "np_kg_ha",
        "deficiency_score"
    ]

    X = df[NUM_COLS]
    y = df["nano_amount_ml"]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)

    joblib.dump({
        "pipeline": pipe,
        "features": NUM_COLS,
        "n_rows": len(df)
    }, model_path)

# --------------------------------------------------
# LOAD OR TRAIN INITIAL MODEL
# --------------------------------------------------
if not os.path.exists(MODEL_PATH):
    synth = generate_synthetic_data()
    train_and_save_model(synth)

bundle = joblib.load(MODEL_PATH)
pipe = bundle["pipeline"]
features = bundle["features"]

# --------------------------------------------------
# FARMER INPUT
# --------------------------------------------------
st.header("🌾 Nano Fertilizer Recommendation")

with st.form("predict_form"):
    crop = st.selectbox("Crop", CROPS)
    soil_type = st.selectbox("Soil Type", SOIL_TYPES)

    c1, c2, c3 = st.columns(3)
    sunlight = c1.number_input("Sunlight (hrs/day)", 0.0, 12.0, 6.0)
    temp = c2.number_input("Temperature (°C)", 0.0, 50.0, 28.0)
    ph = c3.number_input("Soil pH", 3.0, 9.0, 6.5)

    c4, c5, c6 = st.columns(3)
    moisture = c4.number_input("Soil Moisture (%)", 0.0, 100.0, 40.0)
    age = c5.number_input("Plant Age (days)", 1, 120, 30)
    np_kg_ha = c6.number_input("Available soil NPK (kg/ha equivalent)", 0.0, 200.0, 60.0)

    deficiency = st.selectbox("Deficiency Score (0 = none, 3 = severe)", [0, 1, 2, 3])
    plants = st.number_input("Number of plants", 1, 10000, 10)

    submit = st.form_submit_button("🌱 Predict Nano Dosage")

if submit:
    input_df = pd.DataFrame([{
        "sunlight_hrs": sunlight,
        "temp_c": temp,
        "soil_ph": ph,
        "soil_moisture_pct": moisture,
        "plant_age_days": age,
        "np_kg_ha": np_kg_ha,
        "deficiency_score": deficiency
    }])

    input_df = input_df[features]

    ml = float(pipe.predict(input_df)[0])
    ml = max(0.05, min(ml, 3.0))

    mg_per_l = ml * 1000
    ng_per_l = mg_per_l * 1_000_000
    total_ml = ml * plants

    st.subheader("🌱 Recommended Nano Fertilizer Dosage")
    st.metric("Per plant (ml)", f"{ml:.3f} ml")
    st.metric("Concentration", f"{mg_per_l:.0f} mg/L")
    st.metric("Concentration", f"{ng_per_l:.0f} ng/L")
    st.metric("Total for all plants", f"{total_ml:.2f} ml")

# --------------------------------------------------
# REAL DATA UPLOAD & RETRAIN (NO PASSWORD)
# --------------------------------------------------
st.divider()
st.header("📊 Upload Real Experiment Data (CSV)")

file = st.file_uploader("Upload CSV", type="csv")

if file:
    real_df = pd.read_csv(file)
    st.dataframe(real_df.head())

    if st.button("🔄 Retrain Model"):
        synth = generate_synthetic_data()
        combined = pd.concat([synth, real_df], ignore_index=True)
        train_and_save_model(combined)
        st.success("Model retrained successfully with synthetic + real data")

if st.button("⬇ Download Synthetic Training Data"):
    df_train = generate_synthetic_data(n=3000)
    st.download_button(
        "Download training_data.csv",
        df_train.to_csv(index=False),
        "training_data.csv",
        "text/csv"
    )

st.markdown("---")
st.warning("""
⚠ **Important Disclaimer**

- This recommendation is generated by an **AI model trained on synthetic data**.
- It is intended for **research and educational purposes only**.
- Always follow **manufacturer guidelines**, **government agricultural advisories**,
  and consult a **certified agronomist** before applying nano-fertilizers.
- Test on a small number of plants before full-scale use.
- The developers are **not responsible** for misuse or crop damage.
""")
