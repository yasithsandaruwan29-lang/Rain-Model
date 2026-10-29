import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(page_title="Rain Tomorrow Predictor", page_icon="🌧️", layout="centered")

# ---------------------------------------------------------------
# Load the trained model (saved from the notebook)
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("rain_model.json")
    return model

model = load_model()

# Must match the feature order used in training
FEATURES = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
    "shortwave_radiation_sum", "precipitation_sum", "precipitation_hours",
    "windspeed_10m_max", "windgusts_10m_max", "et0_fao_evapotranspiration",
    "day_length_hrs", "wind_sin", "wind_cos", "doy_sin", "doy_cos", "city_enc",
    "rain_lag1", "rain_lag2", "rain_roll3", "rain_roll7", "temp_roll7",
]
CITY_CODES = {"Moratuwa": 0, "Mount Lavinia": 1}

def build_features(v, prev_rain, avg_prev_temp, city, date):
    """Turn the form inputs into the 23 features the model was trained on."""
    doy = pd.Timestamp(date).dayofyear
    row = {k: v[k] for k in [
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
        "shortwave_radiation_sum", "precipitation_sum", "precipitation_hours",
        "windspeed_10m_max", "windgusts_10m_max", "et0_fao_evapotranspiration", "day_length_hrs"]}
    row["wind_sin"] = np.sin(np.radians(v["winddirection_10m_dominant"]))
    row["wind_cos"] = np.cos(np.radians(v["winddirection_10m_dominant"]))
    row["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    row["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    row["city_enc"] = CITY_CODES[city]
    row["rain_lag1"] = prev_rain[0]
    row["rain_lag2"] = prev_rain[1]
    row["rain_roll3"] = v["precipitation_sum"] + prev_rain[0] + prev_rain[1]
    row["rain_roll7"] = v["precipitation_sum"] + sum(prev_rain)
    row["temp_roll7"] = (v["temperature_2m_mean"] + 6 * avg_prev_temp) / 7
    return pd.DataFrame([row])[FEATURES]

# ---------------------------------------------------------------
# User interface
# ---------------------------------------------------------------
st.title("🌧️ Will it rain tomorrow?")
st.write("Enter today's weather and the rainfall of the previous 6 days. "
         "The XGBoost model predicts the chance of rain (> 1 mm) tomorrow "
         "for Moratuwa / Mount Lavinia, Sri Lanka.")

with st.form("weather_form"):
    c1, c2 = st.columns(2)
    city = c1.selectbox("City", list(CITY_CODES.keys()))
    date = c2.date_input("Today's date", value=pd.Timestamp.today())

    st.subheader("Today's weather")
    a, b, c = st.columns(3)
    v = {}
    v["temperature_2m_max"] = a.number_input("Max temp (°C)", value=29.3)
    v["temperature_2m_min"] = b.number_input("Min temp (°C)", value=24.5)
    v["temperature_2m_mean"] = c.number_input("Mean temp (°C)", value=26.6)
    v["apparent_temperature_max"] = a.number_input("Max feels-like (°C)", value=35.0)
    v["apparent_temperature_min"] = b.number_input("Min feels-like (°C)", value=29.1)
    v["apparent_temperature_mean"] = c.number_input("Mean feels-like (°C)", value=31.4)
    v["precipitation_sum"] = a.number_input("Rainfall today (mm)", min_value=0.0, value=0.0)
    v["precipitation_hours"] = b.number_input("Rain hours (0-24)", min_value=0.0, max_value=24.0, value=10.0)
    v["shortwave_radiation_sum"] = c.number_input("Solar radiation (MJ/m²)", min_value=0.0, value=18.4)
    v["windspeed_10m_max"] = a.number_input("Max wind speed (km/h)", min_value=0.0, value=14.2)
    v["windgusts_10m_max"] = b.number_input("Max wind gust (km/h)", min_value=0.0, value=31.7)
    v["winddirection_10m_dominant"] = c.number_input("Wind direction (0-360°)", min_value=0.0, max_value=360.0, value=227.0)
    v["et0_fao_evapotranspiration"] = a.number_input("Evapotranspiration (mm)", min_value=0.0, value=3.7)
    v["day_length_hrs"] = b.number_input("Day length (hours)", min_value=10.0, max_value=14.0, value=12.1)

    st.subheader("Previous days")
    st.caption("Rainfall in mm. Enter 0 for dry days.")
    r = st.columns(6)
    prev_rain = [r[i].number_input(f"{i+1}d ago", min_value=0.0, value=0.0, key=f"r{i}") for i in range(6)]
    avg_prev_temp = st.number_input("Average of the previous 6 days' mean temperature (°C)", value=26.6)

    submitted = st.form_submit_button("Predict tomorrow's rain")

if submitted:
    X_new = build_features(v, prev_rain, avg_prev_temp, city, date)
    prob = float(model.predict_proba(X_new)[0, 1])

    st.divider()
    st.subheader("Prediction")
    st.metric("Probability of rain tomorrow", f"{prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))
    if prob >= 0.7:
        st.error("🌧️ Rain is likely tomorrow.")
    elif prob >= 0.5:
        st.warning("🌦️ Rain is slightly more likely than not.")
    elif prob >= 0.3:
        st.info("⛅ No rain is slightly more likely, but rain is possible.")
    else:
        st.success("☀️ No rain is likely tomorrow.")

    with st.expander("See the features sent to the model"):
        st.dataframe(X_new.T.rename(columns={0: "value"}))

st.caption("Trained on 2010-2023 daily data for Moratuwa and Mount Lavinia. "
           "Test accuracy about 85%. Not a replacement for official forecasts.")