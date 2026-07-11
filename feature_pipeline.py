import requests
import pandas as pd
from datetime import datetime
import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()

# Config from .env
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
FEATURE_GROUP_NAME = os.getenv("FEATURE_GROUP_NAME", "aqi_features")
FEATURE_GROUP_VERSION = int(os.getenv("FEATURE_GROUP_VERSION", "2"))
LATITUDE = float(os.getenv("LATITUDE", "31.5204"))
LONGITUDE = float(os.getenv("LONGITUDE", "74.3587"))
CITY_NAME = os.getenv("CITY_NAME", "Lahore")

def fetch_aqi_data():
    """Fetch current air quality data from Open-Meteo"""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "current": [
            "pm10", "pm2_5", "carbon_monoxide",
            "nitrogen_dioxide", "ozone", "sulphur_dioxide",
            "us_aqi", "european_aqi"
        ]
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def compute_features(data, prev_aqi=None):
    """Compute features from raw API data"""
    now = datetime.now()
    current = data["current"]

    aqi = int(current.get("us_aqi", 0) or 0)
    aqi_change_rate = int(aqi - int(prev_aqi)) if prev_aqi is not None else 0

    features = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "city": CITY_NAME,
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "day_of_week": now.weekday(),
        "pm2_5": current.get("pm2_5", 0) or 0,
        "pm10": current.get("pm10", 0) or 0,
        "carbon_monoxide": current.get("carbon_monoxide", 0) or 0,
        "nitrogen_dioxide": current.get("nitrogen_dioxide", 0) or 0,
        "ozone": current.get("ozone", 0) or 0,
        "sulphur_dioxide": current.get("sulphur_dioxide", 0) or 0,
        "us_aqi": aqi,
        "european_aqi": current.get("european_aqi", 0) or 0,
        "aqi_change_rate": aqi_change_rate,
    }

    return pd.DataFrame([features])

def connect_feature_group():
    """Return the AQI feature group (create on first run)."""
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME,
    )
    fs = project.get_feature_store()
    return fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["timestamp", "city"],
        description="AQI features for Lahore from Open-Meteo",
    )

if __name__ == "__main__":
    print("Fetching AQI data from Open-Meteo...")
    raw_data = fetch_aqi_data()

    fg = connect_feature_group()

    print("Computing features...")
    # prev_aqi is not fetched to avoid reading the entire feature group,
    # which caused the pipeline to exceed the 20-minute CI timeout.
    df = compute_features(raw_data, prev_aqi=None)
    df["aqi_change_rate"] = df["aqi_change_rate"].astype("int64")
    print(df)

    print("Storing in Hopsworks...")
    fg.insert(df, write_options={"kafka_timeout": 60})
    print("Features stored successfully!")
    print("Done!")