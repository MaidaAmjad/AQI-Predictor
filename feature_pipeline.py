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
FEATURE_GROUP_VERSION = int(os.getenv("FEATURE_GROUP_VERSION", "1"))
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

def compute_features(data):
    """Compute features from raw API data"""
    now = datetime.now()
    current = data["current"]

    # Calculate AQI change rate (placeholder for first run)
    aqi = current.get("us_aqi", 0) or 0

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
    }

    return pd.DataFrame([features])

def store_in_hopsworks(df):
    """Store features in Hopsworks Feature Store"""
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME
    )

    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["timestamp", "city"],
        description="AQI features for Lahore from Open-Meteo"
    )

    fg.insert(df)
    print("Features stored successfully!")

if __name__ == "__main__":
    print("Fetching AQI data from Open-Meteo...")
    raw_data = fetch_aqi_data()

    print("Computing features...")
    df = compute_features(raw_data)
    print(df)

    print("Storing in Hopsworks...")
    store_in_hopsworks(df)
    print("Done!")