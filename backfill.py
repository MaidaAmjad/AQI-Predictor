import requests
import pandas as pd
from datetime import datetime, timedelta
import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
FEATURE_GROUP_NAME = os.getenv("FEATURE_GROUP_NAME", "aqi_features")
FEATURE_GROUP_VERSION = 2
LATITUDE = float(os.getenv("LATITUDE", "31.5204"))
LONGITUDE = float(os.getenv("LONGITUDE", "74.3587"))
CITY_NAME = os.getenv("CITY_NAME", "Lahore")

def fetch_historical_aqi(start_date, end_date):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": [
            "pm10", "pm2_5", "carbon_monoxide",
            "nitrogen_dioxide", "ozone", "sulphur_dioxide",
            "us_aqi", "european_aqi"
        ],
        "start_date": start_date,
        "end_date": end_date,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def process_historical_data(data):
    hourly = data["hourly"]
    times = hourly["time"]
    records = []
    prev_aqi = None

    for i, time_str in enumerate(times):
        dt = datetime.fromisoformat(time_str)
        aqi = hourly["us_aqi"][i] or 0
        aqi_change_rate = aqi - prev_aqi if prev_aqi is not None else 0
        prev_aqi = aqi

        records.append({
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "city": CITY_NAME,
            "hour": dt.hour,
            "day": dt.day,
            "month": dt.month,
            "day_of_week": dt.weekday(),
            "pm2_5": hourly["pm2_5"][i] or 0,
            "pm10": hourly["pm10"][i] or 0,
            "carbon_monoxide": hourly["carbon_monoxide"][i] or 0,
            "nitrogen_dioxide": hourly["nitrogen_dioxide"][i] or 0,
            "ozone": hourly["ozone"][i] or 0,
            "sulphur_dioxide": hourly["sulphur_dioxide"][i] or 0,
            "us_aqi": aqi,
            "european_aqi": hourly["european_aqi"][i] or 0,
            "aqi_change_rate": aqi_change_rate,
        })

    return pd.DataFrame(records)

def store_in_hopsworks(df):
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
    fg.insert(df, write_options={"kafka_timeout": 60})
    print(f"Stored {len(df)} records successfully!")

if __name__ == "__main__":
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    print(f"Fetching data from {start_date} to {end_date}...")
    raw_data = fetch_historical_aqi(start_date, end_date)

    print("Processing data...")
    df = process_historical_data(raw_data)
    print(f"Total records: {len(df)}")
    print(df.head())

    print("Storing in Hopsworks...")
    store_in_hopsworks(df)
    print("Backfill complete!")