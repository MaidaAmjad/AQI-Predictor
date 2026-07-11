import requests
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Config from .env
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
FEATURE_GROUP_NAME = os.getenv("FEATURE_GROUP_NAME", "aqi_features")
FEATURE_GROUP_VERSION = int(os.getenv("FEATURE_GROUP_VERSION", "2"))
LATITUDE = float(os.getenv("LATITUDE", "31.5204"))
LONGITUDE = float(os.getenv("LONGITUDE", "74.3587"))
CITY_NAME = os.getenv("CITY_NAME", "Lahore")

# Hopsworks REST API base
HOPSWORKS_BASE_URL = f"https://{HOPSWORKS_HOST}/hopsworks-api/api"
HEADERS = {
    "Authorization": f"ApiKey {HOPSWORKS_API_KEY}",
    "Content-Type": "application/json",
}


def fetch_aqi_data():
    """Fetch current air quality data from Open-Meteo."""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "current": [
            "pm10", "pm2_5", "carbon_monoxide",
            "nitrogen_dioxide", "ozone", "sulphur_dioxide",
            "us_aqi", "european_aqi",
        ],
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def compute_features(data):
    """Compute features from raw API data."""
    now = datetime.utcnow()
    current = data["current"]

    aqi = int(current.get("us_aqi", 0) or 0)

    return {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "city": CITY_NAME,
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "day_of_week": now.weekday(),
        "pm2_5": float(current.get("pm2_5", 0) or 0),
        "pm10": float(current.get("pm10", 0) or 0),
        "carbon_monoxide": float(current.get("carbon_monoxide", 0) or 0),
        "nitrogen_dioxide": float(current.get("nitrogen_dioxide", 0) or 0),
        "ozone": float(current.get("ozone", 0) or 0),
        "sulphur_dioxide": float(current.get("sulphur_dioxide", 0) or 0),
        "us_aqi": aqi,
        "european_aqi": int(current.get("european_aqi", 0) or 0),
        "aqi_change_rate": 0,
    }


def get_project_id():
    """Resolve the numeric project ID from the project name."""
    print(f"Resolving project ID for '{HOPSWORKS_PROJECT_NAME}'...")
    resp = requests.get(
        f"{HOPSWORKS_BASE_URL}/project/getProjectInfo/{HOPSWORKS_PROJECT_NAME}",
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    project_id = resp.json()["projectId"]
    print(f"Project ID: {project_id}")
    return project_id


def insert_via_rest(project_id, row: dict):
    """
    Insert a single row into the feature group using the Hopsworks REST API.
    This bypasses the SDK's Kafka path entirely.
    """
    print("Inserting row via Hopsworks REST API...")
    url = (
        f"{HOPSWORKS_BASE_URL}/project/{project_id}/featurestores/"
        f"featuregroups/{FEATURE_GROUP_NAME}/{FEATURE_GROUP_VERSION}/ingest"
    )
    payload = {"entries": [row]}
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=60)
    if resp.status_code not in (200, 201, 204):
        print(f"REST insert failed ({resp.status_code}): {resp.text}")
        resp.raise_for_status()
    print("Row inserted successfully via REST.")


if __name__ == "__main__":
    print("Fetching AQI data from Open-Meteo...")
    raw_data = fetch_aqi_data()
    print("AQI data fetched.")

    row = compute_features(raw_data)
    print(f"Features computed: {row}")

    project_id = get_project_id()
    insert_via_rest(project_id, row)

    print("Done!")
