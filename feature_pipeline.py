"""
Feature pipeline: fetches current AQI from Open-Meteo and writes to Hopsworks.

Uses the Hopsworks REST API directly to resolve IDs, then inserts via the
SDK with engine="python" (stream / Kafka path) but with a strict timeout so
the job fails fast rather than hanging for 10+ minutes on Kafka SSL errors.
"""

import json
import requests
import pandas as pd
from datetime import datetime
import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
FEATURE_GROUP_NAME = os.getenv("FEATURE_GROUP_NAME", "aqi_features")
FEATURE_GROUP_VERSION = int(os.getenv("FEATURE_GROUP_VERSION", "2"))
LATITUDE = float(os.getenv("LATITUDE", "31.5204"))
LONGITUDE = float(os.getenv("LONGITUDE", "74.3587"))
CITY_NAME = os.getenv("CITY_NAME", "Lahore")

HOPSWORKS_BASE_URL = f"https://{HOPSWORKS_HOST}/hopsworks-api/api"
REST_HEADERS = {
    "Authorization": f"ApiKey {HOPSWORKS_API_KEY}",
    "Content-Type": "application/json",
}


# ---------------------------------------------------------------------------
# Step 1: fetch raw AQI data
# ---------------------------------------------------------------------------
def fetch_aqi_data() -> dict:
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


# ---------------------------------------------------------------------------
# Step 2: build feature row
# ---------------------------------------------------------------------------
def compute_features(data: dict) -> pd.DataFrame:
    now = datetime.utcnow()
    current = data["current"]
    aqi = int(current.get("us_aqi", 0) or 0)

    row = {
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
    df = pd.DataFrame([row])
    df["aqi_change_rate"] = df["aqi_change_rate"].astype("int64")
    return df


# ---------------------------------------------------------------------------
# Step 3: resolve IDs via REST (no SDK needed for this part)
# ---------------------------------------------------------------------------
def get_project_id() -> int:
    resp = requests.get(
        f"{HOPSWORKS_BASE_URL}/project/getProjectInfo/{HOPSWORKS_PROJECT_NAME}",
        headers=REST_HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    pid = resp.json()["projectId"]
    print(f"  project_id = {pid}")
    return pid


def get_feature_store_id(project_id: int) -> int:
    resp = requests.get(
        f"{HOPSWORKS_BASE_URL}/project/{project_id}/featurestores",
        headers=REST_HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    stores = resp.json()
    # The default feature store has the same name as the project (lower-cased)
    project_fs_name = f"{HOPSWORKS_PROJECT_NAME.lower()}_featurestore"
    for store in stores:
        if store.get("featurestoreName", "").lower() == project_fs_name:
            fs_id = store["featurestoreId"]
            print(f"  feature_store_id = {fs_id}")
            return fs_id
    # Fallback: return the first one
    fs_id = stores[0]["featurestoreId"]
    print(f"  feature_store_id (fallback) = {fs_id}")
    return fs_id


def get_feature_group_id(project_id: int, fs_id: int) -> int:
    resp = requests.get(
        f"{HOPSWORKS_BASE_URL}/project/{project_id}/featurestores/{fs_id}"
        f"/featuregroups/{FEATURE_GROUP_NAME}",
        headers=REST_HEADERS,
        params={"version": FEATURE_GROUP_VERSION},
        timeout=30,
    )
    resp.raise_for_status()
    items = resp.json()
    # Response may be a list or a single object
    if isinstance(items, list):
        fg = items[0]
    else:
        fg = items
    fg_id = fg["id"]
    print(f"  feature_group_id = {fg_id}")
    return fg_id


# ---------------------------------------------------------------------------
# Step 4: insert via SDK (stream path) with a short Kafka timeout
# ---------------------------------------------------------------------------
def insert_features(df: pd.DataFrame, project_id: int, fs_id: int, fg_id: int):
    print("Logging in to Hopsworks SDK...")
    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME,
        engine="python",
    )
    fs = project.get_feature_store()

    # Retrieve the feature group by its numeric ID to avoid another metadata call
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    print(f"Feature group type: {type(fg).__name__}")

    print("Inserting row...")
    fg.insert(
        df,
        write_options={
            "kafka_timeout": 30,      # fail fast instead of hanging for minutes
            "wait_for_job": False,    # don't poll the background Hudi job
        },
    )
    print("Insert complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Step 1: Fetch AQI data ---")
    raw_data = fetch_aqi_data()
    print("Fetched.")

    print("--- Step 2: Compute features ---")
    df = compute_features(raw_data)
    print(df.to_string(index=False))

    print("--- Step 3: Resolve Hopsworks IDs ---")
    project_id = get_project_id()
    fs_id = get_feature_store_id(project_id)
    fg_id = get_feature_group_id(project_id, fs_id)

    print("--- Step 4: Insert ---")
    insert_features(df, project_id, fs_id, fg_id)

    print("Done!")
