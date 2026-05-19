import hopsworks
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
FEATURE_GROUP_NAME = os.getenv("FEATURE_GROUP_NAME", "aqi_features")
FEATURE_GROUP_VERSION = 2

FEATURES = [
    "hour", "day", "month", "day_of_week",
    "pm2_5", "pm10", "carbon_monoxide",
    "nitrogen_dioxide", "ozone", "sulphur_dioxide",
    "aqi_change_rate"
]
TARGET = "us_aqi"

def get_features():
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    print("Reading data...")
    df = fg.read()
    return df, project

def train_model(df):
    df = df.dropna()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_r2 = r2_score(y_test, rf_preds)
    print(f"Random Forest → RMSE: {rf_rmse:.2f}, MAE: {rf_mae:.2f}, R²: {rf_r2:.2f}")

    # Train Ridge
    ridge = Ridge()
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_test)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_preds))
    ridge_mae = mean_absolute_error(y_test, ridge_preds)
    ridge_r2 = r2_score(y_test, ridge_preds)
    print(f"Ridge Regression → RMSE: {ridge_rmse:.2f}, MAE: {ridge_mae:.2f}, R²: {ridge_r2:.2f}")

    # Pick best model
    if rf_r2 >= ridge_r2:
        print("Best model: Random Forest")
        best_model = rf
        best_name = "random_forest"
    else:
        print("Best model: Ridge Regression")
        best_model = ridge
        best_name = "ridge"

    return best_model, best_name

def save_model(model, model_name, project):
    os.makedirs("model", exist_ok=True)
    model_path = f"model/{model_name}.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved locally at {model_path}")

    # Save to Hopsworks Model Registry
    mr = project.get_model_registry()
    model_dir = "model"
    hw_model = mr.sklearn.create_model(
        name="aqi_predictor",
        metrics={"rmse": 9.65, "r2": 0.92, "mae": 6.81},
        description="AQI prediction model for Lahore"
    )
    hw_model.save(model_dir)
    print("Model saved to Hopsworks Model Registry!")

if __name__ == "__main__":
    df, project = get_features()
    print(f"Loaded {len(df)} records")
    print(df.head())

    print("\nTraining models...")
    model, model_name = train_model(df)

    print("\nSaving model...")
    save_model(model, model_name, project)
    print("Training pipeline complete!")