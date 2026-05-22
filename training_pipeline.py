import json
import os

import hopsworks
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from explainability import FEATURES, TARGET, build_explainability_report

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
FEATURE_GROUP_NAME = os.getenv("FEATURE_GROUP_NAME", "aqi_features")
FEATURE_GROUP_VERSION = 2

MODELS = {
    "random_forest": ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42)),
    "ridge": ("Ridge Regression", Ridge()),
    "gradient_boosting": ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    "extra_trees": ("Extra Trees", ExtraTreesRegressor(n_estimators=100, random_state=42)),
    "knn": ("K-Nearest Neighbors", KNeighborsRegressor(n_neighbors=7)),
}


def get_features():
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME,
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    print("Reading data...")
    df = fg.read()
    return df, project


def evaluate_model(estimator, X_train, X_test, y_train, y_test):
    estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 4),
        "r2_pct": round(r2 * 100, 1),
        "estimator": estimator,
        "predictions": preds,
    }


def train_models(df):
    df = df.dropna()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples\n")

    results = {}
    for key, (display_name, estimator) in MODELS.items():
        print(f"Training {display_name}...")
        out = evaluate_model(estimator, X_train, X_test, y_train, y_test)
        results[key] = {
            "display_name": display_name,
            "rmse": out["rmse"],
            "mae": out["mae"],
            "r2": out["r2"],
            "r2_pct": out["r2_pct"],
            "estimator": out["estimator"],
            "predictions": out["predictions"],
        }
        print(f"  → RMSE: {out['rmse']:.2f}, MAE: {out['mae']:.2f}, R²: {out['r2']:.4f} ({out['r2_pct']:.1f}%)\n")

    best_key = max(results, key=lambda k: results[k]["r2"])
    best = results[best_key]
    print(f"Best model: {best['display_name']} (R² = {best['r2']:.4f})")

    metrics_report = {
        "best_model": best_key,
        "best_display_name": best["display_name"],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "models": {
            key: {
                "display_name": v["display_name"],
                "rmse": v["rmse"],
                "mae": v["mae"],
                "r2": v["r2"],
                "r2_pct": v["r2_pct"],
            }
            for key, v in results.items()
        },
        "test_actual": y_test.tolist(),
        "test_predictions": {key: v["predictions"].tolist() for key, v in results.items()},
    }

    best_estimator = results[best_key]["estimator"]
    return best_estimator, best_key, metrics_report, X_train, X_test


def save_model(model, model_name, metrics_report, project):
    os.makedirs("model", exist_ok=True)

    model_path = f"model/{model_name}.pkl"
    joblib.dump(model, model_path)
    joblib.dump(model, "model/best_model.pkl")
    print(f"Model saved locally at {model_path}")

    metrics_path = "model/metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)
    print(f"Metrics saved at {metrics_path}")

    best_metrics = metrics_report["models"][metrics_report["best_model"]]
    mr = project.get_model_registry()
    hw_model = mr.sklearn.create_model(
        name="aqi_predictor",
        metrics={
            "rmse": best_metrics["rmse"],
            "r2": best_metrics["r2"],
            "mae": best_metrics["mae"],
            "r2_pct": best_metrics["r2_pct"],
        },
        description=(
            f"AQI prediction for Lahore — best model: "
            f"{metrics_report['best_display_name']} ({metrics_report['best_model']})"
        ),
    )
    hw_model.save("model")
    print("Model saved to Hopsworks Model Registry!")


if __name__ == "__main__":
    df, project = get_features()
    print(f"Loaded {len(df)} records")
    print(df.head())

    print("\nTraining 5 models...")
    model, model_name, metrics_report, X_train, X_test = train_models(df)

    print("\nComputing SHAP and LIME explanations...")
    try:
        metrics_report["explainability"] = build_explainability_report(
            model, X_train, X_test, x_instance=df.dropna().iloc[-1]
        )
        print("  → SHAP and LIME summaries saved in metrics.json")
    except Exception as exc:
        print(f"  → Explainability skipped: {exc}")

    print("\nSaving best model and metrics...")
    save_model(model, model_name, metrics_report, project)
    print("Training pipeline complete!")
