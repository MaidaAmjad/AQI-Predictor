# AQI-Predictor

Air quality monitoring and 7-day AQI forecasting for Lahore, Pakistan.

**Documentation:** open [`docs/index.html`](docs/index.html) for project flow, 5-model training, Overview UI, GitHub Actions CI/CD, artifacts, libraries, setup, and troubleshooting.

## Quick start (local)

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-explain.txt
streamlit run app/app/streamlit_app.py
```

If `pip install -r requirements.txt` fails on Windows with a **twofish** / C++ build error, Hopsworks may already be installed — run `pip install -r requirements-explain.txt` only, then continue.

## GitHub Actions (CI/CD)

Workflows in [`.github/workflows/`](.github/workflows/):

| Workflow | Schedule (UTC) | Script |
|----------|------------------|--------|
| [`feature_pipeline.yml`](.github/workflows/feature_pipeline.yml) | Every hour (`0 * * * *`) | `feature_pipeline.py` |
| [`train_pipeline.yml`](.github/workflows/train_pipeline.yml) | Daily at 06:00 (`0 6 * * *`) | `training_pipeline.py` |

Both support manual **Run workflow** from the Actions tab.

### Required repository secrets

**Settings → Secrets and variables → Actions:**

| Secret | Required |
|--------|----------|
| `HOPSWORKS_API_KEY` | Yes |
| `HOPSWORKS_PROJECT_NAME` | Yes |

Optional: `FEATURE_GROUP_NAME`, `FEATURE_GROUP_VERSION`, `LATITUDE`, `LONGITUDE`, `CITY_NAME`

### First-time setup

1. Push this repo to GitHub and add the secrets above.
2. Run `python backfill.py` locally once (or manually) to seed historical data.
3. Let Actions run hourly (features) and daily (training), or trigger workflows manually.
4. Run the Streamlit app locally to view the dashboard.

See [docs/index.html — §5 GitHub Actions CI/CD](docs/index.html#cicd) for full details.

## Deploy dashboard (Streamlit Community Cloud)

**Full guide:** [DEPLOY_STREAMLIT.md](DEPLOY_STREAMLIT.md)

**Important:** Use **Python 3.12** in the deploy **Advanced settings** dropdown. Python **3.14** breaks this app (`protobuf` + Hopsworks). Rebooting is not enough — if logs show `Python 3.14.5`, delete the app and redeploy with **3.12** selected.

1. Connect the GitHub repo at [share.streamlit.io](https://share.streamlit.io).
2. **Main file:** `app/app/streamlit_app.py`
3. **Advanced settings → Python version:** **3.12** (required)
4. **Secrets:** `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT_NAME`; optional `LATITUDE`, `LONGITUDE`, `MODEL_VERSION`
5. Train pipeline must have run at least once so Hopsworks has `aqi_predictor`
