# AQI-Predictor
Live Link : https://maida-aqi-predictor.streamlit.app/
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

## Screenshots
<img width="1909" height="908" alt="image" src="https://github.com/user-attachments/assets/376c7b4d-7e57-4bba-8d12-a3fee3c179ae" />
<img width="1906" height="908" alt="image" src="https://github.com/user-attachments/assets/f760981d-a64c-4955-b3f0-f4dfed3a3535" />
<img width="1891" height="906" alt="image" src="https://github.com/user-attachments/assets/b3f70bbb-2ce7-4fd9-9599-3f43f9f01c58" />
<img width="1898" height="911" alt="image" src="https://github.com/user-attachments/assets/b251883f-77fe-4000-9c2e-93781aaac3f0" />
<img width="1901" height="906" alt="image" src="https://github.com/user-attachments/assets/aae85473-8c78-4dd0-8fa0-5b386870f5b0" />
<img width="1901" height="903" alt="image" src="https://github.com/user-attachments/assets/e75d76d4-45ed-4525-bbe0-658be8c2194f" />
<img width="1897" height="905" alt="image" src="https://github.com/user-attachments/assets/4bbd208d-362c-4215-8de4-000ba451fc65" />
<img width="1895" height="911" alt="image" src="https://github.com/user-attachments/assets/d947d3ca-e74b-42c9-9813-59eadec0f086" />










4. **Secrets:** `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT_NAME`; optional `LATITUDE`, `LONGITUDE`, `MODEL_VERSION`
5. Train pipeline must have run at least once so Hopsworks has `aqi_predictor`
