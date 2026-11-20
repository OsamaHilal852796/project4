# Student Performance Warning System

Streamlit web app. ML = scikit-learn pipelines (Imputer -> Scaler -> PCA -> model), no rule-based logic.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py
First start trains the models from `data/seed_data.xlsx` (~20 s). Models are stored in `storage/`.

## Deploy (Streamlit Community Cloud)
1. Push this folder to a GitHub repo (keep `app.py` in the repo root).
2. share.streamlit.io -> New app -> pick repo, branch, main file `app.py`.
3. (Optional) App settings -> Secrets -> paste `.streamlit/secrets.toml.example` filled in, for email/SMS.

Other hosts (Render, Railway, Fly.io, any VPS): use the `Dockerfile`.
Free hosts have an ephemeral disk: uploaded subjects/models vanish on restart; the seed subjects retrain automatically.
Use a persistent disk/volume mounted at `storage/` if you need them to survive.

## Where to change things
- `sws/config.py`  risk thresholds, PCA variance, test size
- `sws/ml.py`      models and training
- `sws/notify.py`  email / SMS / WhatsApp
- `views/`         one file per page
