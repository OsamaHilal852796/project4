"""Disk storage: one folder entry per subject (data CSV + trained model).
NOTE: on free cloud hosts the disk is wiped on restart; the seed file is re-trained automatically."""
import re
import joblib
import pandas as pd
from . import config as C
from . import data as D
from . import ml


def slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower() or "subject"


def _p(name, ext):
    return C.STORAGE / f"{slug(name)}.{ext}"


def list_subjects() -> list:
    names = []
    for f in sorted(C.STORAGE.glob("*.joblib")):
        try:
            names.append(joblib.load(f)["meta"]["subject_name"])
        except Exception:
            pass
    return names


def load_bundle(name):
    return joblib.load(_p(name, "joblib"))


def load_data(name) -> pd.DataFrame:
    return pd.read_csv(_p(name, "csv"))


def train_and_save(name, df, meta):
    meta = {**meta, "subject_name": name}
    bundle = ml.train_subject(df, meta)
    df.to_csv(_p(name, "csv"), index=False, encoding="utf-8-sig")
    joblib.dump(bundle, _p(name, "joblib"))
    return bundle


def append_and_retrain(name, new_df):
    """Add rows to a subject (same columns as before) and retrain on everything."""
    bundle = load_bundle(name)
    meta = bundle["meta"]
    old = load_data(name)
    all_df = pd.concat([old, new_df], ignore_index=True)
    if meta.get("id") and meta["id"] in all_df:
        all_df = all_df.drop_duplicates(subset=[meta["id"]], keep="last")   # new rows win
    return train_and_save(name, all_df, meta)


def delete_subject(name):
    for ext in ("joblib", "csv"):
        _p(name, ext).unlink(missing_ok=True)


def bootstrap_seed():
    """First start: train every subject found in data/seed_data.xlsx (no manual step)."""
    if list_subjects() or not C.SEED_FILE.exists():
        return
    df = D.read_any(str(C.SEED_FILE))
    mp = D.guess_mapping(df)
    groups = df.groupby(mp["subject"]) if mp["subject"] else [("Course", df)]
    for sub, g in groups:
        meta = {k: mp[k] for k in ("id", "name", "final", "total")}
        meta["feats"] = D.feature_candidates(g, mp)
        meta["total_max"] = 100.0
        train_and_save(str(sub), g.reset_index(drop=True), meta)
