"""All machine-learning code: training, evaluation, prediction.

Pipeline for every target:  Imputer -> StandardScaler -> PCA -> model
Three targets per subject:
  1. final  (numeric)   final exam score
  2. total  (numeric)   total grade           (optional, needs a total column)
  3. risk   (4 classes) Low / Moderate / High / Critical
The final exam and the total/result columns are NEVER used as input features.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              RandomForestClassifier, GradientBoostingClassifier)
from sklearn.model_selection import (train_test_split, cross_val_score, cross_val_predict,
                                     KFold, StratifiedKFold)
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error, accuracy_score,
                             f1_score, confusion_matrix, classification_report)
from . import config as C
from .data import risk_from_pct, to_numeric_frame


def _pre(n_features):
    steps = [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    # PCA keeps the components that explain PCA_VARIANCE of the variance
    steps.append(("pca", PCA(n_components=C.PCA_VARIANCE, svd_solver="full", random_state=C.SEED)))
    return steps


def _reg_candidates():
    return {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=120, min_samples_leaf=3, random_state=C.SEED, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=120, max_depth=2, learning_rate=0.05, random_state=C.SEED),
    }


def _clf_candidates():
    return {
        "LogisticRegression": LogisticRegression(max_iter=3000),
        "RandomForest": RandomForestClassifier(n_estimators=120, min_samples_leaf=3, random_state=C.SEED, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.08, random_state=C.SEED),
    }


def _pipe(model, n_features):
    return Pipeline(_pre(n_features) + [("model", model)])


def _pick_best(cands, X, y, scoring, cv):
    """Cross-validate every candidate, return (best_name, table)."""
    rows = []
    for name, m in cands.items():
        s = cross_val_score(_pipe(m, X.shape[1]), X, y, cv=cv, scoring=scoring).mean()
        rows.append({"model": name, "cv_score": round(float(s), 4)})
    tab = pd.DataFrame(rows).sort_values("cv_score", ascending=False).reset_index(drop=True)
    return tab.loc[0, "model"], tab


def _reg_metrics(y, p):
    return {"R2": round(float(r2_score(y, p)), 4),
            "MAE": round(float(mean_absolute_error(y, p)), 3),
            "RMSE": round(float(np.sqrt(mean_squared_error(y, p))), 3)}


def _clf_metrics(y, p):
    return {"Accuracy": round(float(accuracy_score(y, p)), 4),
            "F1 (macro)": round(float(f1_score(y, p, average="macro")), 4)}


def _risk_order(classes):
    return [c for c in C.RISK_LABELS if c in set(classes)]


def train_subject(df: pd.DataFrame, meta: dict) -> dict:
    """df = full training data of ONE subject.  meta keys:
       id, name, final, total(optional), feats(list), total_max(float)."""
    feats = list(meta["feats"])
    d = df.copy()
    d[meta["final"]] = pd.to_numeric(d[meta["final"]], errors="coerce")
    has_total = bool(meta.get("total"))
    if has_total:
        d[meta["total"]] = pd.to_numeric(d[meta["total"]], errors="coerce")
    basis = meta["total"] if has_total else meta["final"]
    d = d.dropna(subset=[meta["final"], basis]).reset_index(drop=True)
    if len(d) < C.MIN_ROWS_TO_TRAIN:
        raise ValueError(f"Need at least {C.MIN_ROWS_TO_TRAIN} rows with a final score to train (got {len(d)}).")

    X = to_numeric_frame(d, feats)
    X = X.loc[:, X.notna().any()]                 # drop columns that are completely empty
    feats = list(X.columns)
    y_final = d[meta["final"]].astype(float).values
    y_total = d[meta["total"]].astype(float).values if has_total else None
    risk_max = float(meta["total_max"])
    y_risk = risk_from_pct(d[basis].astype(float).values / risk_max * 100)
    if len(set(y_risk)) < 2:
        raise ValueError("All students fall in one risk class - a classifier cannot be trained on this data.")

    counts = pd.Series(y_risk).value_counts()
    can_strat = counts.min() >= 2
    idx = np.arange(len(d))
    tr, te = train_test_split(idx, test_size=C.TEST_SIZE, random_state=C.SEED,
                              stratify=y_risk if can_strat else None)
    kf = KFold(C.CV_FOLDS, shuffle=True, random_state=C.SEED)
    k_clf = int(min(C.CV_FOLDS, counts.min())) if can_strat else 0
    skf = StratifiedKFold(k_clf, shuffle=True, random_state=C.SEED) if k_clf >= 2 else kf

    report, final_models, oof = {}, {}, {}
    comparison = {}
    targets = {"final": (y_final, "reg")}
    if has_total:
        targets["total"] = (y_total, "reg")
    targets["risk"] = (y_risk, "clf")

    for key, (y, kind) in targets.items():
        cands = _reg_candidates() if kind == "reg" else _clf_candidates()
        scoring = "r2" if kind == "reg" else "f1_macro"
        # choose the best algorithm using ONLY the training part
        best, tab = _pick_best(cands, X.iloc[tr], y[tr], scoring, kf if kind == "reg" else skf)
        comparison[key] = tab
        # hold-out evaluation
        p = _pipe(cands[best], X.shape[1]).fit(X.iloc[tr], y[tr])
        fn = _reg_metrics if kind == "reg" else _clf_metrics
        rep = {"algorithm": best,
               "train": fn(y[tr], p.predict(X.iloc[tr])),
               "test": fn(y[te], p.predict(X.iloc[te]))}
        if kind == "clf":
            labs = _risk_order(y_risk)
            rep["labels"] = labs
            rep["confusion_test"] = confusion_matrix(y[te], p.predict(X.iloc[te]), labels=labs).tolist()
            rep["class_report_test"] = classification_report(y[te], p.predict(X.iloc[te]), labels=labs,
                                                             output_dict=True, zero_division=0)
        report[key] = rep
        # out-of-fold predictions for every student (no student predicted by a model that saw him)
        cvobj = skf if kind == "clf" else kf
        if kind == "clf":
            oof[key] = cross_val_predict(_pipe(cands[best], X.shape[1]), X, y, cv=cvobj, method="predict_proba")
        else:
            oof[key] = cross_val_predict(_pipe(cands[best], X.shape[1]), X, y, cv=cvobj)
        # production model = refit on ALL data
        final_models[key] = _pipe(cands[best], X.shape[1]).fit(X, y)

    # PCA info from the production final-exam model
    pipe = final_models["final"]
    pca = pipe.named_steps["pca"]
    pca_info = {
        "n_components": int(pca.n_components_),
        "n_features": len(feats),
        "explained": pca.explained_variance_ratio_.tolist(),
        "loadings": pd.DataFrame(pca.components_.T, index=feats,
                                 columns=[f"PC{i+1}" for i in range(pca.n_components_)]),
    }

    bundle = {
        "meta": {**meta, "feats": feats, "has_total": has_total, "n_rows": len(d),
                 "final_max": float(np.nanmax(y_final)), "risk_basis": basis},
        "models": final_models,
        "classes": list(final_models["risk"].named_steps["model"].classes_),
        "report": report,
        "comparison": comparison,
        "pca": pca_info,
        "split": {"train": int(len(tr)), "test": int(len(te))},
    }
    bundle["results"] = _build_results(d, bundle, oof, test_idx=set(te.tolist()))
    return bundle


def _risk_table(proba, classes):
    P = pd.DataFrame(proba, columns=classes)
    for c in C.RISK_LABELS:
        if c not in P:
            P[c] = 0.0
    return P[C.RISK_LABELS]


def _assemble(d, bundle, pf, pt, proba):
    m = bundle["meta"]
    P = _risk_table(proba, bundle["classes"])
    out = pd.DataFrame(index=d.index)
    if m.get("id") and m["id"] in d:
        out["Student ID"] = d[m["id"]].values
    if m.get("name") and m["name"] in d:
        out["Name"] = d[m["name"]].values
    out["Predicted Final"] = np.clip(pf, 0, m["final_max"]).round(1)
    if pt is not None:
        out["Predicted Total"] = np.clip(pt, 0, float(m["total_max"])).round(1)
    out["Risk Class"] = P.idxmax(axis=1).values
    out["Confidence %"] = (P.max(axis=1).values * 100).round(1)
    for c in C.RISK_LABELS:
        out[f"P({c}) %"] = (P[c].values * 100).round(1)
    out["At-Risk %"] = ((P["Critical"] + P["High"]).values * 100).round(1)   # P(High)+P(Critical): below 60% of total
    return out


def _build_results(d, bundle, oof, test_idx):
    m = bundle["meta"]
    pf = oof["final"]
    pt = oof.get("total")
    res = _assemble(d, bundle, pf, pt, oof["risk"])
    res.insert(2 if "Name" in res else 1, "Subject", m.get("subject_name", ""))
    res["Actual Final"] = d[m["final"]].values
    if m.get("has_total"):
        res["Actual Total"] = d[m["total"]].values
    basis = d[m["risk_basis"]].astype(float).values / float(m["total_max"]) * 100
    res["Actual Risk"] = risk_from_pct(basis)
    res["Set"] = ["Test" if i in test_idx else "Train" for i in range(len(d))]
    for c in C.CONTACT_COLS:
        res[c] = d[c].values if c in d else ""
    return res


def predict_frame(bundle, df: pd.DataFrame):
    """Score ANY dataframe that has (some of) the feature columns. No retraining.
    Returns (results_df, missing_feature_names)."""
    m = bundle["meta"]
    feats = m["feats"]
    missing = [f for f in feats if f not in df.columns]
    X = to_numeric_frame(df, feats)
    pf = bundle["models"]["final"].predict(X)
    pt = bundle["models"]["total"].predict(X) if "total" in bundle["models"] else None
    proba = bundle["models"]["risk"].predict_proba(X)
    res = _assemble(df, bundle, pf, pt, proba)
    if m["final"] in df.columns and pd.to_numeric(df[m["final"]], errors="coerce").notna().any():
        res["Actual Final"] = pd.to_numeric(df[m["final"]], errors="coerce").values
    for c in C.CONTACT_COLS:
        if c in df:
            res[c] = df[c].values
    return res, missing


def predict_one(bundle, values: dict):
    """values: {feature: score}. Returns a one-row results dataframe."""
    row = pd.DataFrame([{f: values.get(f, np.nan) for f in bundle["meta"]["feats"]}])
    res, _ = predict_frame(bundle, row)
    return res.iloc[0]
