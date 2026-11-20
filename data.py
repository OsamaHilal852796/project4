"""Reading uploaded files, guessing columns, building the risk label."""
import io
import numpy as np
import pandas as pd
from . import config as C


def read_any(file) -> pd.DataFrame:
    """Read xlsx / xls / csv / tsv / json from a path or an uploaded file object."""
    name = getattr(file, "name", str(file)).lower()
    raw = file.read() if hasattr(file, "read") else None
    src = io.BytesIO(raw) if raw is not None else file
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        df = pd.read_excel(src)
    elif name.endswith(".json"):
        df = pd.read_json(src)
    elif name.endswith(".tsv"):
        df = pd.read_csv(src, sep="\t")
    else:
        df = pd.read_csv(src, sep=None, engine="python")  # sniff , ; tab
    df.columns = [str(c).strip() for c in df.columns]
    return df


def guess_mapping(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    pick = lambda *names: next((low[n.lower()] for n in names if n.lower() in low), None)
    return {
        "id": pick(C.DEFAULT_ID, "student_id", "id"),
        "name": pick(C.DEFAULT_NAME, "student_name", "full_name"),
        "subject": pick(C.DEFAULT_SUBJECT, "course"),
        "final": pick(C.DEFAULT_FINAL, "final_exam", "final_score"),
        "total": pick(C.DEFAULT_TOTAL, "total", "total_grade"),
        "ignore": [c for c in cols if any(h in c.lower() for h in C.IGNORE_HINTS)],
    }


def feature_candidates(df, mapping) -> list:
    """Numeric columns that are not id/name/targets/ignored/contact columns."""
    skip = {mapping.get(k) for k in ("id", "name", "subject", "final", "total")}
    skip |= set(mapping.get("ignore", [])) | set(C.CONTACT_COLS)
    out = []
    for c in df.columns:
        if c in skip:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= 0.5:
            out.append(c)
    return out


def risk_from_pct(pct) -> np.ndarray:
    """Percent of max grade -> risk label (used ONLY to build training labels)."""
    pct = np.asarray(pct, dtype=float)
    idx = np.digitize(pct, C.RISK_CUTS_PCT)          # 0=<50 ,1,2,3=>=75
    order = ["Critical", "High", "Moderate", "Low"]
    return np.array([order[i] for i in idx])


def to_numeric_frame(df, cols) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    return out
