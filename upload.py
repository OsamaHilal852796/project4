import math
import pandas as pd
import streamlit as st
from sws import config as C, data as D, ml, store
from . import components as X

NONE = "- none -"


def _show_batch(res, missing, subject, key):
    if missing:
        st.warning("Columns missing in the file (filled with the training median): " + ", ".join(missing))
    c = st.columns(4)
    c[0].metric("Students scored", len(res))
    c[1].metric("Critical", int((res["Risk Class"] == "Critical").sum()))
    c[2].metric("High", int((res["Risk Class"] == "High").sum()))
    c[3].metric("Avg predicted final", f"{res['Predicted Final'].mean():.1f}")
    X.show_table(res, detailed=st.toggle("Show class probabilities", key=f"{key}_t"), height=420)
    X.download_buttons(res.drop(columns=[c for c in res.columns if c.startswith(("student_", "guardian_"))]),
                       f"predictions_{subject}", key=f"{key}_dl")
    X.warning_panel(res.reset_index(drop=True), subject, key=f"{key}_warn")


def _existing(df, subject):
    b = X.get_bundle(subject)
    meta = b["meta"]
    mode = st.radio("What should happen with this file?",
                    ["Predict only (instant results, model unchanged)",
                     "Add to training data and retrain (file must contain the final exam column)"])
    sc = meta.get("subject")
    if sc and sc in df.columns and df[sc].nunique() > 1:
        opts = ["All rows"] + sorted(df[sc].astype(str).unique())
        default = opts.index(subject) if subject in opts else 0
        pick = st.selectbox(f"File contains several values in '{sc}' - use rows of:", opts, index=default)
        if pick != "All rows":
            df = df[df[sc].astype(str) == pick]
    feats = meta["feats"]
    if len(set(feats) - set(df.columns)) == len(feats):
        st.error("None of this subject's feature columns are in the file. Expected: " + ", ".join(feats))
        return
    if mode.startswith("Predict"):
        res, missing = ml.predict_frame(b, df.reset_index(drop=True))
        if "Actual Final" in res:
            st.info("The file has final-exam scores, so the actual value is shown next to the prediction.")
        _show_batch(res, missing, subject, f"up_{subject}")
    else:
        if meta["final"] not in df.columns:
            st.error(f"Column '{meta['final']}' (final exam) is required for retraining.")
            return
        if st.button("Retrain on old + new data", type="primary"):
            with st.spinner("Training..."):
                nb = store.append_and_retrain(subject, df.reset_index(drop=True))
            st.success(f"Retrained on {nb['meta']['n_rows']} students. Open 'Model report' / 'Student results'.")


def _new(df):
    mp = D.guess_mapping(df)
    cols = list(df.columns)
    st.markdown("**1. Map the columns**")
    a, b_, c_ = st.columns(3)
    idc = a.selectbox("Student ID", [NONE] + cols, index=([NONE] + cols).index(mp["id"]) if mp["id"] else 0)
    nmc = b_.selectbox("Student name", [NONE] + cols, index=([NONE] + cols).index(mp["name"]) if mp["name"] else 0)
    subc = c_.selectbox("Subject column (optional)", [NONE] + cols,
                        index=([NONE] + cols).index(mp["subject"]) if mp["subject"] else 0)
    d_, e_ = st.columns(2)
    fin = d_.selectbox("Final exam score (TARGET)", cols, index=cols.index(mp["final"]) if mp["final"] else 0)
    tot = e_.selectbox("Total grade (used for risk)", [NONE] + cols,
                       index=([NONE] + cols).index(mp["total"]) if mp["total"] else 0)
    ign = st.multiselect("Outcome columns to ignore (never used as inputs)", cols, default=mp["ignore"])
    mapping = {"id": None if idc == NONE else idc, "name": None if nmc == NONE else nmc,
               "subject": None if subc == NONE else subc, "final": fin,
               "total": None if tot == NONE else tot, "ignore": ign}
    feats = st.multiselect("Input features (PCA will compress these)", cols,
                           default=D.feature_candidates(df, mapping))
    basis = mapping["total"] or fin
    default_max = 100.0 if mapping["total"] else float(math.ceil(pd.to_numeric(df[fin], errors="coerce").max()))
    tmax = st.number_input(f"Maximum possible value of '{basis}' (risk = % of this)", value=default_max, min_value=1.0)

    st.markdown("**2. Subject name**")
    split = False
    if mapping["subject"]:
        split = st.checkbox(f"Create one subject for each value of '{mapping['subject']}' "
                            f"({df[mapping['subject']].nunique()} values)", value=True)
    name = st.text_input("Subject name", disabled=split)
    if st.button("Train", type="primary", disabled=(not split and not name.strip()) or not feats):
        meta = {"id": mapping["id"], "name": mapping["name"], "final": fin, "total": mapping["total"],
                "feats": feats, "total_max": float(tmax),
                "subject": mapping["subject"]}
        try:
            with st.spinner("Training models (PCA + model selection)..."):
                if split:
                    for sub, g in df.groupby(mapping["subject"]):
                        store.train_and_save(str(sub), g.reset_index(drop=True), meta)
                else:
                    store.train_and_save(name.strip(), df.reset_index(drop=True), meta)
            st.session_state["flash"] = "Training finished. Pick the subject in the sidebar."
            st.rerun()
        except Exception as e:
            st.error(str(e))


def render(subjects):
    st.subheader("Upload data")
    st.caption("Doctors / TAs: upload scores in Excel, CSV or JSON and get predictions immediately.")
    target = st.selectbox("Subject", subjects + ["+ New subject"],
                          index=subjects.index(st.session_state.get("subject")) if st.session_state.get("subject") in subjects else 0)
    f = st.file_uploader("Data file", type=["xlsx", "xls", "csv", "tsv", "json"])
    if not f:
        st.info("Choose a file. Existing subjects need the same score columns the model was trained on; "
                "optional contact columns: " + ", ".join(C.CONTACT_COLS))
        return
    try:
        df = D.read_any(f)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return
    st.caption(f"{len(df)} rows x {len(df.columns)} columns")
    with st.expander("File preview"):
        st.dataframe(df.head(20))
    if target == "+ New subject":
        _new(df)
    else:
        _existing(df, target)
