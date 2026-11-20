import pandas as pd
import streamlit as st
from sws import ml, store
from . import components as X


def render(subject):
    b = X.get_bundle(subject)
    feats = b["meta"]["feats"]
    ref = store.load_data(subject)
    st.subheader(f"Predict a student - {subject}")
    st.caption("Change any score: the prediction updates instantly. No retraining, no code changes.")

    i1, i2 = st.columns(2)
    sid = i1.text_input("Student ID (optional)", key=f"p_id_{subject}")
    name = i2.text_input("Student name (optional)", key=f"p_nm_{subject}")
    vals = {}
    cols = st.columns(4)
    for i, f in enumerate(feats):
        s = pd.to_numeric(ref[f], errors="coerce")
        vals[f] = cols[i % 4].number_input(f, min_value=0.0, value=float(s.median()), step=1.0,
                                           key=f"p_{subject}_{f}", help=f"Training range {s.min():g} - {s.max():g}")
    row = ml.predict_one(b, vals)

    st.divider()
    c = st.columns(4)
    c[0].metric("Predicted final exam", f"{row['Predicted Final']:.1f}")
    if "Predicted Total" in row:
        c[1].metric("Predicted total", f"{row['Predicted Total']:.1f}")
    c[2].metric("Risk class", row["Risk Class"])
    c[3].metric("Confidence", f"{row['Confidence %']:.0f}%")
    X.proba_bar(row)

    one = row.to_frame().T
    one["Student ID"] = sid or "-"
    one["Name"] = name or "-"
    X.warning_panel(one.reset_index(drop=True), subject, key=f"warn_one_{subject}")
