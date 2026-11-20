import altair as alt
import pandas as pd
import streamlit as st
from sws import config as C


def render(subject, bundle):
    b, rep = bundle, bundle["report"]
    st.subheader(f"Model report - {subject}")
    st.caption(f"{b['meta']['n_rows']} students | train {b['split']['train']} / test {b['split']['test']} (hold-out) | "
               "final model is refit on all students")
    names = {"final": "Final exam score (numeric)", "total": "Total grade (numeric)", "risk": "Risk class (4 classes)"}
    for k in [k for k in ("final", "total", "risk") if k in rep]:
        st.markdown(f"#### {names[k]} - algorithm chosen by cross-validation: `{rep[k]['algorithm']}`")
        m = pd.DataFrame({"Train": rep[k]["train"], "Test": rep[k]["test"]})
        l, r = st.columns(2)
        l.dataframe(m)
        r.dataframe(b["comparison"][k].rename(columns={"cv_score": "CV score (R2 / F1)"}), hide_index=True)
    labs = rep["risk"]["labels"]
    cm = pd.DataFrame(rep["risk"]["confusion_test"], index=labs, columns=labs)
    st.markdown("#### Risk confusion matrix (test set)")
    long = cm.reset_index().melt("index", var_name="Predicted", value_name="Students").rename(columns={"index": "Actual"})
    base = alt.Chart(long).encode(x=alt.X("Predicted:N", sort=labs), y=alt.Y("Actual:N", sort=labs))
    st.altair_chart(base.mark_rect().encode(color=alt.Color("Students:Q", scale=alt.Scale(scheme="blues"))) +
                    base.mark_text().encode(text="Students:Q"))

    p = b["pca"]
    st.markdown("#### PCA")
    st.write(f"{p['n_features']} input features were compressed to **{p['n_components']} principal components** "
             f"(keeping {C.PCA_VARIANCE:.0%} of the variance). PCA builds new combined features; the table below shows "
             "how much each original column contributes to each component.")
    ev = pd.DataFrame({"Component": [f"PC{i+1}" for i in range(len(p["explained"]))],
                       "Explained %": [v * 100 for v in p["explained"]]})
    ev["Cumulative %"] = ev["Explained %"].cumsum()
    st.altair_chart(alt.Chart(ev).mark_bar().encode(x="Component:N", y="Explained %:Q",
                    tooltip=["Component", "Explained %", "Cumulative %"]).properties(height=220))
    ld = p["loadings"].round(3).reset_index().melt("index", var_name="PC", value_name="Loading").rename(columns={"index": "Feature"})
    h = alt.Chart(ld).mark_rect().encode(x="PC:N", y=alt.Y("Feature:N", sort=None),
                                         color=alt.Color("Loading:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
                                         tooltip=["Feature", "PC", "Loading"]).properties(height=28 * p["n_features"] + 40)
    st.altair_chart(h)
