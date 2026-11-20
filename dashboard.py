import altair as alt
import streamlit as st
from . import components as X


def render(subject):
    b = X.get_bundle(subject)
    r = b["results"]
    st.subheader(f"Dashboard - {subject}")
    c = st.columns(5)
    c[0].metric("Students", len(r))
    c[1].metric("Avg predicted final", f"{r['Predicted Final'].mean():.1f}")
    if "Predicted Total" in r:
        c[2].metric("Avg predicted total", f"{r['Predicted Total'].mean():.1f}")
    c[3].metric("Critical", int((r["Risk Class"] == "Critical").sum()))
    c[4].metric("High risk", int((r["Risk Class"] == "High").sum()))

    l, rt = st.columns(2)
    with l:
        st.markdown("**Predicted risk distribution**")
        X.risk_bar(r)
    with rt:
        st.markdown("**Predicted vs actual final exam score** (out-of-fold)")
        ch = alt.Chart(r).mark_circle(size=40, opacity=.55).encode(
            x=alt.X("Actual Final:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("Predicted Final:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("Risk Class:N", scale=alt.Scale(domain=list(X.RISK_COLORS), range=list(X.RISK_COLORS.values()))),
            tooltip=[c for c in ["Name", "Actual Final", "Predicted Final", "Risk Class"] if c in r]).properties(height=240)
        st.altair_chart(ch)

    st.markdown("**Most at-risk students**")
    top = r.sort_values(["At-Risk %", "Predicted Total"] if "Predicted Total" in r else "At-Risk %",
                        ascending=[False, True] if "Predicted Total" in r else False).head(15)
    X.show_table(top[[c for c in top.columns if c not in ("Actual Final", "Actual Total", "Actual Risk", "Set", "Subject")]], height=380)
    rep = b["report"]
    st.caption(f"Models: final={rep['final']['algorithm']}, risk={rep['risk']['algorithm']} | "
               f"PCA {b['pca']['n_components']}/{b['pca']['n_features']} components | "
               f"hold-out test accuracy {rep['risk']['test']['Accuracy']:.0%}, final R2 {rep['final']['test']['R2']:.2f}")
