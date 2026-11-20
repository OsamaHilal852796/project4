import streamlit as st
from . import components as X


def render(subject):
    b = X.get_bundle(subject)
    r = b["results"]
    st.subheader(f"Student results - {subject}")
    st.caption("Predictions are out-of-fold: each student is predicted by a model that never saw his own grades.")
    f1, f2, f3 = st.columns([2, 1, 2])
    risks = f1.multiselect("Risk class", ["Critical", "High", "Moderate", "Low"],
                           default=["Critical", "High", "Moderate", "Low"])
    sset = f2.radio("Set", ["All", "Train", "Test"], horizontal=True)
    q = f3.text_input("Search name / ID")
    v = r[r["Risk Class"].isin(risks)]
    if sset != "All":
        v = v[v["Set"] == sset]
    if q:
        mask = v.astype(str).apply(lambda col: col.str.contains(q, case=False, na=False)).any(axis=1)
        v = v[mask]
    st.write(f"{len(v)} students")
    X.show_table(v, detailed=st.toggle("Show class probabilities"), height=480)
    X.download_buttons(v.drop(columns=[c for c in v.columns if c.startswith("student_") or c.startswith("guardian_")]),
                       f"results_{subject}", key=f"dl_res_{subject}")
    X.warning_panel(v.reset_index(drop=True), subject, key=f"warn_res_{subject}")

    if len(v):
        st.divider()
        st.markdown("**Single student view**")
        label = v["Name"].astype(str) + "  |  " + v["Student ID"].astype(str) if "Name" in v else v["Student ID"].astype(str)
        pick = st.selectbox("Student", label.tolist())
        row = v.iloc[label.tolist().index(pick)]
        c = st.columns(4)
        c[0].metric("Predicted final", row["Predicted Final"], f"actual {row['Actual Final']}", delta_color="off")
        if "Predicted Total" in row:
            c[1].metric("Predicted total", row["Predicted Total"], f"actual {row['Actual Total']}", delta_color="off")
        c[2].metric("Predicted risk", row["Risk Class"], f"actual {row['Actual Risk']}", delta_color="off")
        c[3].metric("Confidence", f"{row['Confidence %']}%")
        X.proba_bar(row)
