"""Small UI pieces shared by all pages."""
import io
import altair as alt
import pandas as pd
import streamlit as st
from sws import config as C, notify, store

RISK_COLORS = {"Low": "#2e9e5b", "Moderate": "#e6b800", "High": "#f28c28", "Critical": "#d93636"}
_RISK_SCALE = alt.Scale(domain=list(RISK_COLORS), range=list(RISK_COLORS.values()))


@st.cache_resource(show_spinner=False)
def _load(name, stamp):
    return store.load_bundle(name)


def get_bundle(name):
    """Load a trained subject model (cached; reloaded automatically after retraining)."""
    return _load(name, store._p(name, "joblib").stat().st_mtime_ns)


def style_risk(df):
    def color(v):
        if v in RISK_COLORS:
            fg = "black" if v == "Moderate" else "white"
            return f"background-color:{RISK_COLORS[v]};color:{fg};font-weight:600"
        return ""
    cols = [c for c in ("Risk Class", "Actual Risk") if c in df.columns]
    return df.style.map(color, subset=cols) if cols else df


def show_table(df, detailed=False, height=420):
    d = df.copy()
    hide = list(C.CONTACT_COLS)
    if not detailed:
        hide += [c for c in d.columns if c.startswith("P(")]
    d = d.drop(columns=[c for c in hide if c in d.columns])
    st.dataframe(style_risk(d), height=height, hide_index=True)


def download_buttons(df, base_name, key):
    c1, c2, _ = st.columns([1, 1, 4])
    c1.download_button("Download CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       f"{base_name}.csv", "text/csv", key=f"{key}_csv")
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    c2.download_button("Download Excel", buf.getvalue(), f"{base_name}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"{key}_xlsx")


def risk_bar(df, col="Risk Class", height=240):
    counts = df[col].value_counts().reindex(C.RISK_LABELS, fill_value=0).reset_index()
    counts.columns = ["Risk", "Students"]
    ch = alt.Chart(counts).mark_bar().encode(
        x=alt.X("Risk:N", sort=C.RISK_LABELS, title=None),
        y=alt.Y("Students:Q"),
        color=alt.Color("Risk:N", scale=_RISK_SCALE, legend=None),
        tooltip=["Risk", "Students"]).properties(height=height)
    st.altair_chart(ch)


def proba_bar(row, height=200):
    p = pd.DataFrame({"Risk": C.RISK_LABELS, "Probability %": [row[f"P({c}) %"] for c in C.RISK_LABELS]})
    ch = alt.Chart(p).mark_bar().encode(
        x=alt.X("Risk:N", sort=C.RISK_LABELS, title=None),
        y=alt.Y("Probability %:Q", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("Risk:N", scale=_RISK_SCALE, legend=None),
        tooltip=["Risk", "Probability %"]).properties(height=height)
    st.altair_chart(ch)


# ---------------------------------------------------------------- warnings
def warning_panel(df, subject, key):
    """Edit contacts, choose channels, preview and send warnings for the rows in df."""
    status = notify.config_status(st.secrets)
    with st.expander("Send warnings (email / SMS / WhatsApp)", expanded=False):
        st.caption("Email: " + ("configured" if status["email"] else "not configured")
                   + "  |  SMS (Twilio): " + ("configured" if status["sms"] else "not configured")
                   + "  |  WhatsApp links need no setup.")
        d = df.copy()
        for c in C.CONTACT_COLS:
            if c not in d:
                d[c] = ""
            d[c] = d[c].fillna("").astype(str)
        d.insert(0, "Send", d["Risk Class"].isin(["High", "Critical"]))
        show = [c for c in ["Send", "Student ID", "Name", "Risk Class", "Predicted Total"] if c in d.columns] + C.CONTACT_COLS
        edited = st.data_editor(d[show], key=f"{key}_editor", hide_index=True,
                                disabled=[c for c in show if c not in ["Send"] + C.CONTACT_COLS])
        channels = st.multiselect("Channels", ["Email - student", "Email - guardian", "SMS - guardian",
                                               "WhatsApp link - guardian"], default=["Email - guardian"],
                                  key=f"{key}_ch")
        subj = st.text_input("Email subject", f"Academic warning - {subject}", key=f"{key}_subj")
        tpl = st.text_area("Message template", notify.DEFAULT_TEMPLATE, height=260, key=f"{key}_tpl",
                           help="Placeholders: {recipient} {name} {sid} {subject} {pred_final} {pred_total} {risk} {conf}")
        sel = edited[edited["Send"]]
        if len(sel):
            first = d.loc[sel.index[0]]
            with st.container(border=True):
                st.caption("Preview (first selected student)")
                st.text(notify.render(tpl, first, subject))
        if st.button(f"Send to {len(sel)} selected student(s)", key=f"{key}_send", type="primary",
                     disabled=len(sel) == 0 or not channels):
            log, links = [], []
            for i, r in sel.iterrows():
                full = d.loc[i]
                for ch in channels:
                    to = r["student_email"] if ch == "Email - student" else \
                         r["guardian_email"] if ch == "Email - guardian" else r["guardian_phone"]
                    who = "Student" if ch == "Email - student" else "Parent/Guardian"
                    body = notify.render(tpl, full, subject, recipient=who)
                    name = str(full.get("Name", full.get("Student ID", i)))
                    if not str(to).strip():
                        log.append((name, ch, "skipped - no contact")); continue
                    try:
                        if ch.startswith("Email"):
                            notify.send_email(to, subj, body, st.secrets)
                        elif ch.startswith("SMS"):
                            notify.send_sms(to, body, st.secrets)
                        else:
                            links.append((name, notify.whatsapp_link(to, body)))
                            log.append((name, ch, "link ready")); continue
                        log.append((name, ch, "sent"))
                    except Exception as e:
                        log.append((name, ch, f"FAILED: {e}"))
            st.dataframe(pd.DataFrame(log, columns=["Student", "Channel", "Result"]), hide_index=True)
            for n, link in links:
                st.link_button(f"Open WhatsApp for {n}", link)
