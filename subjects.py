import streamlit as st
from sws import store


def render(subjects):
    st.subheader("Subjects")
    for s in subjects:
        b = store.load_bundle(s)
        c1, c2, c3 = st.columns([3, 2, 1])
        c1.markdown(f"**{s}**")
        c2.write(f"{b['meta']['n_rows']} students, {len(b['meta']['feats'])} features")
        if c3.button("Delete", key=f"del_{s}"):
            st.session_state["confirm_del"] = s
    d = st.session_state.get("confirm_del")
    if d:
        st.warning(f"Delete subject '{d}' with its data and model?")
        a, b_ = st.columns(2)
        if a.button("Yes, delete"):
            store.delete_subject(d)
            st.session_state.pop("confirm_del")
            st.rerun()
        if b_.button("Cancel"):
            st.session_state.pop("confirm_del")
            st.rerun()
    st.info("To add a subject use 'Upload data' -> '+ New subject'. To add rows to a subject use 'Upload data' -> "
            "choose the subject -> 'Add to training data and retrain'.")
