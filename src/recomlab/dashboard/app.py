"""Streamlit dashboard: ``streamlit run src/recomlab/dashboard/app.py``.

Renders view models from :mod:`recomlab.dashboard.views`; it never trains models
or computes metrics itself.
"""

from __future__ import annotations

import streamlit as st

from recomlab.config.settings import load_settings
from recomlab.dashboard import views
from recomlab.services.bootstrap import build_experiments, build_service
from recomlab.services.recommendation import NoActiveModelError

PAGES = ("Leaderboard", "Run details", "Recommendation explorer", "Cold start & beyond-accuracy")

st.set_page_config(page_title="RecomLab", layout="wide")
settings = load_settings()
experiments = build_experiments(settings)
service = build_service(settings)

st.title("RecomLab")
page = st.sidebar.radio("Page", PAGES)

if page == "Leaderboard":
    st.header("Experiment leaderboard")
    rows = views.leaderboard_rows(experiments)
    if rows:
        st.dataframe(rows, width="stretch")
    else:
        st.info("No runs recorded yet.")

elif page == "Run details":
    st.header("Run details")
    runs = [r["run_id"] for r in views.leaderboard_rows(experiments)]
    if not runs:
        st.info("No runs recorded yet.")
    else:
        detail = views.run_detail(experiments, st.selectbox("Run", runs))
        for warning in detail["warnings"]:
            st.warning(warning)
        st.subheader("Identity")
        st.json(detail["identity"])
        left, right = st.columns(2)
        left.subheader("Candidate stage (retrieval)")
        left.json(detail["candidate_stage"] or {"note": "not recorded"})
        right.subheader("Ranking stage")
        right.json(detail["ranking_stage"])
        st.subheader("Evaluation protocol")
        st.json(detail["evaluation"])

elif page == "Recommendation explorer":
    st.header("Recommendation explorer (demo ids)")
    users = views.demo_user_ids(service)
    user = st.selectbox("User", users) if users else None
    k = st.slider("k", 1, 20, 5)
    if user:
        try:
            result = views.explorer(service, user, k)
            st.caption(f"bundle {result['bundle_id']} · candidates {result['n_candidates']}")
            st.dataframe(result["items"], width="stretch")
        except NoActiveModelError:
            st.warning("No active model bundle.")

else:
    st.header("Cold start")
    try:
        cold = views.cold_start_view(service)
        st.dataframe(cold["cold_user_recommendations"], width="stretch")
        st.caption(f"Cold catalog items: {', '.join(cold['cold_items']) or 'none'}")
    except NoActiveModelError:
        st.warning("No active model bundle.")
    st.header("Beyond-accuracy")
    st.dataframe(views.beyond_accuracy_view(experiments), width="stretch")
