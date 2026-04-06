import streamlit as st
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import load_dataframe, stars_html, sentiment_pill, clean_text

def show():
    st.markdown('<div class="page-title">Insurer Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Aggregated stats and reviews by insurer.</div>', unsafe_allow_html=True)

    df = load_dataframe()
    if df is None:
        st.error("Dataset not found. Place data_topics_embeddings.csv in the data/ folder.")
        return

    labelled = df[df["note"].notna()] if "note" in df.columns else df
    counts   = labelled["assureur"].value_counts()
    selected = st.selectbox("Insurer", counts.index.tolist(),
        format_func=lambda x: f"{x}   ({counts[x]:,} reviews)",
        label_visibility="collapsed")

    subset   = labelled[labelled["assureur"] == selected].copy()
    n        = len(subset)
    avg      = subset["note"].mean() if "note" in subset.columns else None
    sent_col = next((c for c in ["label_3", "sent_true"] if c in subset.columns), None)
    if sent_col:
        sc      = subset[sent_col].value_counts().to_dict()
        pos_pct = sc.get("positive", 0) / n * 100 if n else 0
        neg_pct = sc.get("negative", 0) / n * 100 if n else 0
    elif "note" in subset.columns:
        pos_pct = (subset["note"] >= 4).sum() / n * 100 if n else 0
        neg_pct = (subset["note"] <= 2).sum() / n * 100 if n else 0
    else:
        pos_pct = neg_pct = 0

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    for col, label, value in zip(
        [c1, c2, c3, c4],
        ["Reviews", "Avg Rating", "Positive", "Negative"],
        [f"{n:,}", f"{avg:.2f}" if avg else "—", f"{pos_pct:.0f}%", f"{neg_pct:.0f}%"]
    ):
        with col:
            st.markdown(f"""
            <div class="card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ch1, ch2 = st.columns(2, gap="large")
    with ch1:
        if "note" in subset.columns:
            st.markdown('<div class="section-label">Rating distribution</div>', unsafe_allow_html=True)
            rd = subset["note"].value_counts().sort_index()
            rd.index = [f"{int(i)} star{'s' if i > 1 else ''}" for i in rd.index]
            st.bar_chart(rd, height=200, color="#111111")
    with ch2:
        topic_col = next((c for c in ["topic_label", "dominant_topic"] if c in subset.columns), None)
        if topic_col:
            st.markdown('<div class="section-label">Topic distribution</div>', unsafe_allow_html=True)
            td = subset[topic_col].value_counts().head(5)
            td.index = [str(t)[:30] for t in td.index]
            st.bar_chart(td, height=200, color="#111111")

    if "predicted_category" in subset.columns and "note" in subset.columns:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Average rating by category</div>', unsafe_allow_html=True)
        cat_data = subset.groupby("predicted_category")["note"].agg(["mean", "count"]).round(2).sort_values("mean")
        for cat, row_d in cat_data.iterrows():
            pct = int(row_d["mean"] / 5 * 100)
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid #f5f5f5;">
                <span style="font-size:0.88rem;font-weight:500;color:#111111;">{cat}</span>
                <div style="display:flex;align-items:center;gap:16px;">
                    <div style="width:100px;background:#f0f0f0;border-radius:2px;height:5px;">
                        <div style="background:#111111;width:{pct}%;height:5px;border-radius:2px;"></div>
                    </div>
                    <span style="font-size:0.8rem;color:#999999;width:90px;text-align:right;">
                        {row_d['mean']:.2f} / 5 &nbsp; ({int(row_d['count'])})
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Best reviews   (4–5 stars)", "Worst reviews   (1–2 stars)"])

    def show_reviews(sub_df):
        for _, row in sub_df.head(5).iterrows():
            text = clean_text(row.get("_text", ""))[:320]
            note = row.get("note", None)
            sent = str(row.get(sent_col, "") or "") if sent_col else ""
            st.markdown(f"""
            <div style="padding-top:16px;display:flex;gap:10px;align-items:center;margin-bottom:6px;">
                {stars_html(note) if note and str(note) != 'nan' else ''}
                {sentiment_pill(sent) if sent and sent != 'nan' else ''}
            </div>""", unsafe_allow_html=True)
            st.write(text)
            st.markdown('<div style="border-top:1px solid #f0f0f0;margin:8px 0 0 0;"></div>', unsafe_allow_html=True)

    with tab1:
        best = subset[subset["note"] >= 4].sample(min(5, len(subset[subset["note"] >= 4])), random_state=42) if "note" in subset.columns else subset.head(5)
        if best.empty: st.info("No 4–5 star reviews.")
        else: show_reviews(best)

    with tab2:
        worst = subset[subset["note"] <= 2].sample(min(5, len(subset[subset["note"] <= 2])), random_state=42) if "note" in subset.columns else subset.tail(5)
        if worst.empty: st.info("No 1–2 star reviews.")
        else: show_reviews(worst)
