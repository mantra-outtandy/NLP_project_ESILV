import streamlit as st
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import (
    load_dataframe, load_tfidf_star,
    search_bm25, search_cosine,
    stars_html, sentiment_pill, clean_text,
)


def show():
    st.markdown('<div class="page-title">Search Reviews</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Find reviews by keywords (BM25) or by meaning (cosine similarity).</div>',
                unsafe_allow_html=True)

    df         = load_dataframe()
    star_model = load_tfidf_star()

    if df is None:
        st.error("Dataset not found. Place data_topics_embeddings.csv in the data/ folder.")
        return

    # ── Query + mode ──────────────────────────────────────────────────────────
    q_col, mode_col = st.columns([4, 2], gap="medium")
    with q_col:
        query = st.text_input("Query",
            placeholder="e.g. waiting months for reimbursement",
            label_visibility="collapsed")
    with mode_col:
        search_mode = st.radio("Mode",
            ["BM25 (keywords)", "Cosine (meaning)"],
            horizontal=True, label_visibility="collapsed")

    # ── Filters ───────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([3, 2, 1], gap="medium")
    with fc1:
        insurers = ["All insurers"]
        if "assureur" in df.columns:
            insurers += sorted(df["assureur"].dropna().unique().tolist())
        insurer_filter = st.selectbox("Insurer", insurers, label_visibility="collapsed")
    with fc2:
        star_filter = st.selectbox("Rating",
            ["All ratings", "1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
            label_visibility="collapsed")
    with fc3:
        n_results = st.selectbox("Show", [5, 10, 20], label_visibility="collapsed")

    run = st.button("Search")

    if not run or not query.strip():
        st.markdown("""
        <div style="border:1px dashed #e0e0e0;border-radius:8px;padding:48px 24px;
                    text-align:center;margin-top:24px;">
            <p style="color:#aaaaaa;font-size:0.88rem;margin:0;">Type a query and press Search</p>
            <p style="color:#cccccc;font-size:0.78rem;margin:8px 0 0 0;">
                BM25 = exact keywords &nbsp;·&nbsp; Cosine = similar meaning, no SBERT needed
            </p>
        </div>""", unsafe_allow_html=True)
        return

    # ── Apply filters ─────────────────────────────────────────────────────────
    filtered = df.copy()
    if insurer_filter != "All insurers" and "assureur" in filtered.columns:
        filtered = filtered[filtered["assureur"] == insurer_filter]
    if star_filter != "All ratings" and "note" in filtered.columns:
        filtered = filtered[filtered["note"] == int(star_filter[0])]
    filtered = filtered.reset_index(drop=True)

    if filtered.empty:
        st.warning("No reviews match the selected filters.")
        return

    # ── Search ────────────────────────────────────────────────────────────────
    use_cosine = "Cosine" in search_mode

    with st.spinner("Searching…"):
        if use_cosine and star_model is not None:
            top_idx, scores = search_cosine(filtered, query, star_model, n=n_results)
            score_label = "cosine sim"
        else:
            top_idx, scores = search_bm25(filtered, query, n=n_results)
            score_label = "BM25"
            use_cosine = False

    if not top_idx:
        st.info("No matching reviews found. Try different keywords or switch search mode.")
        return

    mode_label = "Cosine similarity" if use_cosine else "BM25"
    st.markdown(
        f'<p style="font-size:0.78rem;color:#999999;margin:16px 0 8px 0;">'
        f'{len(top_idx)} results &nbsp;·&nbsp; {mode_label}</p>',
        unsafe_allow_html=True)

    # ── Results ───────────────────────────────────────────────────────────────
    for rank, idx in enumerate(top_idx, 1):
        row       = filtered.iloc[idx]
        text      = clean_text(row.get("_text", ""))[:420]
        note      = row.get("note", None)
        ins       = str(row.get("assureur", "") or "")
        topic     = str(row.get("topic_label", "") or "")
        sent      = str(row.get("label_3", row.get("sent_true", "")) or "")
        stars_str = stars_html(note) if note and str(note) != "nan" else ""
        sent_str  = sentiment_pill(sent) if sent and sent != "nan" else ""
        score_val = float(scores[idx]) if len(scores) > idx else 0.0
        meta      = " · ".join(filter(None, [ins, topic,
                                              f"{score_label}: {score_val:.3f}"]))

        st.markdown(f"""
        <div style="padding:16px 0 8px 0;border-top:1px solid #f0f0f0;">
            <div class="review-meta">#{rank} &nbsp;·&nbsp; {meta}</div>
            <div style="display:flex;gap:10px;align-items:center;margin:6px 0;">
                {stars_str}{sent_str}
            </div>
        </div>""", unsafe_allow_html=True)
        st.write(text)
