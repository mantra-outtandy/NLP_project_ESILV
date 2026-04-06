import streamlit as st
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import (
    load_tfidf_star, load_tfidf_sentiment, load_label_encoder,
    predict_star, predict_sentiment, predict_category,
    top_tfidf_features, sentiment_pill,
)


def render_bars(feats, bar_class, sign):
    if not feats:
        st.markdown('<p style="font-size:0.82rem;color:#cccccc;">No strong signal.</p>',
                    unsafe_allow_html=True)
        return
    max_val = max(abs(v) for _, v in feats) or 1
    for word, val in feats:
        pct = int(abs(val) / max_val * 100)
        st.markdown(f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="font-size:0.83rem;color:#111111;">{word}</span>
                <span style="font-size:0.75rem;color:#999999;">{sign}{abs(val):.3f}</span>
            </div>
            <div class="bar-wrap"><div class="{bar_class}" style="width:{pct}%;"></div></div>
        </div>""", unsafe_allow_html=True)


def show():
    st.markdown('<div class="page-title">Predict a Review</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Enter any insurance review for an instant prediction.</div>',
        unsafe_allow_html=True)

    with st.spinner("Loading models…"):
        star_model = load_tfidf_star()
        sent_model = load_tfidf_sentiment()
        le         = load_label_encoder()

    if star_model is None:
        st.error("Model files not found. Place .pkl files in the data/ folder.")
        return

    user_text = st.text_area(
        "Review",
        placeholder="Type or paste an insurance review here…",
        height=130,
        label_visibility="collapsed",
    )
    run = st.button("Analyse")

    if not run or not user_text.strip():
        st.markdown("""
        <div style="border:1px dashed #e0e0e0;border-radius:8px;padding:48px 24px;
                    text-align:center;margin-top:24px;">
            <p style="color:#aaaaaa;font-size:0.88rem;margin:0;">
                Enter a review above and click Analyse
            </p>
        </div>""", unsafe_allow_html=True)
        return

    active = user_text.strip()

    star      = predict_star(star_model, active)
    sentiment = predict_sentiment(sent_model, le, active)
    category  = predict_category(active)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Predicted stars</div>
            <div class="metric-value">{star} / 5</div>
            <div style="margin-top:10px;font-size:1.1rem;letter-spacing:3px;">
                {'★' * star}<span style="color:#dddddd;">{'★' * (5 - star)}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Sentiment</div>
            <div style="margin-top:14px;">{sentiment_pill(sentiment) if sentiment else '—'}</div>
            <div style="font-size:0.72rem;color:#aaaaaa;margin-top:12px;">TF-IDF · LogReg · F1 0.67</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Category</div>
            <div style="font-size:1rem;font-weight:600;color:#111111;margin-top:14px;">{category}</div>
            <div style="font-size:0.72rem;color:#aaaaaa;margin-top:12px;">Keyword matching</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="background:#f8f8f8;border-radius:8px;padding:20px 24px;">', unsafe_allow_html=True)
    st.markdown('<div class="section-label" style="margin-bottom:8px;">Why this prediction?</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="font-size:0.82rem;color:#888888;margin-bottom:20px;margin-top:0;">'
        f'Words contributing toward <strong>{star} stars</strong>. '
        f'Dark = supports this rating. Light = contradicts it.</p>',
        unsafe_allow_html=True)
    pos_feats, neg_feats = top_tfidf_features(star_model, active)
    ec1, ec2 = st.columns(2, gap="large")
    with ec1:
        st.markdown('<div class="section-label">Positive influence</div>',
                    unsafe_allow_html=True)
        render_bars(pos_feats, "bar-dark", "+")
    with ec2:
        st.markdown('<div class="section-label">Negative influence</div>',
                    unsafe_allow_html=True)
        render_bars(neg_feats, "bar-light", "")
    st.markdown('</div>', unsafe_allow_html=True)
