import os
import re
import pickle
import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent / "data"))


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING — identical to training
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_text(text):
    """Normalise text exactly as done during model training."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_text(text):
    """Strip HTML for display only."""
    text = str(text)
    while re.search(r'<[^>]+>', text):
        m = re.search(r'class=["\']?review-text["\']?[^>]*>(.*?)(?:</div>|$)',
                      text, re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1)
        else:
            text = re.sub(r'<[^>]+>', '', text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# HTML HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def stars_html(n):
    try:
        n = int(round(float(n)))
    except Exception:
        return ""
    return (f'<span class="stars">{"★" * n}'
            f'<span style="color:#d0d0d0">{"☆" * (5 - n)}</span></span>')


def sentiment_pill(s):
    if not s or str(s) == "nan":
        return ""
    s = str(s).lower()
    cls = {"positive": "pill-positive", "negative": "pill-negative", "neutral": "pill-neutral"}
    return f'<span class="pill {cls.get(s, "pill-neutral")}">{s}</span>'


def tokenise(text):
    clean = preprocess_text(text)
    return [w for w in clean.split() if len(w) > 2]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_tfidf_star():
    p = DATA_DIR / "tfidf_logreg_5class.pkl"
    if not p.exists(): return None
    with open(p, "rb") as f: return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_tfidf_sentiment():
    p = DATA_DIR / "tfidf_logreg_sentiment.pkl"
    if not p.exists(): return None
    with open(p, "rb") as f: return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_label_encoder():
    p = DATA_DIR / "sentiment_label_encoder.pkl"
    if not p.exists(): return None
    with open(p, "rb") as f: return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_category_model():
    """Load the TF-IDF category classifier trained on SBERT labels.
    No SBERT needed at inference — pure sklearn pipeline, instant."""
    p = DATA_DIR / "tfidf_category.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_category_data():
    return None, None


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_dataframe():
    for fname in ["test_predictions.csv", "data_topics_embeddings.csv"]:
        path = DATA_DIR / fname
        if path.exists():
            needed = ["note", "assureur", "produit", "avis_cor_en", "avis_en", "avis",
                      "topic_label", "dominant_topic", "label_3", "sent_true",
                      "predicted_category", "type"]
            try:
                df = pd.read_csv(path, usecols=lambda c: c in needed, low_memory=True)
            except Exception:
                df = pd.read_csv(path, low_memory=True)
            for col in ["avis_cor_en", "avis_en", "avis"]:
                if col in df.columns:
                    df["_text"]       = df[col].fillna("").astype(str).apply(clean_text)
                    df["_text_clean"] = df[col].fillna("").astype(str).apply(preprocess_text)
                    break
            return df
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SEARCH — BM25 (keywords) + TF-IDF cosine (fast semantic, no extra model)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_bm25(df):
    try:
        from rank_bm25 import BM25Okapi
        corpus = [tokenise(t) for t in df["_text_clean"].tolist()]
        return BM25Okapi(corpus)
    except Exception:
        return None


def search_bm25(df, query, n=10):
    """BM25 keyword search. Returns (indices, scores)."""
    bm25 = load_bm25(df)
    if bm25 is None: return [], np.array([])
    tokens = tokenise(query)
    if not tokens: return [], np.array([])
    scores  = bm25.get_scores(tokens)
    top_idx = [int(i) for i in np.argsort(scores)[::-1][:n] if scores[i] > 0]
    return top_idx, scores


def search_cosine(df, query, star_model, n=10):
    """
    TF-IDF cosine similarity search — reuses the star model's vectorizer.
    Always computed on the current df (no caching) to handle filters correctly.
    """
    if star_model is None: return [], np.array([])
    try:
        from sklearn.preprocessing import normalize
        vectorizer = star_model.named_steps["tfidf"]
        texts  = df["_text_clean"].tolist()
        matrix = vectorizer.transform(texts)
        matrix = normalize(matrix, norm="l2")
        qvec   = vectorizer.transform([preprocess_text(query)])
        qvec   = normalize(qvec, norm="l2")
        sims   = (matrix @ qvec.T).toarray().flatten()
        top_idx = [int(i) for i in np.argsort(sims)[::-1][:n] if sims[i] > 0.05]
        return top_idx, sims
    except Exception:
        return [], np.array([])


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict_star(model, text):
    if model is None: return None
    return int(model.predict([preprocess_text(text)])[0])


def predict_sentiment(model, le, text):
    if model is None or le is None: return None
    raw = model.predict([preprocess_text(text)])[0]
    return le.inverse_transform([raw])[0]


def predict_category(text, cat_vecs=None, cat_names=None):
    """
    Predict review category using the TF-IDF classifier trained on SBERT labels.
    No SBERT at inference — pure sklearn, instant.
    Falls back to keyword matching if tfidf_category.pkl is not found.
    """
    clean = preprocess_text(text)
    model = load_category_model()
    if model is not None:
        try:
            return str(model.predict([clean])[0])
        except Exception:
            pass
    # Fallback: keyword matching
    rules = {
        "Pricing":          ["price", "expensive", "cost", "premium", "rate", "fee"],
        "Claims":           ["claim", "accident", "damage", "repair", "expert", "compensation"],
        "Customer Service": ["phone", "call", "advisor", "wait", "response", "email", "contact"],
        "Coverage":         ["reimburse", "reimbursement", "refund", "repay", "coverage", "dental"],
        "Cancellation":     ["cancel", "terminate", "flee", "quit", "leave", "switch"],
        "Enrolment":        ["subscribe", "subscription", "quote", "contract", "join"],
    }
    scores = {cat: sum(1 for kw in kws if kw in clean) for cat, kws in rules.items()}
    best   = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General"


# ══════════════════════════════════════════════════════════════════════════════
# EXPLANATION
# ══════════════════════════════════════════════════════════════════════════════

def top_tfidf_features(model, text, n=8, class_index=None):
    if model is None: return [], []
    try:
        tfidf      = model.named_steps["tfidf"]
        clf        = model.named_steps["clf"]
        vec        = tfidf.transform([preprocess_text(text)])
        feat_names = tfidf.get_feature_names_out()
        indices    = vec.nonzero()[1]
        if len(indices) == 0: return [], []
        if class_index is not None:
            coef_idx = class_index
        else:
            pred_label = int(clf.predict(vec)[0])
            coef_idx   = int(np.where(clf.classes_ == pred_label)[0][0])
        coefs = clf.coef_[coef_idx]
        contributions = [(feat_names[i], float(vec[0, i] * coefs[i])) for i in indices]
        contributions.sort(key=lambda x: x[1], reverse=True)
        pos = [s for s in contributions if s[1] > 0][:n]
        neg = sorted([s for s in contributions if s[1] < 0], key=lambda x: x[1])[:n]
        return pos, neg
    except Exception:
        return [], []


def top_tfidf_features_sentiment(model, le, text, n=8):
    if model is None or le is None: return [], [], ""
    try:
        vec        = model.named_steps["tfidf"].transform([preprocess_text(text)])
        sent_enc   = int(model.named_steps["clf"].predict(vec)[0])
        sent_label = le.inverse_transform([sent_enc])[0]
        pos, neg   = top_tfidf_features(model, text, n=n, class_index=sent_enc)
        return pos, neg, sent_label
    except Exception:
        return [], [], ""
