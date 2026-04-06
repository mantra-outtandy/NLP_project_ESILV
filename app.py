import streamlit as st

st.set_page_config(
    page_title="InsureView",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif !important; }
.stApp { background: #ffffff; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #111111 !important;
    min-width: 220px !important; max-width: 220px !important;
}
section[data-testid="stSidebar"] > div { padding: 32px 24px !important; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span { color: #ffffff !important; }
section[data-testid="stSidebar"] .stRadio > label { display: none; }
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap: 4px !important; }
section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] span:last-child {
    font-size: 0.88rem !important; font-weight: 400 !important; color: #aaaaaa !important;
}

/* ── Hide ALL Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }
section[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="baseButton-header"] { display: none !important; }
button[data-testid="baseButton-header"] { display: none !important; }
section[data-testid="stSidebar"] > div > div > div > button { display: none !important; }

/* ── Main content ── */
.main .block-container { padding: 40px 48px !important; max-width: 1100px !important; }

/* ── Typography ── */
.page-title { font-size: 1.75rem; font-weight: 700; color: #111111; letter-spacing: -0.03em; margin-bottom: 4px; }
.page-sub { font-size: 0.875rem; color: #888888; margin-bottom: 32px; }

/* ── Cards ── */
.card { background: #f8f8f8; border-radius: 8px; padding: 20px 24px; margin-bottom: 16px; }
.card-dark { background: #111111; border-radius: 8px; padding: 24px 28px; margin-bottom: 16px; }

/* ── Metric ── */
.metric-label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: #999999; margin-bottom: 8px; }
.metric-value { font-size: 2rem; font-weight: 700; color: #111111; letter-spacing: -0.03em; line-height: 1.1; }

/* ── Pills ── */
.pill { display: inline-block; padding: 3px 10px; border-radius: 3px; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }
.pill-positive { background: #111111; color: #ffffff; }
.pill-negative { background: #f0f0f0; color: #111111; }
.pill-neutral  { background: #f0f0f0; color: #111111; }

/* ── Stars ── */
.stars { font-size: 1rem; letter-spacing: 2px; color: #111111; }

/* ── Review rows ── */
.review-meta { font-size: 0.75rem; color: #999999; font-weight: 500; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.03em; }

/* ── Inputs ── */
textarea, input[type="text"] {
    border-radius: 6px !important; border: 1px solid #e0e0e0 !important;
    font-size: 0.9rem !important; color: #111111 !important; background: #ffffff !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #111111 !important; box-shadow: none !important; outline: none !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #111111 !important; color: #ffffff !important;
    border: none !important; border-radius: 6px !important;
    font-size: 0.85rem !important; font-weight: 500 !important;
    letter-spacing: 0.02em !important; padding: 9px 22px !important; height: auto !important;
}
.stButton > button:hover { background: #333333 !important; }

/* ── Selectbox ── */
div[data-baseweb="select"] > div {
    border-radius: 6px !important; border: 1px solid #e0e0e0 !important;
    background: #ffffff !important; font-size: 0.88rem !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid #e5e5e5 !important; gap: 0 !important; }
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border: none !important;
    border-bottom: 2px solid transparent !important; border-radius: 0 !important;
    color: #999999 !important; font-size: 0.85rem !important; font-weight: 500 !important; padding: 10px 20px !important;
}
.stTabs [aria-selected="true"] { color: #111111 !important; border-bottom: 2px solid #111111 !important; }

/* ── Expander: hide icon glyphs ── */
summary { list-style: none !important; }
summary::-webkit-details-marker { display: none !important; }
details > summary { font-size: 0.85rem !important; font-weight: 600 !important; color: #111111 !important; cursor: pointer; }
details > summary svg { display: none !important; }
[data-testid="stExpanderToggleIcon"] { display: none !important; }

/* ── Feature bars ── */
.bar-wrap { background: #f0f0f0; border-radius: 2px; height: 5px; margin-top: 4px; }
.bar-dark  { background: #111111; height: 5px; border-radius: 2px; }
.bar-light { background: #cccccc; height: 5px; border-radius: 2px; }

/* ── Section label ── */
.section-label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: #999999; margin-bottom: 12px; }

/* ── st.write text styling ── */
[data-testid="stText"] { font-size: 0.9rem !important; color: #111111 !important; line-height: 1.65 !important; }

/* ── HR ── */
hr { border: none; border-top: 1px solid #f0f0f0; margin: 12px 0; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### InsureView")
    st.markdown('<p style="font-size:0.7rem;color:#666;text-transform:uppercase;letter-spacing:0.1em;margin-top:-12px;">Review Analytics</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["Predict a Review", "Search Reviews", "Insurer Analysis"],
        label_visibility="collapsed"
    )

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.72rem;color:#555;line-height:1.7;">17,464 reviews<br>TF-IDF · BM25 · LogReg</p>', unsafe_allow_html=True)

# ── Pages ─────────────────────────────────────────────────────────────────────
if page == "Predict a Review":
    from _pages.predict import show; show()
elif page == "Search Reviews":
    from _pages.search import show; show()
elif page == "Insurer Analysis":
    from _pages.insurer import show; show()
