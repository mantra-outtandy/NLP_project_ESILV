# InsureView

NLP Project 2 — ESILV DIA4  
Mantra Outtandy & Minji Park

---

InsureView is a Streamlit app for analysing insurance customer reviews. It predicts star ratings and sentiment, classifies reviews by category, enables keyword and semantic search, and provides an insurer-level analytics dashboard.

## Pages

- **Predict a Review** — paste a review, get star rating (1–5), sentiment, category, and a word-level explanation of the prediction
- **Search Reviews** — BM25 for exact keyword matching, or TF-IDF cosine similarity for semantic search
- **Insurer Analysis** — per-insurer dashboard with rating distribution, topic breakdown, and sample reviews

## Project structure

\```
├── app.py              main entry point
├── utils.py            preprocessing, model loading, search, predictions
├── requirements.txt
├── _pages/
│   ├── predict.py
│   ├── search.py
│   └── insurer.py
└── data/
    ├── tfidf_logreg_5class.pkl
    ├── tfidf_logreg_sentiment.pkl
    ├── tfidf_category.pkl
    ├── sentiment_label_encoder.pkl
    ├── category_vectors.npy
    ├── category_names.json
    └── data_topics_embeddings.csv
\```

## Run locally

\```bash
git clone https://github.com/mantra-outtandy/NLP_project_ESILV.git
cd NLP_project_ESILV
pip install -r requirements.txt
streamlit run app.py
\```

The app opens at `http://localhost:8501`.

## Models

| Task | Model | F1 |
|---|---|---|
| Star rating (1–5) | TF-IDF + LogReg | 0.47 |
| Sentiment | TF-IDF + LogReg | 0.67 |
| Category | TF-IDF + LogReg (labels from SBERT) | — |
| Search | BM25 + TF-IDF cosine | — |

TF-IDF outperforms every neural network we tried — BiLSTM, CNN, GloVe, SBERT — because the reviews are short and contain very discriminative phrases. Word2Vec mean vectors score the same as a random baseline (F1=0.09) since averaging word vectors destroys polarity signals.

## A note on accuracy

Some predictions will not be perfectly accurate. This is a known limitation and a deliberate trade-off: we chose models that are fast enough to run without a GPU, at the cost of some accuracy. Heavier models like fine-tuned BERT would perform better but would take too long to load in a demo context.

The hardest cases are mixed reviews (3-star) and sarcasm. Three-star reviews blend positive and negative vocabulary in the same sentence — even BERT scored F1=0.30 on that class, so this is a fundamental data problem rather than a model choice. Sarcasm fails because TF-IDF treats each word independently and has no concept of tone.
