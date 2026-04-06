# InsureView

NLP Project 2 — ESILV DIA4  
Mantra Outtandy & Minji Park

---

InsureView is a Streamlit app for analysing insurance customer reviews. You can predict star ratings and sentiment, search reviews by keyword or meaning, and explore insurer-level stats.

## Pages

- **Predict a Review** — paste a review, get star rating (1–5), sentiment, category, and a word-level explanation of why the model predicted what it did
- **Search Reviews** — BM25 for exact keywords, or TF-IDF cosine similarity for semantic search
- **Insurer Analysis** — per-insurer dashboard with rating distribution, topic breakdown, and sample reviews

## Project structure

```
├── app.py              main entry point
├── utils.py            preprocessing, model loading, search, predictions
├── startup.py          downloads data files from Google Drive on first launch
├── requirements.txt
├── _pages/
│   ├── predict.py
│   ├── search.py
│   └── insurer.py
└── data/               not in repo — see setup below
```

## Run locally

```bash
git clone https://github.com/mantra-outtandy/NLP_project_ESILV.git
cd NLP_project_ESILV
pip install -r requirements.txt
```

Create a `data/` folder and add these files (available on request):

```
tfidf_logreg_5class.pkl
tfidf_logreg_sentiment.pkl
tfidf_category.pkl
sentiment_label_encoder.pkl
category_vectors.npy
category_names.json
data_topics_embeddings.csv
```

Or fill in the Google Drive IDs in `startup.py` and they will download automatically.

```bash
streamlit run app.py
```

## Models

| Task | Model | F1 |
|---|---|---|
| Star rating | TF-IDF + LogReg | 0.47 |
| Sentiment | TF-IDF + LogReg | 0.67 |
| Category | TF-IDF + LogReg (trained on SBERT-generated labels) | — |

TF-IDF beats every neural network we tried — BiLSTM, CNN, GloVe, SBERT — because the reviews are short and contain very discriminative phrases. Word2Vec mean vectors score the same as the dummy baseline (F1=0.09) since averaging destroys polarity signals.

## A note on accuracy

Some predictions may not be perfectly accurate, and that is expected. We had to make a deliberate trade-off: models that are accurate enough to be useful, but fast enough to run without a GPU or a paid API. Heavier models like fine-tuned BERT would improve results but take minutes to load and require significant compute — not realistic for a local Streamlit demo.

The known weak spots are mixed reviews (3-star) and sarcasm. Three-star reviews mix positive and negative vocabulary in the same sentence, which confuses any word-based model regardless of architecture — even BERT scored F1=0.30 on that class. Sarcasm is harder still, since words like "excellent" appear in negative reviews used ironically, and TF-IDF has no way to detect tone.

These are documented limitations, not oversights.
