# InsureView

NLP Project 2 — ESILV DIA4  
Mantra Outtandy & Minji Park

---

InsureView is a Streamlit app for analysing insurance customer reviews. You can predict star ratings and sentiment, search reviews by keyword or meaning, explore insurer-level stats, and ask questions answered by a RAG pipeline.

## Pages

- **Predict a Review** — paste a review, get star rating (1–5), sentiment, category, and a word-level explanation of why the model predicted what it did
- **Search Reviews** — BM25 for exact keywords, or TF-IDF cosine similarity for semantic search
- **Insurer Analysis** — per-insurer dashboard with rating distribution, topic breakdown, and sample reviews
- **Ask a Question** — RAG pipeline: retrieves relevant reviews then generates an answer using BART

## Project structure

```
├── app.py              main entry point
├── utils.py            preprocessing, model loading, search, predictions
├── startup.py          downloads data files from Google Drive on first launch
├── requirements.txt
├── _pages/
│   ├── predict.py
│   ├── search.py
│   ├── insurer.py
│   └── rag.py
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

> The RAG page needs `transformers` and `torch` (~2GB). Skip those in `requirements.txt` if you don't need it.

## Models

| Task | Model | F1 |
|---|---|---|
| Star rating | TF-IDF + LogReg | 0.47 |
| Sentiment | TF-IDF + LogReg | 0.67 |
| Category | TF-IDF + LogReg (trained on SBERT-generated labels) | — |

TF-IDF beats every neural network we tried — BiLSTM, CNN, GloVe, SBERT — because the reviews are short and contain very discriminative phrases. Word2Vec mean vectors score the same as the dummy baseline (F1=0.09) since averaging destroys polarity signals.

The neutral class is the known weak spot (F1=0.30 across all models). Three-star reviews mix positive and negative vocabulary in the same sentence, making them hard to separate regardless of the model.
