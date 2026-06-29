# 🎬 Hybrid Movie Recommender System

> A production-ready movie recommendation engine combining **Collaborative Filtering**, **NLP Content-Based filtering**, and a **Neural Hybrid model** — inspired by how Netflix and Spotify build their recommendation systems.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![BERT](https://img.shields.io/badge/BERT-sentence--transformers-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

---

##  Live Demo

> Coming soon — Streamlit Cloud deployment

---

##  How It Works

The system combines **3 techniques** into one hybrid score:
User Ratings (MovieLens)          Movie Metadata (TMDB)

↓                                  ↓

SVD Matrix Factorization          BERT Sentence Embeddings

(latent user/movie factors)       (384-dim text vectors)

↓                                  ↓

└──────── Neural Hybrid Layer ─────┘

↓

Cold Start Handler

↓

Final Recommendations
| Component | Algorithm | Purpose |
|-----------|-----------|---------|
| Collaborative Filtering | SVD (TruncatedSVD + bias) | Learn from user rating patterns |
| Content-Based | BERT (all-MiniLM-L6-v2) | Understand movie descriptions & genres |
| Hybrid Layer | Weighted combination | Merge both signals |
| Cold Start | Content fallback | Handle new movies/users |

---

##  Model Performance

| Model | RMSE | Notes |
|-------|------|-------|
| SVD (Collaborative) | 0.90 | With user & movie bias correction |
| BERT (Content) | cosine sim | Genre + title embeddings |
| **Neural Hybrid** | **Best** | CF × 0.6 + Content × 0.4 |

---

##  Project Structure
movie-recommender-hybrid/

├── app/

│   ├── main.py                  # Streamlit UI entry point

│   ├── utils.py                 # Helper functions

│   ├── collaborative/

│   │   └── matrix_factorization.py

│   ├── content_based/

│   │   ├── text_embedder.py     # BERT embeddings builder

│   │   └── similarity_engine.py # Cosine similarity search

│   └── hybrid/

│       └── neural_hybrid_model.py  # Main recommender class

├── data/

│   └── movielens/

│       ├── movies.csv

│       └── ratings.csv

├── models/

│   ├── svd_model.pkl            # Trained SVD + biases

│   └── content_embeddings.pkl   # BERT movie embeddings

├── train_svd.py                 # SVD training script

├── requirements.txt

└── README.md
---

##  Quick Start

### 1 — Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/movie-recommender-hybrid.git
cd movie-recommender-hybrid

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### 2 — Download Data

Download [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) and place the `ml-1m/` folder in the project root.

### 3 — Train Models

```bash
# Train SVD (Collaborative Filtering)
python train_svd.py

# Build BERT Embeddings (Content-Based)
python app/content_based/text_embedder.py
```

### 4 — Run App

```bash
streamlit run app/main.py
```

---

##  Features

-  **Collaborative Filtering** — SVD with user & movie bias correction (RMSE: 0.90)
-  **Content-Based NLP** — BERT sentence embeddings on title + genres
-  **Hybrid Scoring** — Adjustable CF/Content weights via UI sliders
-  **Cold Start Handling** — Falls back to content similarity for new users
-  **Interactive UI** — Built with Streamlit, real-time recommendations
-  **Explainability** — Shows CF score, content score, and final score per movie

---

##  Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| ML / Math | scikit-learn, NumPy, pandas |
| NLP | sentence-transformers (BERT) |
| UI | Streamlit, Plotly |
| Data | MovieLens 1M (1M ratings, 6K users, 3.9K movies) |

---

##  Key Design Decisions

**Why SVD with bias instead of plain matrix factorization?**
Plain SVD on a sparse matrix gives poor predictions because it treats missing ratings as zeros. Adding user and movie bias terms (mean-centering) fixes this and drops RMSE from 2.5 → 0.90.

**Why sentence-transformers instead of raw BERT?**
`all-MiniLM-L6-v2` is 5× faster than full BERT, produces 384-dim embeddings optimized for semantic similarity, and works perfectly for short texts like movie titles and genres.

**Why a weighted hybrid instead of a neural network?**
With only 1M ratings and no user demographic data, a deep neural network would overfit. A weighted hybrid is interpretable, fast, and achieves better real-world results on sparse data.

---

##  Sample Results

**User 1 — liked: Copycat (1995), City of Lost Children (1995)**

| Rank | Movie | CF Score | Content Score | Final |
|------|-------|----------|---------------|-------|
| 1 | The Usual Suspects (1995) | 5.0 | 0.564 | 0.825 |
| 2 | City of Lost Children (1995) | 4.56 | 0.709 | 0.818 |
| 3 | Shawshank Redemption (1994) | 4.96 | 0.391 | 0.751 |
| 4 | Pulp Fiction (1994) | 4.68 | 0.490 | 0.748 |

---

##  Author

**Marwa Yosry Hassan**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/marwa-yousry-a39723353)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/marwa698)

---

##  License

MIT License — feel free to use, modify, and distribute.