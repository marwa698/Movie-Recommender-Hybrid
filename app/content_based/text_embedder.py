import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib, os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'movielens')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def build_movie_embeddings():
    print("📂 Loading movies...")
    movies = pd.read_csv(os.path.join(DATA_DIR, 'movies.csv'))

    # تجهيز النص = title + genres
    movies['text'] = movies['title'] + ' ' + movies['genres'].str.replace('|', ' ', regex=False)

    print("🤖 Loading sentence-transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print(f"🔨 Building embeddings for {len(movies)} movies...")
    embeddings = model.encode(
        movies['text'].tolist(),
        batch_size=64,
        show_progress_bar=True
    )

    # حفظ
    content_data = {
        'embeddings': embeddings,
        'movie_ids': movies['movieId'].tolist(),
        'titles': movies['title'].tolist(),
        'genres': movies['genres'].tolist()
    }
    joblib.dump(content_data, os.path.join(MODELS_DIR, 'content_embeddings.pkl'))
    print(f"✅ Embeddings shape: {embeddings.shape}")
    print("✅ models/content_embeddings.pkl saved!")
    return content_data

if __name__ == '__main__':
    build_movie_embeddings()