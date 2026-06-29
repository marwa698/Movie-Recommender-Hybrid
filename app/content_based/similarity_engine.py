import numpy as np
import joblib, os
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

class SimilarityEngine:
    def __init__(self):
        print("📂 Loading content embeddings...")
        data = joblib.load(os.path.join(MODELS_DIR, 'content_embeddings.pkl'))
        self.embeddings = data['embeddings']
        self.movie_ids  = data['movie_ids']
        self.titles     = data['titles']
        self.genres     = data['genres']
        self.id2idx     = {mid: i for i, mid in enumerate(self.movie_ids)}
        print(f"✅ Loaded {len(self.movie_ids)} movies")

    def get_similar_movies(self, movie_id, top_n=10):
        idx = self.id2idx.get(movie_id)
        if idx is None:
            return []

        movie_vec = self.embeddings[idx].reshape(1, -1)
        scores    = cosine_similarity(movie_vec, self.embeddings)[0]
        top_idxs  = np.argsort(scores)[::-1][1:top_n+1]  # بدون نفس الفيلم

        return [
            {
                'movie_id': self.movie_ids[i],
                'title':    self.titles[i],
                'genres':   self.genres[i],
                'score':    float(scores[i])
            }
            for i in top_idxs
        ]

if __name__ == '__main__':
    engine = SimilarityEngine()

    # تجربة بـ Toy Story (movieId=1)
    print("\n🎬 أفلام مشابهة لـ Toy Story:")
    results = engine.get_similar_movies(movie_id=1, top_n=5)
    for r in results:
        print(f"  {r['title']:40s} | score: {r['score']:.3f} | {r['genres']}")