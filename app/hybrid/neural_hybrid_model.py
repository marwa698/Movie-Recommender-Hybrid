import numpy as np
import joblib, os
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

class HybridRecommender:
    def __init__(self, cf_weight=0.6, content_weight=0.4):
        self.cf_weight      = cf_weight
        self.content_weight = content_weight

        print("📂 Loading SVD model...")
        svd_data = joblib.load(os.path.join(MODELS_DIR, 'svd_model.pkl'))
        self.reconstructed = svd_data['reconstructed_centered']
        self.user2idx      = svd_data['user2idx']
        self.movie2idx     = svd_data['movie2idx']
        self.idx2movie     = svd_data['idx2movie']
        self.global_mean   = svd_data['global_mean']
        self.user_bias     = svd_data['user_bias']
        self.movie_bias    = svd_data['movie_bias']

        print("📂 Loading content embeddings...")
        content_data    = joblib.load(os.path.join(MODELS_DIR, 'content_embeddings.pkl'))
        self.embeddings = content_data['embeddings']
        self.movie_ids  = content_data['movie_ids']
        self.titles     = content_data['titles']
        self.genres     = content_data['genres']
        self.id2idx     = {mid: i for i, mid in enumerate(self.movie_ids)}
        print("✅ Hybrid Recommender ready!")

    def _cf_score(self, user_id, movie_id):
        u  = self.user2idx.get(user_id)
        m  = self.movie2idx.get(movie_id)
        ub = self.user_bias.get(user_id, 0)
        mb = self.movie_bias.get(movie_id, 0)
        if u is None or m is None:
            return self.global_mean
        score = self.global_mean + ub + mb + self.reconstructed[u, m]
        return float(np.clip(score, 1, 5))

    def _content_score(self, liked_movie_ids, candidate_movie_id):
        cand_idx = self.id2idx.get(candidate_movie_id)
        if cand_idx is None or not liked_movie_ids:
            return 0.0
        cand_vec = self.embeddings[cand_idx].reshape(1, -1)
        scores = []
        for mid in liked_movie_ids:
            idx = self.id2idx.get(mid)
            if idx is not None:
                sim = cosine_similarity(
                    self.embeddings[idx].reshape(1, -1), cand_vec
                )[0][0]
                scores.append(sim)
        return float(np.mean(scores)) if scores else 0.0

    def recommend(self, user_id, liked_movie_ids=None, top_n=10, exclude_seen=True):
        u_idx = self.user2idx.get(user_id)
        seen_movies = set()
        if u_idx is not None and exclude_seen:
            seen_movies = {
                self.idx2movie[i]
                for i, v in enumerate(self.reconstructed[u_idx])
                if v > 0
            }

        results = []
        for movie_id in self.movie_ids:
            if movie_id in seen_movies:
                continue

            cf      = self._cf_score(user_id, movie_id)
            content = self._content_score(liked_movie_ids or [], movie_id)
            cf_norm = (cf - 1) / 4  # normalize 1-5 → 0-1

            if liked_movie_ids:
                final = self.cf_weight * cf_norm + self.content_weight * content
            else:
                final = cf_norm

            idx = self.id2idx.get(movie_id)
            results.append({
                'movie_id':      movie_id,
                'title':         self.titles[idx] if idx is not None else 'Unknown',
                'genres':        self.genres[idx] if idx is not None else '',
                'cf_score':      round(cf, 3),
                'content_score': round(content, 3),
                'final_score':   round(final, 3)
            })

        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results[:top_n]


if __name__ == '__main__':
    hybrid = HybridRecommender()
    print("\n🎬 توصيات للـ User 1 (بيحب Toy Story):")
    recs = hybrid.recommend(user_id=1, liked_movie_ids=[1], top_n=5)
    for r in recs:
        print(f"  {r['title']:40s} | CF: {r['cf_score']:.2f} | content: {r['content_score']:.3f} | final: {r['final_score']:.3f}")