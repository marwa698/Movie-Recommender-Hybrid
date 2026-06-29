import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'movielens')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("📂 Loading data...")
ratings = pd.read_csv(os.path.join(BASE_DIR, 'ml-1m', 'ratings.dat'),
    sep='::', names=['userId','movieId','rating','timestamp'],
    engine='python', encoding='latin-1')
movies = pd.read_csv(os.path.join(BASE_DIR, 'ml-1m', 'movies.dat'),
    sep='::', names=['movieId','title','genres'],
    engine='python', encoding='latin-1')

print(f"✅ Ratings: {len(ratings):,}")

train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)

# ---- User & Movie biases ----
global_mean = train_df['rating'].mean()
user_bias  = train_df.groupby('userId')['rating'].mean() - global_mean
movie_bias = train_df.groupby('movieId')['rating'].mean() - global_mean

# ---- User-Item Matrix (mean-centered) ----
user_ids  = ratings['userId'].unique()
movie_ids = ratings['movieId'].unique()
user2idx  = {u: i for i, u in enumerate(user_ids)}
movie2idx = {m: i for i, m in enumerate(movie_ids)}
idx2movie = {i: m for m, i in movie2idx.items()}

print("🔨 Building matrix...")
matrix = np.zeros((len(user_ids), len(movie_ids)))
for row in train_df.itertuples():
    u = user2idx.get(row.userId)
    m = movie2idx.get(row.movieId)
    if u is not None and m is not None:
        # نطرح الـ bias عشان نعمل centering
        ub = user_bias.get(row.userId, 0)
        mb = movie_bias.get(row.movieId, 0)
        matrix[u, m] = row.rating - global_mean - ub - mb

print("✅ Matrix built!")

# ---- SVD ----
from sklearn.decomposition import TruncatedSVD
print("🚀 Training SVD...")
svd = TruncatedSVD(n_components=50, random_state=42)
U   = svd.fit_transform(matrix)
Vt  = svd.components_
reconstructed_centered = U @ Vt

# ---- تقييم ----
actual, predicted = [], []
for row in test_df.itertuples():
    u  = user2idx.get(row.userId)
    m  = movie2idx.get(row.movieId)
    ub = user_bias.get(row.userId, 0)
    mb = movie_bias.get(row.movieId, 0)
    if u is not None and m is not None:
        pred = global_mean + ub + mb + reconstructed_centered[u, m]
    else:
        pred = global_mean
    pred = np.clip(pred, 1, 5)
    predicted.append(pred)
    actual.append(row.rating)

rmse = np.sqrt(mean_squared_error(actual, predicted))
print(f"✅ SVD RMSE: {rmse:.4f}")

# ---- حفظ ----
model_data = {
    'U': U, 'Vt': Vt,
    'reconstructed_centered': reconstructed_centered,
    'user2idx': user2idx, 'movie2idx': movie2idx, 'idx2movie': idx2movie,
    'global_mean': global_mean,
    'user_bias': user_bias.to_dict(),
    'movie_bias': movie_bias.to_dict(),
}
joblib.dump(model_data, os.path.join(MODELS_DIR, 'svd_model.pkl'))
movies.to_csv(os.path.join(DATA_DIR, 'movies.csv'), index=False)
ratings.to_csv(os.path.join(DATA_DIR, 'ratings.csv'), index=False)
print("✅ Saved!")