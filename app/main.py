import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.hybrid.neural_hybrid_model import HybridRecommender
import pandas as pd

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

@st.cache_resource
def load_model():
    return HybridRecommender()

@st.cache_data
def load_movies():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return pd.read_csv(os.path.join(BASE_DIR, 'data', 'movielens', 'movies.csv'))

# ---- Header ----
st.title("🎬 Hybrid Movie Recommender")
st.caption("SVD (Collaborative) + BERT (Content-Based) + Neural Hybrid")

# ---- Load ----
with st.spinner("Loading models..."):
    hybrid = load_model()
    movies = load_movies()

# ---- Sidebar ----
st.sidebar.header("⚙️ Settings")
user_id = st.sidebar.number_input("User ID", min_value=1, max_value=6040, value=1)
top_n   = st.sidebar.slider("Number of recommendations", 5, 20, 10)
cf_w    = st.sidebar.slider("CF Weight",      0.0, 1.0, 0.6, 0.1)
con_w   = st.sidebar.slider("Content Weight", 0.0, 1.0, 0.4, 0.1)

hybrid.cf_weight      = cf_w
hybrid.content_weight = con_w

# ---- اختيار أفلام محبوبة ----
st.subheader("❤️ اختار أفلام بتحبها (اختياري)")
all_titles  = movies['title'].tolist()
liked_titles = st.multiselect("ابحث واختار:", all_titles, max_selections=5)
liked_ids   = movies[movies['title'].isin(liked_titles)]['movieId'].tolist()

# ---- توصيات ----
if st.button("🚀 Get Recommendations", type="primary"):
    with st.spinner("Generating recommendations..."):
        recs = hybrid.recommend(
            user_id=int(user_id),
            liked_movie_ids=liked_ids if liked_ids else None,
            top_n=top_n
        )

    st.subheader(f"🎯 Top {top_n} Recommendations for User {user_id}")

    cols = st.columns(3)
    for i, r in enumerate(recs):
        with cols[i % 3]:
            st.markdown(f"""
            <div style='background:#1e1e2e;padding:15px;border-radius:10px;margin:5px;border-left:4px solid #7c3aed'>
                <h4 style='color:#e2e8f0;margin:0'>{r['title']}</h4>
                <p style='color:#94a3b8;font-size:12px'>{r['genres']}</p>
                <p style='margin:5px 0'>
                    ⭐ CF: <b style='color:#fbbf24'>{r['cf_score']}</b> &nbsp;
                    📝 Content: <b style='color:#34d399'>{r['content_score']}</b>
                </p>
                <p>🏆 Score: <b style='color:#a78bfa'>{r['final_score']}</b></p>
            </div>
            """, unsafe_allow_html=True)

    # ---- جدول تفصيلي ----
    st.subheader("📊 Detailed Results")
    df = pd.DataFrame(recs)
    st.dataframe(df[['title','genres','cf_score','content_score','final_score']],
                 use_container_width=True)