import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Merge ratings with movie titles
data = pd.merge(ratings, movies, on="movieId")

# Create user-item matrix
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# Compute item similarity
movie_similarity = cosine_similarity(user_movie_matrix.T)
movie_similarity_df = pd.DataFrame(movie_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

# Function to get recommendations
def recommend_movies(movie_name, n=5):
    if movie_name not in movie_similarity_df.columns:
        return ["Movie not found in dataset!"]
    similar_movies = movie_similarity_df[movie_name].sort_values(ascending=False)
    recommended = similar_movies.drop(movie_name).head(n)
    return recommended.index.tolist()

# Streamlit interface
st.title("Movie Recommendation System")
st.write("Type a movie you like and get recommendations!")

movie_input = st.selectbox("Select a movie:", movies['title'].tolist())
num_recs = st.slider("Number of recommendations:", 1, 20, value=5, key=movie_input)
if st.button("Recommend"):
    recommendations = recommend_movies(movie_input, num_recs)
    st.write("Because you liked **{}**, you might also like:".format(movie_input))
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
    
