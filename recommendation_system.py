
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
        print("Movie not found in dataset!")
        return []
    similar_movies = movie_similarity_df[movie_name].sort_values(ascending=False)
    recommended = similar_movies.drop(movie_name).head(n)
return recommended.index.tolist()

# Test recommendations
favorite_movie = "Sabrina (1995)"
print(f"Because you liked {favorite_movie}, you might also like:")
print(recommend_movies(favorite_movie, 10))
