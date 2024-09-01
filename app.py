import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('games.csv')
    return df

# Preprocess the dataset to combine relevant text columns
@st.cache_data
def preprocess_data(df):
    df['combined_text'] = df['Name'] + ' ' + df['About the game'] + ' ' + df['Genres'] + ' ' + df['Tags'] + ' ' + df['Developers'] + ' ' + df['Publishers']
    return df

# Generate embeddings using the all-MiniLM-L6-v2 model
@st.cache_resource
def get_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@st.cache_data
def generate_embeddings(df, _model):
    # Ensure all values in 'combined_text' are strings and replace NaN with an empty string
    df['combined_text'] = df['combined_text'].fillna('').astype(str)
    df['embeddings'] = df['combined_text'].apply(lambda x: _model.encode(x, convert_to_tensor=True))
    return df

# Recommender function to find the most similar games
def recommend_games(input_game_name, df, top_n=10):
    # Find the input game's embedding
    input_game_embedding = df[df['Name'] == input_game_name]['embeddings'].values[0]

    # Compute cosine similarity with all other games
    similarities = df['embeddings'].apply(lambda x: cosine_similarity(x.unsqueeze(0), input_game_embedding.unsqueeze(0)).item())

    # Sort by similarity and return the top N results
    df['similarity'] = similarities
    recommendations = df.sort_values(by='similarity', ascending=False).head(top_n + 1)  # +1 to exclude the input game itself
    return recommendations[['Name', 'Genres', 'Tags', 'similarity']]

# Streamlit app
def main():
    st.title("Game Recommender System")

    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)

    # Load model and generate embeddings
    model = get_model()
    df = generate_embeddings(df, model)

    # User input for the game name
    game_name = st.selectbox("Select a game", df['Name'].unique())

    # Show recommendations when the button is clicked
    if st.button("Find Similar Games"):
        recommendations = recommend_games(game_name, df)
        st.write("Top 10 similar games:")
        for index, row in recommendations.iterrows():
            st.write(f"{index+1}. **{row['Name']}** - Genres: {row['Genres']} - Similarity: {row['similarity']:.4f}")

if __name__ == "__main__":
    main()
