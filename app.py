import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import numpy as np

# Streamlit app title
st.title('Game Recommender App')

# Load the dataset with debugging
st.write("Loading dataset...")
try:
    df = pd.read_csv('games.csv')
    st.write("Dataset loaded successfully.")
    st.write(f"Dataframe shape: {df.shape}")
    st.write("First few rows of the dataset:")
    st.write(df.head())
except Exception as e:
    st.write(f"Error loading dataset: {e}")

# Ensure 'About the game' column exists and is correctly formatted
if 'About the game' not in df.columns:
    st.write("Error: 'About the game' column is missing.")
else:
    df['About the game'] = df['About the game'].fillna('')  # Replace NaNs with empty strings
    df['About the game'] = df['About the game'].astype(str)  # Convert all entries to strings
    st.write("Dataset preprocessing complete.")

# Initialize the tokenizer and model with debugging
st.write("Initializing tokenizer and model...")
try:
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    st.write("Tokenizer and model loaded successfully.")
except Exception as e:
    st.write(f"Error loading tokenizer/model: {e}")

def get_embeddings(text):
    if not isinstance(text, str) or not text.strip():
        return np.zeros((768,))  # Return a zero vector if text is invalid
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    except Exception as e:
        st.write(f"Error in get_embeddings function: {e}")
        return np.zeros((768,))  # Return a zero vector if an error occurs

# Apply embeddings to each game with debugging
st.write("Computing embeddings...")
try:
    df['embeddings'] = df['About the game'].apply(get_embeddings)
    st.write("Embeddings computed successfully.")
except Exception as e:
    st.write(f"Error computing embeddings: {e}")

def get_similar_games(game_name, top_n=5):
    try:
        game_idx = df[df['Name'] == game_name].index[0]
        game_embedding = df.loc[game_idx, 'embeddings'].reshape(1, -1)
        similarities = cosine_similarity(game_embedding, list(df['embeddings']))
        similar_indices = similarities[0].argsort()[-top_n:][::-1]
        return df.iloc[similar_indices]
    except Exception as e:
        st.write(f"Error in get_similar_games function: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if an error occurs

# Dropdown for game selection with debugging
st.write("Creating dropdown menu...")
game_names = df['Name'].tolist() if 'Name' in df.columns else []
selected_game = st.selectbox('Select a Game', game_names)

if selected_game:
    st.write(f"Selected Game: {selected_game}")
    try:
        similar_games = get_similar_games(selected_game)
        if not similar_games.empty:
            st.write(f"### Recommended Games similar to `{selected_game}`")
            st.dataframe(similar_games[['Name', 'Short description of the game']])
        else:
            st.write("No similar games found.")
    except Exception as e:
        st.write(f"Error retrieving similar games: {e}")
