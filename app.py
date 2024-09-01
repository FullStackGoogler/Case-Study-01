import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import numpy as np
import pickle
import os

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
    df['About the game'] = df['About the game'].fillna('')
    df['About the game'] = df['About the game'].astype(str)
    st.write("Dataset preprocessing complete.")

# Initialize the model with debugging
st.write("Initializing model...")
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    st.write("Model loaded successfully.")
except Exception as e:
    st.write(f"Error loading model: {e}")

def get_embeddings(texts):
    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        return np.zeros((384,))  # Return a zero vector if input is invalid
    
    try:
        embeddings = model.encode(texts, show_progress_bar=True)
        return embeddings  # Returns a numpy array of shape (batch_size, 384)
    except Exception as e:
        st.write(f"Error in get_embeddings function: {e}")
        return np.zeros((384,))  # Return a zero vector if an error occurs

def compute_and_save_embeddings():
    st.write("Computing embeddings...")
    try:
        # Using a larger batch size for efficiency
        batch_size = 128
        embeddings_list = []
        num_batches = len(df) // batch_size + int(len(df) % batch_size > 0)
        
        for i in range(num_batches):
            batch_texts = df['About the game'].iloc[i*batch_size:(i+1)*batch_size].tolist()
            embeddings = get_embeddings(batch_texts)
            embeddings_list.extend(embeddings)

        df['embeddings'] = embeddings_list

        with open('embeddings.pkl', 'wb') as f:
            pickle.dump(df[['Name', 'embeddings']], f)

        st.write("Embeddings computed and saved successfully.")
    except Exception as e:
        st.write(f"Error computing and saving embeddings: {e}")

def load_embeddings():
    try:
        with open('embeddings.pkl', 'rb') as f:
            saved_data = pickle.load(f)
        return saved_data
    except Exception as e:
        st.write(f"Error loading embeddings: {e}")
        return None

# Check and load/save embeddings
if os.path.exists('embeddings.pkl'):
    df = load_embeddings()
else:
    compute_and_save_embeddings()

def get_similar_games(game_name, top_n=5):
    try:
        selected_game_description = df[df['Name'] == game_name]['About the game'].values[0]
        game_embedding = get_embeddings([selected_game_description]).reshape(1, -1)

        all_game_descriptions = df['About the game'].tolist()
        all_embeddings = get_embeddings(all_game_descriptions)

        similarities = cosine_similarity(game_embedding, all_embeddings)
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
