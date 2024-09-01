import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the dataset
df = pd.read_csv('games.csv')

# Ensure no missing or invalid entries
df['About the game'] = df['About the game'].fillna('')  # Replace NaNs with empty strings
df['About the game'] = df['About the game'].astype(str)  # Convert all entries to strings

# Initialize the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def get_embeddings(text):
    if not isinstance(text, str) or not text.strip():
        return np.zeros((768,))  # Return a zero vector or handle the case as needed
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


# Apply embeddings to each game
df['embeddings'] = df['About the game'].apply(get_embeddings)

def get_similar_games(game_name, top_n=5):
    game_idx = df[df['Name'] == game_name].index[0]
    game_embedding = df.loc[game_idx, 'embeddings'].reshape(1, -1)
    similarities = cosine_similarity(game_embedding, list(df['embeddings']))
    similar_indices = similarities[0].argsort()[-top_n:][::-1]
    return df.iloc[similar_indices]

# Streamlit app
st.title('Game Recommender App')

# Dropdown for game selection
game_names = df['Name'].tolist()
selected_game = st.selectbox('Select a Game', game_names)

if selected_game:
    similar_games = get_similar_games(selected_game)
    st.write(f"### Recommended Games similar to `{selected_game}`")
    st.dataframe(similar_games[['Name', 'Short description of the game']])
