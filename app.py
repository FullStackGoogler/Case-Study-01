import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# Load the dataset
df = pd.read_csv('path_to_steam_dataset.csv')

# Initialize the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def get_embeddings(text):
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

def recommend_games(selected_game):
    similar_games = get_similar_games(selected_game)
    return similar_games[['Name', 'Short description of the game']]

# List of game names for the dropdown
game_names = df['Name'].tolist()

# Create Gradio interface
interface = gr.Interface(
    fn=recommend_games,
    inputs=gr.Dropdown(game_names, label="Select a Game"),
    outputs=gr.Dataframe(headers=['Game Name', 'Description'])
)

interface.launch()
