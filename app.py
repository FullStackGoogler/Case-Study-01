import pandas as pd
import numpy as np
from datetime import datetime

import re
import streamlit as st
import streamlit.components.v1 as components

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util

from huggingface_hub import InferenceClient

from itertools import product

from prometheus_client import start_http_server, Counter, Summary

# Prometheus metrics
REQUEST_COUNTER = Counter('app_requests_total', 'Total number of requests')
SUCCESSFUL_REQUESTS = Counter('app_successful_requests_total', 'Total number of successful requests')
FAILED_REQUESTS = Counter('app_failed_requests_total', 'Total number of failed requests')
REQUEST_DURATION = Summary('app_request_duration_seconds', 'Time spent processing request')

# Preprocess data in the CSV file
@st.cache_data
def data_processing():
    games = pd.read_csv('games.csv')

    # Remove any games with the "Sexual Content" tag
    initial_count = len(games)
    games = games[~games['Tags'].str.contains('Sexual Content', na=False)]
    games = games[~games['Name'].str.contains('Sex', case=False, na=False)]
    removed_count = initial_count - len(games)

    # Debugging: Display the count of removed entries
    print(f"Number of games removed due to 'Sexual Content': {removed_count}")
    
    # Rename columns to match the original code
    games.rename(columns={
        'Name': 'Title',
        'Release date': 'Release Date',
        'Publishers': 'Team',
        'Developers': 'Team',
        'About the game': 'Summary'
    }, inplace=True)
    
    # Remove duplicates
    games = games.loc[:, ~games.columns.duplicated()] # Cols
    games.drop_duplicates(inplace=True) # Rows
    
    # Handle apostrophes in titles
    games['fixed_title'] = games['Title'].astype(str).str.replace(r"'", "", regex=True)
    
    # Convert release dates
    games['Release Date'] = pd.to_datetime(games['Release Date'], errors='coerce')
    
    # Extract year from the 'Release Date'
    games['year'] = games['Release Date'].dt.year
    
    # Drop rows where year is missing or Release Date is 'releases on TBD'
    games = games.dropna(subset=['year'])
    games = games[games['Release Date'].notnull()]
    
    return games

# Helper function for displaying the most similar resutls using Streamlit's functions
def display_results(top5):
    for index, row in top5.iterrows():
        st.markdown(f"### **{row['Title']}** ({row['similarity']:.2f})")
        st.markdown(f"**Developer and Publisher:** {row['Team']}")
        st.markdown(f"{row['Summary']}")
        
        # Display the game's image
        st.image(row['Header image'])
        
        # Add a horizontal line for separation between results
        st.markdown("---")

# Hypertuning helper function to test similarity weight combinations
def find_best_weights(title_similiarities, summary_similarities, terms_similarities, team_similarities, games_filtered): # TODO: Seemingly always converges to a value of 1.0 for tags; look into tweaking the starting weights?
    # Define possible values for weights
    weights_range = np.arange(0, 1.1, 0.1)  # [0, 0.1, 0.2, ..., 1.0]
    
    best_avg_similarity = 0
    best_max_similarity = 0
    best_avg_weights = None
    best_max_weights = None
    
    # Generate all combinations of weights that sum to 1
    for title_weight, summary_weight, terms_weight, team_weight in product(weights_range, repeat=4):
        if np.isclose(title_weight + summary_weight + terms_weight + team_weight, 1.0):
            # Compute final similarity score for the current weight combination
            final_similarity = (title_weight * np.array(title_similiarities) +
                                summary_weight * np.array(summary_similarities) +
                                terms_weight * np.array(terms_similarities) +
                                team_weight * np.array(team_similarities))
            
            # Calculate average and max similarity score
            avg_similarity = np.mean(final_similarity)
            max_similarity = np.max(final_similarity)
            
            if avg_similarity > best_avg_similarity:
                best_avg_similarity = avg_similarity
                best_avg_weights = (title_weight, summary_weight, terms_weight, team_weight)

            if max_similarity > best_max_similarity:
                best_max_similarity = max_similarity
                best_max_weights = (title_weight, summary_weight, terms_weight, team_weight)
    
    return best_avg_weights, best_max_weights, best_avg_similarity, best_max_similarity

# Function to calculate the most similar games for a given game
#@st.cache_resource
def calculate_similarities(name, data_original, data_filtered, use_local_model):
    # Drop the selected game
    data_original.drop(data_original.query('Title == @name').index, inplace=True)

    # Filter the remaining games to include "good" games, as well as eliminate any older games

    # Keep games that have a rating of at least 85%
    data_original['positive_rating_percentage'] = data_original['Positive'] / (data_original['Positive'] + data_original['Negative']) * 100
    data_original = data_original[data_original['positive_rating_percentage'] > 85]
    remaining_count = len(data_original)

    # Keep games with more than 500 recommendations
    data_original = data_original[data_original['Recommendations'] > 500]
    rec_count = len(data_original)

    # print(f"Number of games with a rating of at least 85%: {remaining_count}")
    # print(f"Number of games with at least 500 recommendations: {rec_count}")
    
    # Filter games from the last X years
    years_before = datetime.now().year - 3
    games_filtered = data_original.query(f'year > {years_before}')

    result = data_filtered

    # Get game title
    selected_game_title = str(result.Title.item())

    # Combine terms from Categories, Genres, and Tags columns
    selected_game_terms = ' '.join(sorted([ # TODO: Consider running similarity scores instead for each of these columns separately?
        str(result.Categories.item()),
        str(result.Genres.item()),
        str(result.Tags.item())
    ]))

    # Get Publisher/Developers
    selected_game_team = ' '.join(sorted([
        str(result['Team'].item())
    ]))

    # Take the summary of the game and transform to string
    summary_selected_game = re.sub(r'\s{2,}', '', str(result.Summary.item()).replace('\n', ' '), flags=re.MULTILINE)
    
    # Transform all summaries, tags, categories, and publishers/developers into lists of strings
    summaries_all_games = games_filtered['Summary'].fillna('').str.replace(r'[\n\s]{2,}', ' ', regex=True).values.tolist()

    all_game_titles = games_filtered['Title'].apply(lambda x: ' '.join(sorted(str(x)))).tolist()
    
    all_game_terms = games_filtered.apply(
        lambda row: ' '.join(sorted([
            str(row['Categories']),
            str(row['Genres']),
            str(row['Tags'])
        ])),
        axis=1
    ).tolist()
    
    all_game_teams = games_filtered['Team'].apply(lambda x: ' '.join(sorted(str(x)))).tolist()

    # Ensure all elements are strings
    all_game_titles = [str(summary) for summary in all_game_titles]
    summaries_all_games = [str(summary) for summary in summaries_all_games]
    all_game_terms = [str(terms) for terms in all_game_terms]
    all_game_teams = [str(team) for team in all_game_teams]

    if use_local_model: # Locally run product
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Compute embeddings for each part
        embedding_summary = model.encode(summary_selected_game, convert_to_tensor=True)
        embedding_terms = model.encode(selected_game_terms, convert_to_tensor=True)
        embedding_team = model.encode(selected_game_team, convert_to_tensor=True)
    
        # Compute embeddings for all games
        embeddings_summaries = model.encode(summaries_all_games, convert_to_tensor=True)
        embeddings_terms = model.encode(all_game_terms, convert_to_tensor=True)
        embeddings_teams = model.encode(all_game_teams, convert_to_tensor=True)
    
        # Compute similarity
        similarity_summaries = util.pytorch_cos_sim(embedding_summary, embeddings_summaries)
        similarity_terms = util.pytorch_cos_sim(embedding_terms, embeddings_terms)
        similarity_teams = util.pytorch_cos_sim(embedding_team, embeddings_teams)
    
        # Combine similarity scores
        final_similarity = (0.4 * similarity_summaries + 0.4 * similarity_terms + 0.2 * similarity_teams) # TODO: Tinker around with these weights more?
        
        # Add final similarity scores back to the DataFrame
        games_filtered['similarity'] = final_similarity[0].tolist()
        
        top5 = games_filtered.sort_values(by='similarity', ascending=False)[:5]
        
        st.write(f'\n These are the 5 most similar games to {name}:')
        
        display_results(top5)
    else: # API based product
        client = InferenceClient()

        # Get similarities for titles
        title_similarities = client.sentence_similarity(
            selected_game_title,
            other_sentences=all_game_titles,
            model='sentence-transformers/all-MiniLM-L6-v2'
        )
        
        # Get similarities for summaries
        summary_similarities = client.sentence_similarity(
            summary_selected_game,
            other_sentences=summaries_all_games,
            model='sentence-transformers/all-MiniLM-L6-v2'
        )
        
        # Get similarities for terms
        terms_similarities = client.sentence_similarity(
            selected_game_terms,
            other_sentences=all_game_terms,
            model='sentence-transformers/all-MiniLM-L6-v2'
        )
        
        # Get similarities for teams
        team_similarities = client.sentence_similarity(
            selected_game_team,
            other_sentences=all_game_teams,
            model='sentence-transformers/all-MiniLM-L6-v2'
        )
        
        # Find the best weights
        best_avg_weights, best_max_weights, best_avg_similarity, best_max_similarity = find_best_weights(
            title_similarities, summary_similarities, terms_similarities, team_similarities, games_filtered
        )
        
        # st.write(f'Best weight combination for average similarity: {best_avg_weights}, Average Similarity: {best_avg_similarity}')
        # st.write(f'Best weight combination for max similarity: {best_max_weights}, Max Similarity: {best_max_similarity}')

        title_weight, summary_weight, terms_weight, team_weight = best_avg_weights
        
        final_similarity = (0 * np.array(title_similarities) +
                            0 * np.array(summary_similarities) +
                            1 * np.array(terms_similarities) +
                            0 * np.array(team_similarities))

        games_filtered['similarity'] = final_similarity.tolist()
        
        top5 = games_filtered.sort_values(by='similarity', ascending=False)[:5]
        
        st.write(f'\n These are the 5 most similar games to {name}:')
        
        display_results(top5)

# Application
st.set_page_config(layout="wide")
st.title("GG Go Next!")

# Sidebar
with st.sidebar:
    st.header("🎮")
    st.write("Select a game from the dropdown menu, and the app will calculate five recently released games most similar to the selected game! *Note that it does a few seconds to crunch the results.*")
    st.sidebar.info("This application was originally a project from Elisa Ribeiro, whose repo can be found [here](https://github.com/ElisaRMA). I have since improved the algorithm used to determine the results, and updated the dataset to better capture all the games on Steam.",icon="ℹ️")

    # Toggle for whether to use the API, or run it locally
    use_local_model = st.checkbox("Use Local Model", value=False) # TODO: Make sure that selecting this box during calculations doesn't mess anything up

# keeping track of session_state so buttons can be used inside multiple conditionals
def set_stage(stage):
    st.session_state.stage = stage
    
if 'stage' not in st.session_state:
    st.session_state.stage = 0

games = data_processing()

# Selection Box
option = st.selectbox(
    'Select a Game',
    games.Title.sort_values().unique(), placeholder="Choose your game", index=None)

find_me = st.button("Find a similar game!",on_click=set_stage, args=(1,)) # Set stage to 1

REQUEST_COUNTER.inc()  # Increment request counter
request_timer = REQUEST_DURATION.time()  # Start timing the request

# if session state is larger then one, meaning, the user clicked on the "Find a similar game!" button
# at this point, session_state.stage is equal to one
if st.session_state.stage > 0:
    # when a user clicks on the 'x' on the selectbox to clean selection, an attribute error was thrown because you can't replace a characters on a None object. 
    # So, we use try except to catch that error and instead of showing it, just display a pretty message for the user to reselect a game
    try:
        # takes the aposthrophe out 
        option = option.replace(r"'", "")
    except AttributeError:
        st.write('Reselect a game, please')
        set_stage(0)

    filtered = games.loc[games['fixed_title'] == option]
    
    try:
        # some testing on the filtered data so the user can confirm the game # TODO: Likely not needed
        if len(filtered) == 1:
            st.write('Is this the game you selected?')
            st.write(filtered.Title.item() + ' - ' + filtered.Team.item())
        
            header_image_url = filtered['Header image'].item()
            if header_image_url:  # Ensure there's an image to display
                st.image(header_image_url, caption=filtered.Title.item())

            # buttons of yes/no to confirm are created side by side using columns
            col1, col2 = st.columns(2)
            with col1:
                button1 = st.button('Yes', use_container_width=True, on_click=set_stage, args=(2,))

            with col2:
                button2 = st.button('No',use_container_width=True, on_click=set_stage, args=(3,))

            # after clicking on yes/no the the session_state.stage will be larger than 1 (2 for button1 and 3 for button3.
        
            if st.session_state.stage > 1 and button1:
                st.write(f'\n Calculating similarities between {option} and other games released in the past 3 years...\n\n\n')
                calculate_similarities(option, games, filtered, use_local_model)


            elif st.session_state.stage > 1 and button2:
                st.write('\n Please, check the name and try again')
                
                # reset everything to initial step
                set_stage(0)

        # TODO: Is this case needed if I just properly remove duplicates?
        # if there is two games with the same name in the dataset (e.g. cases like remakes), user will be asked to select which game (which year/team)
        elif len(filtered) > 1:
            indexes = filtered.index.tolist()
            st.write('\n We found more than one match: \n')
            st.write(f'Please, select the index\n')
            st.write(filtered[['Title', 'Release Date', 'Team']])
            user_selection = st.selectbox(label='Select the index', options=indexes,index=None)
            
            # If they click submit session stage will be set to 4
            submit = st.button("Submit",on_click=set_stage, args=(4,))
            
            # If it is larger than 3, then calculate the similarity with the new selected game
            if st.session_state.stage > 3:
                filtered_fixed = filtered.loc[[user_selection]]

                st.dataframe(filtered_fixed[['Title', 'Release Date', 'Team']])
            
                st.write(f'\n Now, we will calculate similarities between {option} and other games from the last 3 years. Please wait...\n\n\n')
                calculate_similarities(option, games, filtered_fixed, use_local_model)

        SUCCESSFUL_REQUESTS.inc()  # Increment successful request counter
    except Exception as e:
        FAILED_REQUESTS.inc()  # Increment failed request counter
        yield history + [(message, f"Error: {str(e)}")]
    finally:
        request_timer.observe_duration()  # Stop timing the request
