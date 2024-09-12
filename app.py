import pandas as pd
from datetime import datetime

import re
import streamlit as st
import streamlit.components.v1 as components

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util

# Helper Functions

@st.cache_data
def data_processing():
    games = pd.read_csv('games.csv')

    # Remove any games with the "Sexual Content" tag
    initial_count = len(games)
    games = games[~games['Tags'].str.contains('Sexual Content', na=False)]
    games = games[~games['Name'].str.contains('Sex', case=False, na=False)]
    removed_count = initial_count - len(games)

    # Debugging: Display the count of removed entries
    #print(f"Number of games removed due to 'Sexual Content': {removed_count}")
    #print(f"Number of games with a rating of at least 85%: {remaining_count}")
    #print(f"Number of games with at least 500 recommendations: {rec_count}")
    
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
    
    # Debugging types & sample values
    print("Columns and their types:")
    print(games.dtypes)
    print("\nSample values:")
    print(games.head())
    
    # Convert release dates
    games['Release Date'] = pd.to_datetime(games['Release Date'], errors='coerce')
    
    # Extract year from the 'Release Date'
    games['year'] = games['Release Date'].dt.year
    
    # Drop rows where year is missing or Release Date is 'releases on TBD'
    games = games.dropna(subset=['year'])
    games = games[games['Release Date'].notnull()]
    
    # Keep only games released in the last 5 years
    five_years_before = datetime.now().year - 4
    games = games[games['year'] > five_years_before]
    recent_count = len(games)

    print(f"Number of games released within the last 4 years: {recent_count}")
    
    return games

def display_results(top5):
    for index, row in top5.iterrows():
        st.markdown(f"### **{row['Title']}** ({row['similarity']:.2f})")
        st.markdown(f"**Developer and Publisher:** {row['Team']}")
        st.markdown(f"{row['Summary']}")
        
        # Display the game's image
        st.image(row['Header image'], caption=row['Title'])
        
        # Add a horizontal line for separation between results
        st.markdown("---")

#@st.cache_resource
def calculate_similarities(name, data_original, data_filtered):
    # Drop the game that were selected
    data_original.drop(data_original.query('Title == @name').index, inplace=True)

    # Keep games that have a rating of at least 85%
    data_original['positive_rating_percentage'] = data_original['Positive'] / (data_original['Positive'] + data_original['Negative']) * 100
    data_original = data_original[data_original['positive_rating_percentage'] > 85]
    #remaining_count = len(games)

    # Keep games with more than 500 recommendations
    data_original = data_original[data_original['Recommendations'] > 500]
    #rec_count = len(games)
    
    # Filter games from the last 5 years
    five_years_before = datetime.now().year - 2
    games_filtered = data_original.query(f'year > {five_years_before}')

    result = data_filtered

    # Combine terms from Categories, Genres, and Tags columns
    selected_game_terms = ' '.join(sorted([
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
    summaries_all_games = [str(summary) for summary in summaries_all_games]
    all_game_terms = [str(terms) for terms in all_game_terms]
    all_game_teams = [str(team) for team in all_game_teams]

    # Run the model
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
    final_similarity = (0.4 * similarity_summaries + 0.45 * similarity_terms + 0.15 * similarity_teams) # TODO: Tinker around with these weights more?
    
    # Add final similarity scores back to the DataFrame
    games_filtered['similarity'] = final_similarity[0].tolist()
    
    # Order the DataFrame based on the similarity scores
    top5 = games_filtered.sort_values(by='similarity', ascending=False)[:5]
    
    st.write(f'\n These are the 5 most similar games to {name}:')
    display_results(top5)

# Application

st.set_page_config(layout="wide")

# Title! 
st.title("GG Go Next!")

# Sidebar
with st.sidebar:
    st.header("🎮")
    st.write("Select a game from the dropdown menu, and the app will calculate the five games most similar to the selected game! *Note that it does a few minutes to crunch the results.*")
    st.sidebar.info("This application was originally a project from Elisa Ribeiro, whose repo can be found here: [GitHub repo](https://github.com/ElisaRMA). I have since added my own improvements to the algorithm, updated the dataset used, and tweaked the UI.",icon="ℹ️")

# keeping track of session_state so buttons can be used inside multiple conditionals
def set_stage(stage):
    st.session_state.stage = stage
    
if 'stage' not in st.session_state:
    st.session_state.stage = 0

# loads data (and caches, because the function is being cached)
games = data_processing()

# selection box for user to choose a game
option = st.selectbox(
    'Select a game',
    games.Title.sort_values().unique(), placeholder="Choose your game", index=None)


# button to 'start' the app functions
# after clicking it, sets stage to 1: is running set_stage function with argument 1
find_me = st.button("Find a similar game!",on_click=set_stage, args=(1,))

# if session state is larger then one, meaning, the user clicked on the "Find a similar game!" button
# at this point, session_state.stage is equal to one
if st.session_state.stage > 0:
    # when a user clicks on the 'x' on the selectbox to clean selection, an attribute error was thrown because you can't replace a characters on a None object. 
    # So, we use try except to catch that error and instead of showing it, just display a pretty message for the user to reselect a game
    try:
        # takes the aposthrofe out 
        option = option.replace(r"'", "")
    except AttributeError:
        st.write('Reselect a game, please')
        set_stage(0)

    #print("\nDebugging Information:")
    #print(f"Type of fixed_title column: {games['fixed_title'].dtype}")
    #print(f"Type of option variable: {type(option)}")
    #print(f"Value of option variable: {option}")
    #print(f"Sample values from fixed_title column:")
    #print(games['fixed_title'].head())

    #filtered = games.query('fixed_title == @option')
    filtered = games.loc[games['fixed_title'] == option]
    
    # some testing on the filtered data so the user can confirm the game
    if len(filtered) == 1:
        st.write('Is this the game you selected?')
        st.write(filtered.Title.item() + ' - ' + filtered.Team.item())

        # buttons of yes/no to confirm are created side by side using columns
        col1, col2 = st.columns(2)
        with col1:
            button1 = st.button('Yes', use_container_width=True, on_click=set_stage, args=(2,))

        with col2:
            button2 = st.button('No',use_container_width=True, on_click=set_stage, args=(3,))

        # after clicking on yes/no the the session_state.stage will be larger than 1 (2 for button1 and 3 for button3.
    
        if st.session_state.stage > 1 and button1:
            st.write(f'\n Now, we will calculate similarities between {option} and other games from the last 4 years. Please wait...\n\n\n')
            calculate_similarities(option, games, filtered)


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
            calculate_similarities(option, games, filtered_fixed)