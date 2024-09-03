import streamlit as st
import pandas as pd
import re
from datetime import datetime
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util

@st.cache_data
def data_processing():
    # CSV File
    games = pd.read_csv('games.csv')

    # Filter out games with "Sexual Content" in the Tags column
    initial_count = len(games)
    games = games[~games['Tags'].str.contains('Sexual Content', na=False)]
    removed_count = initial_count - len(games)

    # Filter for games with a positive rating percentage > 90%
    games['positive_rating_percentage'] = games['Positive'] / (games['Positive'] + games['Negative']) * 100
    games = games[games['positive_rating_percentage'] > 85]
    remaining_count = len(games)

    # Games with more than 500 recommendations
    games = games[games['Recommendations'] > 500]
    after_recommendations_filter = len(games)

    # Debugging: Display the count of removed entries
    st.write(f"Number of games removed due to 'Sexual Content': {removed_count}")
    st.write(f"Number of games with 90%+ review ratings: {remaining_count}")
    st.write(f"Number of games with more recommend: {after_recommendations_filter}")
    
    # Rename columns to match the original code
    games.rename(columns={
        'Name': 'Title',
        'Release date': 'Release Date',
        'Publishers': 'Team',
        'Developers': 'Team',
        #'Genres': 'Genres',
        #'Tags': 'Genres',
        'About the game': 'Summary'
    }, inplace=True)
    
    # Drop duplicate columns
    games = games.loc[:, ~games.columns.duplicated()]
    
    # Drop any duplicate rows
    games.drop_duplicates(inplace=True)
    
    # Handle apostrophes in titles
    games['fixed_title'] = games['Title'].astype(str).str.replace(r"'", "", regex=True)
    
    # Print column types and sample values for debugging
    print("Columns and their types:")
    print(games.dtypes)
    print("\nSample values:")
    print(games.head())
    
    # Convert release dates and handle null values
    games['Release Date'] = pd.to_datetime(games['Release Date'], errors='coerce')
    
    # Extract year from the 'Release Date'
    games['year'] = games['Release Date'].dt.year
    
    # Drop rows where year is missing or Release Date is 'releases on TBD'
    games = games.dropna(subset=['year'])
    games = games[games['Release Date'].notnull()]
    
    # Keep only games released in the last 4 years
    five_years_before = datetime.now().year - 4
    games = games[games['year'] > five_years_before]
    
    return games

#@st.cache_resource
def calculate_similarities(name, data_original, data_filtered):
    # Drop the games that were selected
    data_original.drop(data_original.query('Title == @name').index, inplace=True)
    
    # Filter games from the last 3 years
    five_years_before = datetime.now().year - 7
    games_filtered = data_original.query(f'year > {five_years_before}')

    result = data_filtered

    # Step 1: Gather terms from Categories, Genres, Tags, and sort them
    selected_game_terms = ' '.join(sorted([
        str(result.Categories.item()),
        str(result.Genres.item()),
        str(result.Tags.item())
    ]))

    # Step 2: Gather terms from Publisher/Developer
    selected_game_team = ' '.join(sorted([
        str(result['Team'].item())
    ]))

    # Step 3: Transform the summary of the selected game
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

    # Combine similarity scores (optionally with weights)
    final_similarity = (0.5 * similarity_summaries + 0.3 * similarity_terms + 0.2 * similarity_teams)
    
    # Add final similarity scores back to the DataFrame
    games_filtered['similarity'] = final_similarity[0].tolist()
    
    # Order the DataFrame based on the similarity scores
    top5 = games_filtered.sort_values(by='similarity', ascending=False)[:5]
    
    st.write(f'\n These are the 5 most similar games to {name}:')
    st.dataframe(top5[['Title', 'similarity']])





############ App ############

st.set_page_config(layout="wide")

# Title! 
st.title("🎮 Welcome to the Game Recommender!")

# Sidebar
with st.sidebar:
    st.header("How to use it")
    st.write("Select a game from the dropdown menu and wait! \n\n You'll receive the top 5 most similar games released in the last 3 years. \n\n Enjoy!")
    st.sidebar.info("Want to know more about this recommender and how I built it? Access the [GitHub repo](https://github.com/ElisaRMA)",icon="ℹ️")



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

    print("\nDebugging Information:")
    print(f"Type of fixed_title column: {games['fixed_title'].dtype}")
    print(f"Type of option variable: {type(option)}")
    print(f"Value of option variable: {option}")
    print(f"Sample values from fixed_title column:")
    print(games['fixed_title'].head())

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
            st.write(f'\n Now, we will calculate similarities between {option} and other games from the last 3 years. Please wait...\n\n\n')
            calculate_similarities(option, games, filtered)


        elif st.session_state.stage > 1 and button2:
            st.write('\n Please, check the name and try again')
            
            # reset everything to initial step
            set_stage(0)


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