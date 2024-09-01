import streamlit as st
import pandas as pd
import re
from datetime import datetime
#from FlagEmbedding import BGEM3FlagModel
# Use a pipeline as a high-level helper
#from transformers import pipeline
#from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util


############ Functions ############

# st.cache_resoure is for: 

@st.cache_data
def data_processing():

    # loading the data and preparing it
    games = pd.read_csv('./games.csv') #.drop('Unnamed: 0', axis=1)
    
    # some duplicate rows
    games.drop_duplicates(inplace=True)

    # deal with apostrophes
    games['fixed_title'] = games.Title.str.replace(r"'", "")

    # dropping weird dates and null values
    games[['date','year']] = games['Release Date'].str.split(',',expand=True)
    games.year.dropna(inplace=True)
    games = games[games["Release Date"] != "releases on TBD"]
    
    games['year'] = games['year'].astype(int)

    return games


#@st.cache_resource
def calculate_similarities(name, data_original, data_filtered):

    # subsetting 5 years - this will be the one searched - could add functionality
    five_years_before = datetime.now().year - 3

    # drop the games that were selected
    data_original.drop(data_original.query('Title == @name').index)

    games_filtered = data_original.query(f'year > {five_years_before}')

    result = data_filtered

    # Take the summary of the game and transform to string
    summary_selected_game = re.sub(r'\s{2,}', '', result.Summary.item().replace('\n', ' '), flags=re.MULTILINE)

    # transform all the column Summary into strings, as a list, obeying the order of the table
    summaries_all_games = games_filtered.Summary.str.replace(r'[\n\s]{2,}', ' ', regex=True).values.tolist()

    # run the model


    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    #Compute embedding for both lists
    embedding_1= model.encode(summary_selected_game, convert_to_tensor=True)
    embedding_2 = model.encode(summaries_all_games, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(embedding_1, embedding_2)

    
    # place it back in the dataset
    games_filtered.loc[:,'similarity'] = similarity[0].tolist()

    games_filtered['summary_fixed'] = games_filtered.Summary.str.replace(r'[\n\s]{2,}', ' ', regex=True)

    # order the dataset based on the score
    # output the game.Title based on the input from user
    top5 = games_filtered.sort_values(by='similarity', ascending=False)[:5]

    st.write(f'\n These are the 5 most similar games to {name}:')
    st.dataframe(top5[['Title','summary_fixed']])


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


    filtered = games.query('fixed_title == @option')

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