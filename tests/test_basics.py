import pytest
import pandas as pd
from app import data_processing, calculate_similarities  # Adjust import based on your script name

def test_data_processing():
    # Run data_processing function
    result = data_processing()
    
    # Check if the result is a DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Check if 'Title' column exists
    assert 'Title' in result.columns

    assert len(result) > 0

def test_calculate_similarities():
    # Sample data setup
    sample_data = pd.DataFrame({
        'Title': ['Game1', 'Game2'],
        'Summary': ['A great game', 'Another great game'],
        'Categories': ['Action', 'Adventure'],
        'Genres': ['RPG', 'Strategy'],
        'Tags': ['Fun', 'Challenging'],
        'Team': ['Dev1', 'Dev2'],
        'Release Date': ['2021-01-01', '2022-01-01'],
        'Positive': [100, 200],
        'Negative': [10, 20],
        'Recommendations': [600, 700]
    })
    
    # Simulate the selected game
    selected_game = sample_data.iloc[0]
    filtered_data = sample_data.copy()
    
    # Test with local model
    result = calculate_similarities('Game1', sample_data, selected_game, use_local_model=True)
    
    # Check if the result is a DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Check if similarity column is present
    assert 'similarity' in result.columns
    
    # Check if there are top 5 results
    assert len(result) > 0

def test_calculate_similarities_empty_data():
    # Empty DataFrame
    sample_data = pd.DataFrame()
    
    # Test with empty DataFrame
    result = calculate_similarities('Game1', sample_data, sample_data, use_local_model=True)
    
    # Check if result is an empty DataFrame
    assert result.empty

