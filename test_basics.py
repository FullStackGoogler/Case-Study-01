import pytest
import pandas as pd
from app import data_processing, calculate_similarities

def test_data_processing():
    result = data_processing()

    print(result.columns)
    
    assert isinstance(result, pd.DataFrame)
    
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
    
    selected_game = sample_data.iloc[0]
    filtered_data = sample_data.copy()
    
    result = calculate_similarities('Game1', sample_data, selected_game, use_local_model=False)

    assert isinstance(result, pd.DataFrame)
    
    assert 'similarity' in result.columns

    assert len(result) > 0

def test_calculate_similarities_empty_data():
    # Empty DataFrame
    sample_data = pd.DataFrame()

    result = calculate_similarities('Game1', sample_data, sample_data, use_local_model=True)

    assert result.empty

