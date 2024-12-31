import pytest
import pandas as pd
from src.data_processing import split_data

def test_split_data():
    """
    Test the split_data function for proper data splitting.
    """
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "feature2": [5, 6, 7, 8],
        "target": [0, 1, 0, 1],
    })
    X_train, X_test, y_train, y_test = split_data(df, 0.5, 42)
    assert len(X_train) == 2
    assert len(X_test) == 2
