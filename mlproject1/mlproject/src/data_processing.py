import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Data loaded as a pandas DataFrame.
    """
    return pd.read_csv(file_path)

def split_data(df: pd.DataFrame, test_size: float, random_state: int):
    """
    Split the dataset into training and testing sets.

    Args:
        df (pd.DataFrame): The input dataset containing features and target.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Tuple containing training features (X_train), testing features (X_test),
               training labels (y_train), and testing labels (y_test).
    """
    X = df.drop("target", axis=1)
    y = df["target"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
