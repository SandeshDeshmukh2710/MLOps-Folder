import joblib
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, params: dict) -> RandomForestClassifier:
    """
    Train a Random Forest model with the given parameters.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training labels.
        params (dict): Hyperparameters for the Random Forest model.

    Returns:
        RandomForestClassifier: Trained Random Forest model.
    """
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model

def save_model(model: RandomForestClassifier, file_path: str):
    """
    Save the trained model to a file.

    Args:
        model (RandomForestClassifier): Trained model to save.
        file_path (str): Path to save the model file.
    """
    joblib.dump(model, file_path)
