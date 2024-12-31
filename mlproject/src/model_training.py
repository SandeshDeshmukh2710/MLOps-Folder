import joblib
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

MODEL_MAPPING = {
    "LogisticRegression": LogisticRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "XGBClassifier": XGBClassifier,
}


def train_model(X_train, y_train,X_test,y_test, model,params: dict) -> XGBClassifier:
    """
    Train a Random Forest model with the given parameters.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training labels.
        params (dict): Hyperparameters for the Random Forest model.

    Returns:
        RandomForestClassifier: Trained Random Forest model.
    """
    model = MODEL_MAPPING[model](**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model

def save_model(model: XGBClassifier, file_path: str):
    """
    Save the trained model to a file.

    Args:
        model (RandomForestClassifier): Trained model to save.
        file_path (str): Path to save the model file.
    """
    joblib.dump(model, file_path)
