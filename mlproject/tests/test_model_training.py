import pytest
from sklearn.datasets import make_classification
from model_experimentation import train_model

def test_train_model():
    """
    Test the train_model function to verify that it trains a model successfully.
    """
    X, y = make_classification(n_samples=100, n_features=5)
    params = {"n_estimators": 10, "random_state": 42}
    model = train_model(X, y, params)
    assert hasattr(model, "predict")  # Check if model is trained
