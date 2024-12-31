from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test) -> float:
    """
    Evaluate the model's performance on the test set.

    Args:
        model (RandomForestClassifier): Trained model to evaluate.
        X_test (pd.DataFrame or np.ndarray): Test features.
        y_test (pd.Series or np.ndarray): True labels for the test set.

    Returns:
        float: Accuracy of the model on the test data.
    """
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)
