from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import os
import yaml

def evaluate_model(model, X_test, y_test, output_path) -> dict:
    """
    Evaluate the model's performance on the test set using multiple metrics.

    Args:
        model (sklearn model): Trained model to evaluate (e.g., XGBoost).
        X_test (pd.DataFrame or np.ndarray): Test features.
        y_test (pd.Series or np.ndarray): True labels for the test set.
        metrics (list, optional): List of metrics to compute. Default is None, which computes all metrics.

    Returns:
        dict: Dictionary of metrics and their corresponding values.
    """
    # Generate predictions
    predictions = model.predict(X_test)

    # Define default metrics if not provided
    
    metrics = ["accuracy", "precision", "recall", "f1", "confusion_matrix"]

    # Initialize a dictionary to store the results
    results = {}

    # Accuracy
    if "accuracy" in metrics:
        results["accuracy"] = float(accuracy_score(y_test, predictions))

    # Precision
    if "precision" in metrics:
        results["precision"] = float(precision_score(y_test, predictions))
                                     

    # Recall
    if "recall" in metrics:
        results["recall"] = float(recall_score(y_test, predictions))

    # F1-Score
    if "f1" in metrics:
        results["f1"] = float(f1_score(y_test, predictions))

    try:
        # Save to a YAML file
        with open(output_path, "w") as file:
            yaml.safe_dump(results, file)
        print(f"Evaluation metrics for best model saved to {output_path}")
    except Exception as e:
        print(f"Error saving Evaluation metrics for best model : {e}")

    return results
