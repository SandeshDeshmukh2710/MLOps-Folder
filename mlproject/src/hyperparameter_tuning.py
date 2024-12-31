from sklearn.model_selection import GridSearchCV
import yaml
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import os

MODEL_MAPPING = {
    "LogisticRegression": LogisticRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "XGBClassifier": XGBClassifier,
}


def train_and_tune_models(X_train, y_train, config):
    """
    Train and tune multiple models using GridSearchCV based on the configuration file.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training labels.
        config (dict): Configuration dictionary with model details and grid search settings.

    Returns:
        dict: Best model, its parameters, and score for each algorithm.
    """
    best_models = {}
    models_config = config["models"]
    grid_search_config = config["grid_search"]

    for model_name, details in models_config.items():
        print(f"Training and tuning {model_name}...")

        # Initialize the model
        if model_name not in MODEL_MAPPING:
            raise ValueError(f"Model '{model_name}' is not supported. Choose from {list(MODEL_MAPPING.keys())}.")
        
        model = MODEL_MAPPING[model_name]()

        # Perform GridSearchCV
        param_grid = details["params"]
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=grid_search_config["scoring"],
            cv=grid_search_config["cv"],
            verbose=grid_search_config["verbose"],
            n_jobs=grid_search_config["n_jobs"],
        )
        search.fit(X_train, y_train)

        # Save the best model and its performance
        best_models[model_name] = {
            "model": search.best_estimator_,
            "params": search.best_params_,
            "score": search.best_score_,
        }

    return best_models


def save_best_model(models: dict, file_path: str):
    """Save the best model among all trained models.

    Args:
        models (dict): Dictionary containing models and their scores.
        file_path (str): Path to save the best model file.
    """
    best_model_name = max(models, key=lambda x: models[x]["score"])
    best_model = models[best_model_name]["model"]
    print(f"Best Model: {best_model_name} with score {models[best_model_name]['score']}")
    joblib.dump(best_model, file_path)


def save_best_model_params(best_models: dict, output_path: str):
    """
    Save the best parameters for the model with the highest score to a YAML file.

    Args:
        best_models (dict): Dictionary containing best models and their parameters.
        output_path (str): Path to save the YAML file with the best model's parameters.
    """
    best_model_name = max(best_models, key=lambda x: best_models[x]["score"])
    best_model_details = {
        "model_name": best_model_name,
        "best_params": best_models[best_model_name]["params"],
        "best_score": float(best_models[best_model_name]["score"]),
    }

    try:
        with open(output_path, "w") as file:
            yaml.safe_dump(best_model_details, file)
        print(f"Best model details saved to {output_path}")
    except Exception as e:
        print(f"Error saving best model details: {e}")
