import yaml
from src.logger import setup_logger
from src.data_processing import load_data, split_data
from src.model_evaluation import evaluate_model
from src.model_training import train_model, save_model
from src.hyperparameter_tuning import train_and_tune_models

import mlflow
import mlflow.sklearn

# Initialize the logger
logger = setup_logger("logs/info.log", "logs/error.log")

def main():
    try:
        logger.info("--------------Starting the MLPipeline--------------")

        # Start MLflow Experiment
        mlflow.set_experiment("MLPipeline_Experiments")
        with mlflow.start_run(run_name="MLPipeline_Run"):
            # Load configuration
            with open("config/config.yaml", "r") as file:
                config = yaml.safe_load(file)
            logger.info("Configuration loaded successfully.")
            mlflow.log_artifact("config/config.yaml")  # Log the configuration file

            # Data processing
            data = load_data(config["data"]["path"])
            logger.info(f"Data loaded from {config['data']['path']}. Shape: {data.shape}")

            X_train, X_test, y_train, y_test = split_data(
                data, 
                **config["evaluation"]
            )
            logger.info(f"Data split completed: Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

            # Hyperparameter tuning with multiple models
            logger.info("Starting hyperparameter tuning process...")
            best_models = train_and_tune_models(X_train, y_train, config)
            logger.info("Hyperparameter tuning completed successfully.")

            # Select the best model
            best_model_name = max(best_models, key=lambda x: best_models[x]["score"])
            best_model_details = best_models[best_model_name]
            logger.info(f"Best model after tuning - {best_model_name}")
            p = best_model_details["params"]
            logger.info(f"Params of Best model after tuning - {p}")
            mlflow.log_params(best_model_details["params"])  # Log the best hyperparameters

            # Model training
            logger.info("Starting model training process...")
            model = train_model(X_train, y_train,X_test, y_test, best_model_name, best_model_details["params"])
            mlflow.sklearn.log_model(model, "model")  # Log the trained model
            logger.info(f"Model trained successfully.")

            # Model evaluation
            logger.info("Starting model evaluation process...")
            metrics = evaluate_model(model, X_test, y_test, config["metric_path"]["path"])
            mlflow.log_metrics(metrics)  # Log evaluation metrics
            logger.info("Model evaluation completed.")
            logger.info(f"Performance of {best_model_name} - {metrics}")

            # Log custom metrics and artifacts
            mlflow.log_artifact(config["metric_path"]["path"])  # Log metrics file

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        logger.info("--------------Pipeline execution completed--------------")

if __name__ == "__main__":
    main()
