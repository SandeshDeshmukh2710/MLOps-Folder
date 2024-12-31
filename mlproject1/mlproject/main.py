import yaml
from src.logger import setup_logger
from src.data_processing import load_data, split_data
from src.model_training import train_model, save_model
from src.model_evaluation import evaluate_model

# Initialize the logger with separate log files for INFO and ERROR logs
logger = setup_logger("logs/info.log", "logs/error.log")

def main():
    """
    Main function to execute the model development pipeline:
    - Load data
    - Split data into training and testing sets
    - Train a Random Forest model
    - Save the trained model
    - Evaluate the model
    """
    try:
        logger.info("--------------Starting the MLPipeline--------------")

        # Load configuration
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)
        logger.info("Configuration loaded successfully.")

        # Data processing
        data = load_data(config["data"]["path"])
        logger.info(f"Data loaded from {config['data']['path']}.")

        X_train, X_test, y_train, y_test = split_data(
            data, 
            # Question for later in the session -----------> 
            **config["evaluation"]
        )
        logger.info("Data split into training and testing sets.")

        # Model training
        model = train_model(X_train, y_train, config["model"]["hyperparameters"])
        save_model(model, config["model"]["save_path"])
        logger.info(f"Model trained and saved to {config['model']['save_path']}.")

        # Model evaluation
        accuracy = evaluate_model(model, X_test, y_test)
        logger.info(f"Model Accuracy: {accuracy:.2f}")
        logger.info(f"Model Hyperparams: {config['model']['hyperparameters']}")
        logger.info(f"Model Evaluation metrics: {config['evaluation']}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        logger.info("--------------Pipeline execution completed--------------")

if __name__ == "__main__":
    main()
