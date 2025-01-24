import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

class IrisClassification:
    def __init__(self, experiment_name="Iris_Classification_Experiment"):
        # Load dataset and split it into train/test sets
        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=42)
        
        # Setup MLflow experiment
        mlflow.set_experiment(experiment_name)
    
    def log_confusion_matrix(self, conf_matrix, model_name):
        """
        Logs confusion matrix as an image artifact
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plt.figure(figsize=(6, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=self.iris.target_names, yticklabels=self.iris.target_names)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()

        # Ensure the artifacts directory exists
        if not os.path.exists("artifacts"):
            os.makedirs("artifacts")

        # Save and log the confusion matrix image
        plt.savefig(f"artifacts/{model_name.lower().replace(' ', '_')}_conf_matrix_{timestamp}.png")
        mlflow.log_artifact(f"artifacts/{model_name.lower().replace(' ', '_')}_conf_matrix_{timestamp}.png")

    def train_logistic_regression(self):
        """
        Trains and logs Logistic Regression model with MLflow
        """
        with mlflow.start_run():
            lr_model = LogisticRegression(max_iter=50, C=1000, solver='liblinear')  # Underfitting
            lr_model.fit(self.X_train, self.y_train)

            lr_predictions = lr_model.predict(self.X_test)
            lr_accuracy = accuracy_score(self.y_test, lr_predictions)
            lr_conf_matrix = confusion_matrix(self.y_test, lr_predictions)

            # Log parameters, metrics, and model
            mlflow.log_param("model", "Logistic Regression")
            mlflow.log_param("max_iter", 50)
            mlflow.log_param("C", 1000)
            mlflow.log_metric("accuracy", lr_accuracy)

            # Log confusion matrix as an image
            self.log_confusion_matrix(lr_conf_matrix, "Logistic Regression")

            # Log the model
            mlflow.sklearn.log_model(lr_model, "logistic_regression_model")

            # Register the model
            log_model_uri = f"runs:/{mlflow.active_run().info.run_id}/logistic_regression_model"
            try:
                registered_model = mlflow.register_model(log_model_uri, "Logistic_Regression_Model")
            except mlflow.exceptions.MlflowException as e:
                print(f"Error registering Logistic Regression model: {e}")

            print(f"Logistic Regression Accuracy: {lr_accuracy}")
            print("Confusion Matrix:")
            print(lr_conf_matrix)

            mlflow.end_run()

    def train_random_forest(self):
        """
        Trains and logs Random Forest model with MLflow
        """
        with mlflow.start_run():
            rf_model = RandomForestClassifier(n_estimators=10, max_depth=3, min_samples_split=10, random_state=0, criterion='entropy')
            rf_model.fit(self.X_train, self.y_train)

            rf_predictions = rf_model.predict(self.X_test)
            rf_accuracy = accuracy_score(self.y_test, rf_predictions)
            rf_conf_matrix = confusion_matrix(self.y_test, rf_predictions)

            # Log parameters, metrics, and model
            mlflow.log_param("model", "Random Forest")
            mlflow.log_param("n_estimators", 10)
            mlflow.log_param("max_depth", 3)
            mlflow.log_param("min_samples_split", 10)
            mlflow.log_metric("accuracy", rf_accuracy)

            # Log confusion matrix as an image
            self.log_confusion_matrix(rf_conf_matrix, "Random Forest")

            # Log the model
            mlflow.sklearn.log_model(rf_model, "random_forest_model")

            # Register the model
            rf_model_uri = f"runs:/{mlflow.active_run().info.run_id}/random_forest_model"
            try:
                registered_rf_model = mlflow.register_model(rf_model_uri, "Random_Forest_Model")
            except mlflow.exceptions.MlflowException as e:
                print(f"Error registering Random Forest model: {e}")

            print(f"Random Forest Accuracy: {rf_accuracy}")
            print("Confusion Matrix:")
            print(rf_conf_matrix)

            mlflow.end_run()

    def load_models(self):
        """
        Loads the latest models from MLflow Model Registry
        """
        try:
            loaded_lr_model = mlflow.sklearn.load_model("models:/Logistic_Regression_Model/latest")
            print("Loaded Logistic Regression model successfully!")
        except Exception as e:
            print(f"Error loading Logistic Regression model: {e}")

        try:
            loaded_rf_model = mlflow.sklearn.load_model("models:/Random_Forest_Model/latest")
            print("Loaded Random Forest model successfully!")
        except Exception as e:
            print(f"Error loading Random Forest model: {e}")

# Usage Example

# Initialize the class
iris_classification = IrisClassification()

# Train and log the Logistic Regression model
iris_classification.train_logistic_regression()

# Train and log the Random Forest model
iris_classification.train_random_forest()

# Load the latest models
iris_classification.load_models()
