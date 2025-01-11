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

# 1. Load and split the dataset
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 2. Set up MLflow for experiment tracking
mlflow.set_experiment("Iris_Classification_Experiment")  # Custom experiment name

# 3. Logistic Regression Model (underfitting by reducing iterations and increasing regularization)
with mlflow.start_run():

    lr_model = LogisticRegression(max_iter=50, C=1000, solver='liblinear')  # Underfitting
    lr_model.fit(X_train, y_train)

    lr_predictions = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    lr_conf_matrix = confusion_matrix(y_test, lr_predictions)

    # Log parameters and metrics
    mlflow.log_param("model", "Logistic Regression")
    mlflow.log_param("max_iter", 50)
    mlflow.log_param("C", 1000)
    mlflow.log_metric("accuracy", lr_accuracy)

    # Ensure the artifacts directory exists
    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")
    
    # Save and log confusion matrix image
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.figure(figsize=(6,6))
    sns.heatmap(lr_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title("Confusion Matrix - Logistic Regression")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"artifacts/lr_conf_matrix_{timestamp}.png")
    mlflow.log_artifact(f"artifacts/lr_conf_matrix_{timestamp}.png")

    # Log the model
    mlflow.sklearn.log_model(lr_model, "logistic_regression_model")

    # Register the model
    log_model_uri = f"runs:/{mlflow.active_run().info.run_id}/logistic_regression_model"
    try:
        registered_model = mlflow.register_model(log_model_uri, "Logistic_Regression_Model")
    except mlflow.exceptions.MlflowException as e:
        print(f"Error registering Logistic Regression model: {e}")

    mlflow.end_run()

    print(f"Logistic Regression Accuracy: {lr_accuracy}")
    print("Confusion Matrix:")
    print(lr_conf_matrix)

# 4. Random Forest Model (underfitting by reducing trees, depth, and min_samples_split)
with mlflow.start_run():
    
    rf_model = RandomForestClassifier(n_estimators=10, max_depth=3, min_samples_split=10, random_state=0, criterion='entropy')
    rf_model.fit(X_train, y_train)

    rf_predictions = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)

    rf_conf_matrix = confusion_matrix(y_test, rf_predictions)

    # Log parameters and metrics
    mlflow.log_param("model", "Random Forest")
    mlflow.log_param("n_estimators", 10)
    mlflow.log_param("max_depth", 3)
    mlflow.log_param("min_samples_split", 10)
    mlflow.log_metric("accuracy", rf_accuracy)

    # Save and log confusion matrix image
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.figure(figsize=(6,6))
    sns.heatmap(rf_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"artifacts/rf_conf_matrix_{timestamp}.png")
    mlflow.log_artifact(f"artifacts/rf_conf_matrix_{timestamp}.png")

    # Log the model
    mlflow.sklearn.log_model(rf_model, "random_forest_model")

    # Register the model
    rf_model_uri = f"runs:/{mlflow.active_run().info.run_id}/random_forest_model"
    try:
        registered_rf_model = mlflow.register_model(rf_model_uri, "Random_Forest_Model")
    except mlflow.exceptions.MlflowException as e:
        print(f"Error registering Random Forest model: {e}")

    mlflow.end_run()

    print(f"Random Forest Accuracy: {rf_accuracy}")
    print("Confusion Matrix:")
    print(rf_conf_matrix)

# Load the latest models
try:
    loaded_lr_model = mlflow.sklearn.load_model("models:/Logistic_Regression_Model/latest")
except Exception as e:
    print(f"Error loading Logistic Regression model: {e}")

try:
    loaded_rf_model = mlflow.sklearn.load_model("models:/Random_Forest_Model/latest")
except Exception as e:
    print(f"Error loading Random Forest model: {e}")
