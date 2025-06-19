import mlflow
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np
from datetime import datetime
from dagshub import dagshub_logger
import dagshub

dagshub.init(repo_owner='dewidesis', repo_name='SMSML_Dewi', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/dewidesis/SMSML_Dewi.mlflow")

data = pd.read_csv("lung_cancer_preprocessing.csv")

data = data.astype("float64")

X = data.drop("lung_cancer", axis=1)
y = data["lung_cancer"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.2
)

input_example = X_train[0:5]

X_train, X_test, y_train, y_test = train_test_split(
    data.drop("lung_cancer", axis=1),
    data["lung_cancer"],
    random_state=42,
    test_size=0.2
)
input_example = X_train[0:5]

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"KNN_Modelling_{timestamp}"

with mlflow.start_run(run_name=run_name):
    # Log parameters
    n_neighbors = 5
    algorithm = 'auto'
    mlflow.log_param("n_neighbors", n_neighbors)
    mlflow.log_param("algorithm", algorithm)
    
    mlflow.autolog(disable=True)
    
    # Train model
    model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm)
    model.fit(X_train, y_train)

    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)