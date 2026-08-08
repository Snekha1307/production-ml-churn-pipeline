import os
import pandas as pd
import yaml
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def train_model():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    df = pd.read_parquet(config["processed_data_path"])
    X = df.drop(columns=["CustomerID", config["target_column"]])
    y = df[config["target_column"]]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=config["random_state"]
    )
    
    model = XGBClassifier(random_state=config["random_state"], n_estimators=100)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model Trained Successfully! Test Accuracy: {acc:.4f}")
    
    os.makedirs(config["model_dir"], exist_ok=True)
    joblib.dump(model, os.path.join(config["model_dir"], "model.joblib"))
    print("Model saved to models/model.joblib")

if __name__ == "__main__":
    train_model()
