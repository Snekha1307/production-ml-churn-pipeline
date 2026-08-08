import os
import pandas as pd
import numpy as np
import yaml

def engineer_features(df):
    df = df.copy()
    df["AverageMonthlySpend"] = np.where(df["Tenure"] > 0, df["TotalCharges"] / df["Tenure"], 0)
    df = pd.get_dummies(df, columns=["ContractType", "InternetService", "TechSupport"], drop_first=True)
    return df

if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    raw_df = pd.read_csv(config["raw_data_path"])
    processed_df = engineer_features(raw_df)
    os.makedirs("data/feature_store", exist_ok=True)
    processed_df.to_parquet(config["processed_data_path"], index=False)
    print("Feature engineering complete. Saved to feature store.")
