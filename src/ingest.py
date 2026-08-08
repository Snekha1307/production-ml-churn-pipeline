import os
import pandas as pd
from datetime import datetime

def ingest_batch_data(raw_file_path, data_store_path="data/processed_training.csv"):
    os.makedirs(os.path.dirname(data_store_path), exist_ok=True)
    if not os.path.exists(raw_file_path):
        print(f"[INGEST ERROR] File {raw_file_path} not found.")
        return
    df_raw = pd.read_csv(raw_file_path)
    n_rows = len(df_raw)
    if os.path.exists(data_store_path):
        df_raw.to_csv(data_store_path, mode="a", header=False, index=False)
    else:
        df_raw.to_csv(data_store_path, index=False)
    log_entry = f"{datetime.utcnow().isoformat()} - Ingested {n_rows} rows from {raw_file_path} into {data_store_path}"
    print(f"[INGEST SUCCESS] {log_entry}")
    with open("data/ingestion_log.txt", "a") as f:
        f.write(log_entry + "\n")

if __name__ == "__main__":
    print("Running batch ingestion pipeline...")
