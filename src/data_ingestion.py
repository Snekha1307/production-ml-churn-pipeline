import pandas as pd
import yaml

def load_data(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    df = pd.read_csv(config["raw_data_path"])
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
    return df

if __name__ == "__main__":
    df = load_data("config/config.yaml")
