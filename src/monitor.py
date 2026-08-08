import numpy as np
import pandas as pd

def check_data_quality_and_drift(reference_df, current_df, threshold=2.0):
    print("Data Quality and Drift Check")
    null_count = current_df.isnull().sum().sum()
    if null_count > 0:
        print(f"[WARNING] Detected {null_count} missing values in incoming batch!")
    else:
        print("[PASS] No missing values detected.")
    for col in reference_df.select_dtypes(include=[np.number]).columns:
        ref_mean, ref_std = reference_df[col].mean(), reference_df[col].std()
        curr_mean = current_df[col].mean()
        if ref_std > 0:
            z_score = abs(curr_mean - ref_mean) / ref_std
            if z_score > threshold:
                print(f"[ALERT] Drift detected in feature {col}! Z-score = {z_score:.2f}")
            else:
                print(f"[PASS] Feature {col} within normal bounds (Z-score = {z_score:.2f}).")

def retraining_trigger_signal(drift_alert, new_data_count, days_since_retrain):
    if drift_alert or new_data_count >= 10000 or days_since_retrain >= 30:
        print("[TRIGGER] Retraining criteria met! Initiating automated pipeline.")
        return True
    print("[INFO] System healthy. Retraining not required.")
    return False

if __name__ == "__main__":
    ref_data = pd.DataFrame(np.random.randn(100, 3), columns=["f1", "f2", "f3"])
    curr_data = pd.DataFrame(np.random.randn(50, 3) + 0.5, columns=["f1", "f2", "f3"])
    check_data_quality_and_drift(ref_data, curr_data)
    retraining_trigger_signal(drift_alert=False, new_data_count=12000, days_since_retrain=10)
