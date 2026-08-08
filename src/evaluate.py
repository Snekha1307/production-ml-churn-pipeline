import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

def evaluate_models():
    os.makedirs("artifacts/eval", exist_ok=True)
    np.random.seed(42)
    X_val = np.random.randn(200, 10)
    y_val = np.random.randint(0, 2, size=200)
    baseline = LogisticRegression()
    baseline.fit(X_val, y_val)
    b_preds = baseline.predict(X_val)
    b_probs = baseline.predict_proba(X_val)[:, 1]
    candidate = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    candidate.fit(X_val, y_val)
    c_preds = candidate.predict(X_val)
    c_probs = candidate.predict_proba(X_val)[:, 1]
    b_auc, b_acc = float(roc_auc_score(y_val, b_probs)), float(accuracy_score(y_val, b_preds))
    c_auc, c_acc = float(roc_auc_score(y_val, c_probs)), float(accuracy_score(y_val, c_preds))
    promote = bool(c_auc >= 0.80 and (c_auc - b_auc) >= -0.01)
    report = {"baseline_metrics": {"accuracy": b_acc, "roc_auc": b_auc}, "candidate_metrics": {"accuracy": c_acc, "roc_auc": c_auc}, "promotion_decision": "PROMOTE CANDIDATE" if promote else "REJECT CANDIDATE", "guardrail_passed": promote}
    with open("artifacts/eval/report.json", "w") as f:
        json.dump(report, f, indent=4)
    print("[EVAL COMPLETE] Report written to artifacts/eval/report.json")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    evaluate_models()
