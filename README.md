# Production ML Churn Prediction Pipeline

An end-to-end Machine Learning pipeline that processes features, trains an XGBoost classifier, and serves real-time predictions via a FastAPI REST API.

## Tech Stack
- Python
- XGBoost (Machine Learning Model)
- FastAPI & Uvicorn (REST API Serving)
- Pandas & PyArrow (Feature Store & Parquet support)
- Pydantic (Data validation)

## Directory Structure
```
production-ml-churn-pipeline/
|-- config/
|   +-- config.yaml
|-- src/
|   |-- features.py
|   |-- train.py
|   +-- app.py
|-- requirements.txt
|-- .gitignore
+-- README.md
```

## How to Run the Project

### 1. Clone & Setup Environment
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd production-ml-churn-pipeline
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2. Feature Engineering
```bash
python src/features.py
```

### 3. Train Model
```bash
python src/train.py
```

### 4. Run REST API Server
```bash
python -m uvicorn src.app:app --port 8000
```
Navigate to http://127.0.0.1:8000/docs to test live predictions.
