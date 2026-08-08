# Production ML Churn Prediction Pipeline

![CI Pipeline](https://github.com/Snekha1307/production-ml-churn-pipeline/actions/workflows/ci.yml/badge.svg)

An end-to-end production-ready Machine Learning pipeline designed to predict customer churn. Built with FastAPI, XGBoost, Docker, and GitHub Actions CI/CD.

---

## Project Structure

```text
├── .github/workflows/   # GitHub Actions CI/CD pipeline
├── config/              # Configuration files
├── models/              # Trained ML models (.joblib)
├── src/                 # Application source code
│   ├── app.py           # FastAPI REST API implementation
│   ├── predict.py       # Inference pipeline
│   └── train.py         # Model training script
├── tests/               # Unit test suite for API & model
├── Dockerfile           # Production container configuration
├── requirements.txt     # Python dependency specifications
└── README.md            # Project documentation
```

---

## Getting Started

### 1. Prerequisites
- Python 3.12+
- Git
- Docker (optional)

### 2. Local Setup
Clone the repository and set up a virtual environment:

```bash
git clone [https://github.com/Snekha1307/production-ml-churn-pipeline.git](https://github.com/Snekha1307/production-ml-churn-pipeline.git)
cd production-ml-churn-pipeline

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

---

## Running Tests & API Locally

### Run Unit Tests
To execute the automated `pytest` suite:
```bash
python -m pytest -v
```

### Launch FastAPI Server
To start the prediction REST API server locally:
```bash
uvicorn src.app:app --reload
```
Once running, access the interactive Swagger documentation at **`http://127.0.0.1:8000/docs`**.

---

## Docker Deployment

### Build Container Image
```bash
docker build -t churn-pipeline-app .
```

### Run Container
```bash
docker run -d -p 8000:8000 churn-pipeline-app
```

---

## Continuous Integration (CI/CD)

This repository uses **GitHub Actions** to automate quality checks. On every `push` or `pull_request` to the `main` branch, the CI pipeline automatically:
1. Sets up Python 3.12 environment.
2. Installs required dependencies.
3. Executes the full `pytest` suite to ensure API and model integrity.
