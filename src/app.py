import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Churn Prediction API")

model = joblib.load("models/model.joblib")

class PredictionInput(BaseModel):
    Tenure: int = 12
    MonthlyCharges: float = 70.5
    TotalCharges: float = 850.0
    AverageMonthlySpend: float = 70.8
    ContractType_One_year: bool = Field(False, alias="ContractType_One year")
    ContractType_Two_year: bool = Field(False, alias="ContractType_Two year")
    InternetService_Fiber_optic: bool = Field(False, alias="InternetService_Fiber optic")
    InternetService_No: bool = Field(False, alias="InternetService_No")
    TechSupport_Yes: bool = Field(False, alias="TechSupport_Yes")

    class Config:
        populate_by_name = True

@app.get("/")
def home():
    return {"status": "API is online!"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    data = pd.DataFrame([input_data.model_dump(by_alias=True)])
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability)
    }
