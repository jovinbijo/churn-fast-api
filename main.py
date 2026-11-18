import joblib
import numpy as np
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field

app=FastAPI(title="cutomer churn prediction api",description="api service powered by fatapi and scikit-learn",version="1.0.0")
try:
    model=joblib.load("churn_model.pkl")
except Exception:
    model=None

class customer_input(BaseModel):
    tenure:int=Field(...,ge=1,le=120,description="months with company",example=12)
    monthly_charges:float=Field(...,gt=0,description="monthly bill",example=85.50)
    total_charges:float=Field(...,gt=0,description="total bill amount",example=1026.00)

class prediction_output(BaseModel):
    churn_pred:int
    churn_prob:float
    risk_level:str

@app.get("/")
def home():
    return {"message":"welcome to customer churn prediction api"}

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=500,detail="ml model is missing")
    return  {"status":"healthy","model_loaded":True}      


@app.post("/predict",response_model=prediction_output)
def predict_churn(data:customer_input):
    if model is None:
        raise HTTPException(status_code=500,detail="model not loaded")
    features=np.array([[data.tenure,data.monthly_charges,data.total_charges]])
    pred_class=int(model.predict(features)[0])
    pred_prob=float(model.predict_proba(features)[0][1])
    risk="high risk" if pred_class==1 else "low risk"


    return prediction_output(churn_pred=pred_class,churn_prob=round(pred_prob*100,2),risk_level=risk)