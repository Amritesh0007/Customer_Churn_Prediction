from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List

app = FastAPI(title="Predictive Outreach API")

# Enable CORS for the React dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and encoders
model = joblib.load("churn_model.joblib")
le_gender = joblib.load("le_gender.joblib")
le_city = joblib.load("le_city.joblib")
le_occupation = joblib.load("le_occupation.joblib")

# Load sample data for the dashboard to display
full_df = pd.read_csv("synthetic_churn_dataset_100k.csv")

class CustomerPrediction(BaseModel):
    customer_id: str
    churn_probability: float
    risk_level: str
    recommended_action: str

@app.get("/top-risk-customers", response_model=List[CustomerPrediction])
async def get_top_risk_customers(limit: int = 50):
    # For demonstration, we'll pick a diverse set of high-risk customers
    # In a real app, this would be a database query
    
    # Preprocess a subset to get probabilities
    subset = full_df.head(1000).copy()
    X = subset.drop(columns=["customer_id", "churn"])
    
    X["gender"] = le_gender.transform(X["gender"])
    X["city"] = le_city.transform(X["city"])
    X["occupation"] = le_occupation.transform(X["occupation"])
    
    probs = model.predict_proba(X)[:, 1]
    subset["churn_probability"] = probs
    
    top_risk = subset.sort_values(by="churn_probability", ascending=False).head(limit)
    
    results = []
    for _, row in top_risk.iterrows():
        prob = row["churn_probability"]
        
        # Risk Level logic
        if prob > 0.8:
            risk_level = "Critical"
            action = "Personalized RM Visit" if row["income"] > 100000 else "Direct Phone Call"
        elif prob > 0.6:
            risk_level = "High"
            action = "Personalized Email Offer"
        else:
            risk_level = "Medium"
            action = "SMS Notification"
            
        results.append(CustomerPrediction(
            customer_id=row["customer_id"],
            churn_probability=prob,
            risk_level=risk_level,
            recommended_action=action
        ))
        
    return results

@app.get("/stats")
async def get_stats():
    # Return some aggregate stats for the dashboard
    total = len(full_df)
    churn_count = int(full_df["churn"].sum())
    stay_count = total - churn_count
    
    return {
        "total_customers": total,
        "churn_rate": churn_count / total,
        "high_risk_count": churn_count,
        "revenue_at_risk": int(churn_count * full_df[full_df["churn"] == 1]["income"].mean() * 0.1) # illustrative
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
