import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import joblib

# Load trained model and encoders
print("Loading trained model...")
model = joblib.load("churn_model.joblib")
le_gender = joblib.load("le_gender.joblib")
le_city = joblib.load("le_city.joblib")
le_occupation = joblib.load("le_occupation.joblib")

def predict_churn(customer_data):
    """
    Predict churn probability for customer(s)
    
    Parameters:
    customer_data: dict or list of dicts with customer features
    
    Returns:
    dict with churn prediction and probability
    """
    # Convert to DataFrame if dict
    if isinstance(customer_data, dict):
        customer_data = [customer_data]
    
    df = pd.DataFrame(customer_data)
    
    # Encode categorical variables
    df["gender"] = le_gender.transform(df["gender"])
    df["city"] = le_city.transform(df["city"])
    df["occupation"] = le_occupation.transform(df["occupation"])
    
    # Predict
    churn_pred = model.predict(df)
    churn_proba = model.predict_proba(df)[:, 1]
    
    results = []
    for i in range(len(df)):
        results.append({
            "churn": bool(churn_pred[i]),
            "churn_probability": round(churn_proba[i], 4),
            "stay_probability": round(1 - churn_proba[i], 4),
            "risk_level": "HIGH" if churn_proba[i] > 0.7 else "MEDIUM" if churn_proba[i] > 0.4 else "LOW"
        })
    
    return results

# Example usage
if __name__ == "__main__":
    print("\n=== Customer Churn Prediction System ===\n")
    
    # Test with sample customers
    test_customers = [
        {
            "age": 65,
            "gender": "Male",
            "city": "New York",
            "occupation": "Engineer",
            "dependents": 2,
            "income": 20000,
            "transaction_frequency": 3,
            "transaction_amount": 3000,
            "last_transaction_days": 90,
            "account_balance": 1000,
            "payment_failures": 4,
            "app_login_frequency": 2,
            "email_open_rate": 0.1,
            "feature_usage": 2,
            "website_visits": 3,
            "session_duration": 5,
            "complaints": 5,
            "support_calls": 4,
            "refund_requests": 3,
            "service_tickets": 4
        },
        {
            "age": 35,
            "gender": "Female",
            "city": "London",
            "occupation": "Doctor",
            "dependents": 1,
            "income": 80000,
            "transaction_frequency": 15,
            "transaction_amount": 7000,
            "last_transaction_days": 5,
            "account_balance": 25000,
            "payment_failures": 0,
            "app_login_frequency": 20,
            "email_open_rate": 0.8,
            "feature_usage": 8,
            "website_visits": 15,
            "session_duration": 20,
            "complaints": 0,
            "support_calls": 0,
            "refund_requests": 0,
            "service_tickets": 0
        }
    ]
    
    predictions = predict_churn(test_customers)
    
    for i, pred in enumerate(predictions, 1):
        print(f"\nCustomer {i}:")
        print(f"  Churn Risk: {pred['risk_level']}")
        print(f"  Probability of Churning: {pred['churn_probability']:.2%}")
        print(f"  Probability of Staying: {pred['stay_probability']:.2%}")
        print(f"  Predicted to Churn: {'YES ⚠️' if pred['churn'] else 'NO ✅'}")
    
    print("\n=== Ready for Production Use ===")
    print("Import this module to make predictions on new customer data!")
