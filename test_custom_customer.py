import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

# Load trained model and encoders
print("="*60)
print("CUSTOMER CHURN PREDICTION - INTERACTIVE MODE")
print("="*60)
print("\nLoading trained model...")

try:
    model = joblib.load("churn_model.joblib")
    le_gender = joblib.load("le_gender.joblib")
    le_city = joblib.load("le_city.joblib")
    le_occupation = joblib.load("le_occupation.joblib")
    print("✅ Model loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error: Model files not found. Please run train_model.py first.")
    exit(1)

def get_customer_input():
    """Get customer data from user input"""
    print("\n" + "-"*60)
    print("Enter Customer Details (or 'q' to quit)")
    print("-"*60)
    
    try:
        age = int(input("Age: ") or 30)
        
        gender_input = input("Gender (Male/Female/Non-binary): ").strip() or "Male"
        if gender_input not in le_gender.classes_:
            print(f"Invalid gender. Using default: Male")
            gender_input = "Male"
        
        city_input = input("City (New York/London/Tokyo/Berlin/Mumbai): ").strip() or "New York"
        if city_input not in le_city.classes_:
            print(f"Invalid city. Using default: New York")
            city_input = "New York"
        
        occupation_input = input("Occupation (Engineer/Teacher/Doctor/Artist/Sales/Other): ").strip() or "Engineer"
        if occupation_input not in le_occupation.classes_:
            print(f"Invalid occupation. Using default: Engineer")
            occupation_input = "Engineer"
        
        dependents = int(input("Number of Dependents (0-4): ") or 1)
        income = float(input("Annual Income ($): ") or 60000)
        
        print("\n--- Transaction Behaviour ---")
        transaction_frequency = int(input("Transaction Frequency (per month): ") or 12)
        transaction_amount = float(input("Average Transaction Amount ($): ") or 5000)
        last_transaction_days = int(input("Days Since Last Transaction: ") or 30)
        account_balance = float(input("Account Balance ($): ") or 10000)
        payment_failures = int(input("Payment Failures (last 3 months): ") or 0)
        
        print("\n--- Digital Engagement ---")
        app_login_frequency = int(input("App Logins (per month): ") or 15)
        email_open_rate = float(input("Email Open Rate (0-1): ") or 0.5)
        feature_usage = int(input("Feature Usage Score (1-10): ") or 5)
        website_visits = int(input("Website Visits (per month): ") or 10)
        session_duration = float(input("Avg Session Duration (minutes): ") or 15)
        
        print("\n--- Customer Support Signals ---")
        complaints = int(input("Number of Complaints: ") or 0)
        support_calls = int(input("Support Calls: ") or 0)
        refund_requests = int(input("Refund Requests: ") or 0)
        service_tickets = int(input("Service Tickets: ") or 0)
        
        return {
            "age": age,
            "gender": gender_input,
            "city": city_input,
            "occupation": occupation_input,
            "dependents": dependents,
            "income": income,
            "transaction_frequency": transaction_frequency,
            "transaction_amount": transaction_amount,
            "last_transaction_days": last_transaction_days,
            "account_balance": account_balance,
            "payment_failures": payment_failures,
            "app_login_frequency": app_login_frequency,
            "email_open_rate": email_open_rate,
            "feature_usage": feature_usage,
            "website_visits": website_visits,
            "session_duration": session_duration,
            "complaints": complaints,
            "support_calls": support_calls,
            "refund_requests": refund_requests,
            "service_tickets": service_tickets
        }
        
    except ValueError as e:
        print(f"\n❌ Invalid input. Please enter numeric values where required.")
        return None

def predict_churn(customer_data):
    """Predict churn for a single customer"""
    df = pd.DataFrame([customer_data])
    
    # Encode categorical variables
    df["gender"] = le_gender.transform(df["gender"])
    df["city"] = le_city.transform(df["city"])
    df["occupation"] = le_occupation.transform(df["occupation"])
    
    # Predict
    churn_pred = model.predict(df)[0]
    churn_proba = model.predict_proba(df)[0][1]
    
    return churn_pred, churn_proba

def display_prediction(customer_data, churn_pred, churn_proba):
    """Display prediction results"""
    risk_level = "🔴 HIGH RISK" if churn_proba > 0.7 else "🟡 MEDIUM RISK" if churn_proba > 0.4 else "🟢 LOW RISK"
    
    print("\n" + "="*60)
    print("CHURN PREDICTION RESULT")
    print("="*60)
    print(f"\nCustomer Profile:")
    print(f"  Age: {customer_data['age']}, {customer_data['gender']}, {customer_data['city']}")
    print(f"  Occupation: {customer_data['occupation']}")
    print(f"  Income: ${customer_data['income']:,.0f}")
    
    print(f"\n📊 Prediction:")
    print(f"  Risk Level: {risk_level}")
    print(f"  Probability of Churning: {churn_proba:.2%}")
    print(f"  Probability of Staying: {1-churn_proba:.2%}")
    print(f"  Predicted to Churn: {'⚠️ YES' if churn_pred == 1 else '✅ NO'}")
    
    print(f"\n💡 Key Risk Factors:")
    if customer_data['last_transaction_days'] > 60:
        print(f"   ⚠️ High days since last transaction ({customer_data['last_transaction_days']} days)")
    if customer_data['payment_failures'] > 2:
        print(f"   ⚠️ Multiple payment failures ({customer_data['payment_failures']})")
    if customer_data['complaints'] > 2:
        print(f"   ⚠️ High number of complaints ({customer_data['complaints']})")
    if customer_data['email_open_rate'] < 0.2:
        print(f"   ⚠️ Low email engagement ({customer_data['email_open_rate']:.1%})")
    if customer_data['app_login_frequency'] < 5:
        print(f"   ⚠️ Low app usage ({customer_data['app_login_frequency']} logins/month)")
    
    if churn_proba < 0.3:
        print(f"   ✅ Strong engagement signals")
        print(f"   ✅ Low risk indicators")
    
    print("\n" + "="*60)

# Main interactive loop
if __name__ == "__main__":
    print("\n🎯 Welcome to the Customer Churn Prediction System!")
    print("   This tool uses XGBoost ML model to predict customer churn risk.\n")
    
    while True:
        customer_data = get_customer_input()
        
        if customer_data is None:
            print("\n⚠️ Skipping invalid input...")
            continue
        
        churn_pred, churn_proba = predict_churn(customer_data)
        display_prediction(customer_data, churn_pred, churn_proba)
        
        choice = input("\nWould you like to predict another customer? (y/n): ").strip().lower()
        if choice != 'y':
            print("\n" + "="*60)
            print("Thank you for using the Churn Prediction System!")
            print("="*60)
            break
