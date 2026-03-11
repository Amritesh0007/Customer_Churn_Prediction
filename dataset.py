import pandas as pd
import numpy as np

np.random.seed(42)

n = 100000

# 1. Customer Demographics (≈ 10–15% influence)
age = np.random.randint(18, 70, n)
gender = np.random.choice(["Male", "Female", "Non-binary"], n, p=[0.48, 0.48, 0.04])
city = np.random.choice(["New York", "London", "Tokyo", "Berlin", "Mumbai"], n)
occupation = np.random.choice(["Engineer", "Teacher", "Doctor", "Artist", "Sales", "Other"], n)
dependents = np.random.randint(0, 5, n)
income = np.random.normal(60000, 20000, n).clip(15000, 150000)

# 2. Transaction Behaviour (≈ 30–40% influence) ⭐ MOST IMPORTANT
transaction_frequency = np.random.poisson(12, n)
transaction_amount = np.random.normal(5000, 2000, n).clip(200, 20000)
last_transaction_days = np.random.randint(1, 120, n)
account_balance = np.random.uniform(500, 50000, n)
payment_failures = np.random.poisson(0.5, n)

# 3. Digital Engagement (≈ 20–30%)
app_login_frequency = np.random.poisson(15, n)
email_open_rate = np.random.uniform(0, 1, n)
feature_usage = np.random.randint(1, 10, n)
website_visits = np.random.poisson(10, n)
session_duration = np.random.normal(15, 5, n).clip(1, 60)

# 4. Customer Support Signals (≈ 15–25%)
complaints = np.random.poisson(1, n)
support_calls = np.random.poisson(1, n)
refund_requests = np.random.poisson(0.2, n)
service_tickets = np.random.poisson(0.8, n)

data = pd.DataFrame({
    "customer_id": ["C" + str(i).zfill(6) for i in range(1, n+1)],
    "age": age,
    "gender": gender,
    "city": city,
    "occupation": occupation,
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
})

# Refined Churn Logic to target 25% churn rate
# Weighting factors based on user request
demographic_score = (data["age"] > 60).astype(int) * 0.1 + (data["income"] < 25000).astype(int) * 0.05
transaction_score = (data["last_transaction_days"] > 60).astype(int) * 0.2 + (data["payment_failures"] > 1).astype(int) * 0.1 + (data["transaction_frequency"] < 5).astype(int) * 0.1
engagement_score = (data["email_open_rate"] < 0.15).astype(int) * 0.15 + (data["app_login_frequency"] < 5).astype(int) * 0.15
support_score = (data["complaints"] > 2).astype(int) * 0.15 + (data["refund_requests"] > 1).astype(int) * 0.1

total_score = demographic_score + transaction_score + engagement_score + support_score

# Add some randomness (noise)
total_score += np.random.normal(0, 0.1, n)

# Determine threshold for ~25% churn
threshold = np.percentile(total_score, 75)
data["churn"] = (total_score >= threshold).astype(int)

# Save dataset
data.to_csv("synthetic_churn_dataset_100k.csv", index=False)

print("Dataset generated: synthetic_churn_dataset_100k.csv")
print(f"Total Rows: {len(data)}")
print(f"Churn Distribution:\n{data['churn'].value_counts()}")
print(data.head())