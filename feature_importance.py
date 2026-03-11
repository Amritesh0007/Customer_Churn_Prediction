import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset and model
print("Loading dataset and model...")
df = pd.read_csv("synthetic_churn_dataset_100k.csv")
model = joblib.load("churn_model.joblib")

# Preprocessing
X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

# Encode categorical variables
le_gender = LabelEncoder()
X["gender"] = le_gender.fit_transform(X["gender"])

le_city = LabelEncoder()
X["city"] = le_city.fit_transform(X["city"])

le_occupation = LabelEncoder()
X["occupation"] = le_occupation.fit_transform(X["occupation"])

# Get predictions
y_pred = model.predict(X)
y_pred_proba = model.predict_proba(X)[:, 1]

print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Get feature importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 20 Most Important Features:")
print("-" * 60)
for idx, row in importance_df.head(20).iterrows():
    print(f"{row['Feature']:30s} {row['Importance']:.4f}")

# Categorize features by influence group
print("\n" + "="*60)
print("FEATURE INFLUENCE BY CATEGORY")
print("="*60)

demographic_features = ['age', 'gender', 'city', 'occupation', 'dependents', 'income']
transaction_features = ['transaction_frequency', 'transaction_amount', 'last_transaction_days', 
                       'account_balance', 'payment_failures']
digital_features = ['app_login_frequency', 'email_open_rate', 'feature_usage', 
                   'website_visits', 'session_duration']
support_features = ['complaints', 'support_calls', 'refund_requests', 'service_tickets']

def calculate_group_importance(features, importance_df):
    mask = importance_df['Feature'].isin(features)
    return importance_df[mask]['Importance'].sum()

demographic_imp = calculate_group_importance(demographic_features, importance_df)
transaction_imp = calculate_group_importance(transaction_features, importance_df)
digital_imp = calculate_group_importance(digital_features, importance_df)
support_imp = calculate_group_importance(support_features, importance_df)

total_imp = demographic_imp + transaction_imp + digital_imp + support_imp

print(f"\n1. Customer Demographics (10-15% expected):")
print(f"   Actual Importance: {demographic_imp:.4f} ({demographic_imp/total_imp*100:.2f}%)")

print(f"\n2. Transaction Behaviour ⭐ (30-40% expected):")
print(f"   Actual Importance: {transaction_imp:.4f} ({transaction_imp/total_imp*100:.2f}%)")

print(f"\n3. Digital Engagement (20-30% expected):")
print(f"   Actual Importance: {digital_imp:.4f} ({digital_imp/total_imp*100:.2f}%)")

print(f"\n4. Customer Support Signals (15-25% expected):")
print(f"   Actual Importance: {support_imp:.4f} ({support_imp/total_imp*100:.2f}%)")

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)

top_5_features = importance_df.head(5)['Feature'].tolist()
print(f"\n🎯 Top 5 drivers of churn:")
for i, feat in enumerate(top_5_features, 1):
    imp_val = importance_df[importance_df['Feature'] == feat]['Importance'].values[0]
    print(f"   {i}. {feat}: {imp_val:.4f}")

if 'payment_failures' in top_5_features or 'last_transaction_days' in top_5_features:
    print("\n✅ Transaction Behaviour features are dominant - matches business expectation!")

if 'complaints' in top_5_features or 'support_calls' in top_5_features:
    print("✅ Customer Support signals are influential - validates hypothesis!")

print("\n💡 Recommendations:")
print("   - Focus retention efforts on customers with high payment failures")
print("   - Monitor customers with long gaps since last transaction")
print("   - Proactive support for customers with multiple complaints")
print("   - Improve digital engagement to reduce churn risk")
