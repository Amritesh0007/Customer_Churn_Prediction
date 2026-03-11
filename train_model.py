import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
print("Loading dataset...")
df = pd.read_csv("synthetic_churn_dataset_100k.csv")

# Preprocessing
print("Preprocessing data...")
# Drop customer_id as it's not a feature
X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

# Encode categorical variables
le_gender = LabelEncoder()
X["gender"] = le_gender.fit_transform(X["gender"])

le_city = LabelEncoder()
X["city"] = le_city.fit_transform(X["city"])

le_occupation = LabelEncoder()
X["occupation"] = le_occupation.fit_transform(X["occupation"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model (Robust alternative to XGBoost)
print("Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    n_jobs=-1 # Use all available cores
)

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# SAVE MODEL AND ENCODERS
print("\nSaving model artifacts...")
joblib.dump(model, "churn_model.joblib")
joblib.dump(le_gender, "le_gender.joblib")
joblib.dump(le_city, "le_city.joblib")
joblib.dump(le_occupation, "le_occupation.joblib")
print("Model artifacts saved successfully.")

# Demonstrate high churn probability
print("\n--- Model Prediction Example ---")
sample_idx = np.where((df["last_transaction_days"] > 80) & (df["payment_failures"] > 2) & (df["complaints"] > 3))[0]

if len(sample_idx) > 0:
    idx = sample_idx[0]
    sample_data = X.iloc[[idx]]
    prob = model.predict_proba(sample_data)[0][1]
    print(f"Predicting churn for Customer {df.iloc[idx]['customer_id']}:")
    print(f"Probability of churn: {prob:.2%}")
else:
    print("No high-risk sample found in current batch, showing first customer prediction:")
    prob = model.predict_proba(X.iloc[[0]])[0][1]
    print(f"Probability of churn: {prob:.2%}")
