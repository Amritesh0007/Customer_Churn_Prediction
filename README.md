# 🎯 Customer Churn Prediction - Professional Dashboard

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55.0-red.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-green.svg)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Project Overview

A **professional, interactive web dashboard** for predicting customer churn using XGBoost machine learning. Features real-time predictions, bulk analysis, AI insights, and an executive dashboard with beautiful visualizations.

### ✨ Key Features

- 🎯 **Real-time Predictions** - Individual customer churn prediction with AI insights
- 📊 **Executive Dashboard** - KPIs, charts, and high-risk customer tracking
- 👥 **Bulk Prediction** - Upload CSV files for batch predictions (10,000+ customers)
- 💡 **AI Insights Panel** - Strategic recommendations and campaign simulator
- 🔄 **Smart Column Mapping** - Auto-recognizes 100+ column name variations
- 📈 **Model Performance** - 86.2% accuracy, 91.6% ROC-AUC

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Amritesh0007/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run web_dashboard.py --server.port 8501
```

### 4. Open in Browser
Navigate to: **http://localhost:8501**

---

## 📁 Project Structure

```
Customer_Churn_Prediction/
├── 📊 Core Files
│   ├── web_dashboard.py              # Main Streamlit web application
│   ├── train_model.py                # XGBoost model training script
│   ├── dataset.py                    # Synthetic data generator (100K customers)
│   └── synthetic_churn_dataset_100k.csv  # Training dataset
│
├── 🎨 Visualization & Reports
│   ├── executive_dashboard.py        # Static executive dashboard generator
│   ├── feature_importance.py         # Feature importance analysis
│   ├── visualize_performance.py      # Model performance visualizations
│   └── professional_churn_dashboard.png  # Sample dashboard output
│
├── 🔧 Utilities
│   ├── predict_churn.py              # Individual prediction API
│   ├── test_custom_customer.py       # Interactive testing tool
│   ├── csv_mapper.py                 # CSV column mapping utility
│   ├── sample_customer_data.csv      # Sample CSV template
│   ├── run_all.sh                    # Batch runner script
│   └── start_dashboard.sh            # Dashboard launcher
│
├── 💾 Model Artifacts
│   ├── churn_model.joblib            # Trained XGBoost model
│   ├── le_gender.joblib              # Gender encoder
│   ├── le_city.joblib                # City encoder
│   └── le_occupation.joblib          # Occupation encoder
│
└── 📖 Documentation
    ├── README.md                     # This file
    ├── QUICK_START.md                # Quick start guide
    ├── WEB_DASHBOARD_GUIDE.md        # Web dashboard documentation
    └── DASHBOARD_GUIDE.md            # Static dashboard guide
```
---
## Project Architecture ---

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/f0655bcd-73a4-49c2-8429-3aba5b38a23c" />


---

## 🎯 Features Deep Dive

### 1. Executive Dashboard (`📊` tab)
- **KPI Cards**: Model accuracy (86.2%), churn rate (25%), total customers (100K)
- **Top 10 Churn Drivers**: Interactive horizontal bar chart
- **Risk Distribution**: By transaction behavior, demographics, engagement, support
- **High-Risk Table**: Top 15 customers with >70% churn probability
- **Segmentation Pie Chart**: Low/Medium/High/Very High risk distribution

### 2. Predict Customer Churn (`🔮` tab)
**Individual prediction form with:**
- Customer Demographics (age, gender, city, occupation, income)
- Transaction Behavior (frequency, amount, last transaction, payment failures)
- Digital Engagement (app logins, email open rate, feature usage)
- Support Signals (complaints, support calls, refund requests)

**Output:**
- Risk Level: 🔴 HIGH / 🟡 MEDIUM / 🟢 LOW
- Churn Probability: e.g., "78.5%"
- AI Insights: Specific recommendations
- Risk Factors: Visual breakdown

### 3. Bulk Prediction (`👥` tab)
**Upload CSV with automatic column mapping:**

**Auto-recognizes these variations:**
- `app_logins`, `login_count` → `app_login_frequency`
- `email_opens`, `open_rate` → `email_open_rate`
- `balance` → `account_balance`
- `failed_payments` → `payment_failures`
- And 100+ more!

**Features:**
- Smart defaults for missing non-critical columns
- Real-time mapping feedback
- Summary statistics by risk level
- Downloadable results CSV

### 4. AI Insights (`💡` tab)
**Strategic analytics:**
- Portfolio-wide churn analysis
- Statistical findings (e.g., ">45 days inactive = 4x risk")
- Campaign impact simulator with sliders
- ROI calculator (budget, retention rate, customer value)
- Priority action items checklist

---

## 📊 Model Performance

### Overall Metrics
- **Accuracy**: 86.2%
- **ROC-AUC**: 91.6%
- **Precision (Churn)**: 76.1%
- **Recall (Churn)**: 65.2%
- **Training Data**: 100,000 synthetic customers

### Top 5 Churn Drivers
1. **last_transaction_days** (45.2%) - Most critical factor
2. **email_open_rate** (18.4%)
3. **complaints** (9.7%)
4. **age** (9.5%)
5. **payment_failures** (5.2%)

### Category Influence
- **Transaction Behaviour**: 54.01% ⭐ (Expected: 30-40%)
- **Digital Engagement**: 21.91% (Expected: 20-30%)
- **Customer Demographics**: 12.74% (Expected: 10-15%)
- **Customer Support Signals**: 11.35% (Expected: 15-25%)

---

## 💻 Usage Examples

### Individual Prediction
```python
from predict_churn import predict_churn

customer = {
    'age': 45, 'gender': 'Female', 'city': 'London',
    'occupation': 'Engineer', 'income': 75000,
    'last_transaction_days': 15, 'payment_failures': 0,
    'app_login_frequency': 12, 'email_open_rate': 0.6,
    'complaints': 0
}

result = predict_churn(customer)
print(f"Churn Probability: {result[0]['churn_probability']:.2%}")
print(f"Risk Level: {result[0]['risk_level']}")
```

### Bulk Prediction
```python
import pandas as pd
import joblib

# Load model
model = joblib.load('churn_model.joblib')

# Upload CSV
df = pd.read_csv('your_customers.csv')

# Preprocess and predict
X = df.drop(columns=['customer_id'])
predictions = model.predict_proba(X)

# Add to dataframe
df['churn_probability'] = predictions[:, 1]
df['risk_level'] = df['churn_probability'].apply(
    lambda x: 'HIGH' if x > 0.7 else 'MEDIUM' if x > 0.4 else 'LOW'
)

# Save results
df.to_csv('predictions_with_churn_risk.csv', index=False)
```

---

## 🔧 Configuration

### Required Python Packages
See `requirements.txt`:
```
streamlit==1.55.0
pandas==2.3.3
numpy==2.3.5
xgboost==3.2.0
scikit-learn==1.7.2
matplotlib==3.10.8
seaborn==0.13.2
plotly==5.24.1
joblib==1.5.2
```

### CSV Format for Bulk Prediction

**Required Columns (exact names or alternatives):**
```csv
customer_id,age,gender,city,occupation,dependents,income,
transaction_frequency,transaction_amount,last_transaction_days,
account_balance,payment_failures,app_login_frequency,email_open_rate,
feature_usage,website_visits,session_duration,complaints,
support_calls,refund_requests,service_tickets
```

**Sample Template:** See `sample_customer_data.csv`

---

## 🎨 Customization

### Change Port
```bash
streamlit run web_dashboard.py --server.port 8502
```

### Network Access
```bash
streamlit run web_dashboard.py --server.address 0.0.0.0
```

### Headless Mode
```bash
streamlit run web_dashboard.py --server.headless true
```

### Regenerate Dataset
```bash
python dataset.py  # Creates synthetic_churn_dataset_100k.csv
```

### Retrain Model
```bash
python train_model.py  # Saves new churn_model.joblib
```

---

## 📈 Business Use Cases

### 1. Call Center
**Scenario**: Customer calls with complaint  
**Action**: Agent enters data → Gets 82% churn risk → Transfers to retention specialist  
**Result**: Personalized offer → Customer retained ✅

### 2. Marketing Campaign
**Scenario**: Q2 retention campaign planning  
**Action**: Upload 10K customers → Identify 1,200 high-risk → Simulate $15K budget  
**Result**: $180K revenue saved, 12:1 ROI → Campaign approved ✅

### 3. Executive Review
**Scenario**: Monthly board meeting  
**Action**: Show dashboard KPIs, top churn drivers, AI insights  
**Result**: Board approves AI expansion ✅

---

## 🐛 Troubleshooting

### Dashboard Won't Open
```bash
# Check port availability
lsof -i :8501

# Kill process if needed
kill -9 <PID>

# Restart
streamlit run web_dashboard.py --server.port 8501
```

### Model Not Loading
```bash
# Verify files exist
ls -la *.joblib

# If missing, retrain
python train_model.py
```

### Column Mapping Errors
```bash
# Use sample template
cp sample_customer_data.csv your_data.csv

# Or use mapper tool
streamlit run csv_mapper.py --server.port 8502
```

---

## 📚 Documentation

- **[README.md](README.md)** - This file (project overview)
- **[QUICK_START.md](QUICK_START.md)** - Quick start guide (5 minutes)
- **[WEB_DASHBOARD_GUIDE.md](WEB_DASHBOARD_GUIDE.md)** - Complete web dashboard docs
- **[DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md)** - Static dashboard guide

---

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

### Areas for Improvement
- [ ] Add more column name variations to mapping
- [ ] Integrate with real banking databases
- [ ] Add time-series churn prediction
- [ ] Implement A/B testing framework
- [ ] Create mobile app version

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Amritesh Kumar**  
GitHub: [@Amritesh0007](https://github.com/Amritesh0007)

---

## 🙏 Acknowledgments

- XGBoost team for the excellent library
- Streamlit team for the amazing dashboard framework
- Scikit-learn contributors

---

## 📬 Contact

For questions or support:
- Open an issue on GitHub
- Email: amriteshkumar475@gmail.com

---

## 🎉 Ready to Get Started?

```bash
git clone https://github.com/Amritesh0007/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction
pip install -r requirements.txt
streamlit run web_dashboard.py
```

**Your professional churn prediction dashboard will open at http://localhost:8501!** 🚀

---

*Last Updated: March 2026*
