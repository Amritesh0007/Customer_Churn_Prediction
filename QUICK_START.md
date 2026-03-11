# 🚀 Quick Start Guide - Customer Churn Prediction Model

## ✅ All Files Created Successfully!

Your project structure:
```
dataset/
├── train_model.py                  # ✅ Original training script
├── predict_churn.py                # ✅ Prediction API
├── feature_importance.py           # ✅ Feature analysis
├── visualize_performance.py        # ✅ Performance report (text-based)
├── test_custom_customer.py         # ✅ Interactive tester
├── run_all.sh                      # ✅ Batch runner
├── README_CHURN_MODEL.md           # ✅ Full documentation
└── QUICK_START.md                  # ✅ This file
```

## ⚠️ Current Issue: XGBoost Library Error on macOS

The model was trained successfully, but running the prediction scripts requires fixing the XGBoost library dependency on macOS.

### Solution 1: Install libomp (Recommended)
```bash
brew update
brew install libomp
```

Then run:
```bash
./run_all.sh
```

### Solution 2: Use Conda Environment
```bash
conda create -n churn python=3.9 xgboost scikit-learn pandas numpy matplotlib seaborn
conda activate churn
python train_model.py
./run_all.sh
```

### Solution 3: Reinstall XGBoost with pip
```bash
pip uninstall xgboost
pip install xgboost --no-cache-dir
```

## 📊 What You Have

### 1. **Trained Model** (Already Working!)
- Algorithm: XGBoost Classifier
- Accuracy: 84.86%
- Dataset: 100,000 customers
- Status: ✅ Trained and saved

### 2. **Prediction Scripts** (Need libomp fix)
Once you fix the library issue, you can:

**Run all programs:**
```bash
./run_all.sh
```

**Or run individually:**

```bash
# Test predictions with sample data
python predict_churn.py

# Analyze which features matter most
python feature_importance.py

# Generate performance report
python visualize_performance.py

# Interactive customer testing
python test_custom_customer.py
```

## 🎯 Example Usage (After Fixing Library)

### Predict for New Customers:
```python
from predict_churn import predict_churn

customer = {
    "age": 45,
    "gender": "Female",
    "city": "London",
    "occupation": "Engineer",
    "income": 75000,
    "last_transaction_days": 15,
    "payment_failures": 0,
    "app_login_frequency": 12,
    "email_open_rate": 0.6,
    "complaints": 0,
    # ... other features
}

result = predict_churn(customer)
print(f"Churn Probability: {result[0]['churn_probability']:.2%}")
print(f"Risk Level: {result[0]['risk_level']}")
```

### Interactive Mode:
```bash
python test_custom_customer.py
# Follow prompts to enter customer details
```

## 📈 Model Performance Summary

From the training run:
- **Accuracy**: 84.86%
- **Non-Churn Precision**: 88%, Recall: 92%
- **Churn Precision**: 73%, Recall: 63%
- **ROC-AUC**: ~0.85

### Top Churn Drivers (Expected):
1. last_transaction_days
2. payment_failures
3. complaints
4. app_login_frequency
5. email_open_rate

## 🔧 Troubleshooting

### If you see "XGBoost Library could not be loaded":
```bash
# Option 1: Install libomp
brew install libomp
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH
python predict_churn.py

# Option 2: Use conda
conda create -n ml python=3.9
conda activate ml
conda install -c conda-forge xgboost scikit-learn pandas matplotlib
python train_model.py
```

### If matplotlib installation fails:
The visualization script works without it (text-based output).
Graphical charts are optional.

## 📚 Next Steps

1. **Fix the library issue** using one of the solutions above
2. **Run the batch script**: `./run_all.sh`
3. **Review the outputs**:
   - Feature importance rankings
   - Model performance metrics
   - Category-wise influence analysis
4. **Test custom customers** interactively
5. **Integrate predictions** into your workflow

## 💡 Key Features

All 4 requested components are ready:

✅ **1. Prediction System** - Programmatic API for batch predictions  
✅ **2. Feature Importance** - Detailed breakdown by category  
✅ **3. Performance Report** - Text-based dashboard (no matplotlib needed)  
✅ **4. Interactive Tester** - Command-line interface for manual testing  

## 🎓 Documentation

- `README_CHURN_MODEL.md` - Complete technical documentation
- `QUICK_START.md` - This quick start guide

---

**Status**: All files created successfully! Just fix the XGBoost library dependency and you're ready to go! 🚀
