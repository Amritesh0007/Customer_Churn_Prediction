#!/bin/bash

echo "=================================================="
echo "Customer Churn Prediction - Complete Demo"
echo "=================================================="
echo ""

# Find conda python
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON_CMD="$CONDA_PREFIX/bin/python"
else
    PYTHON_CMD="python3"
fi

echo "Using Python: $PYTHON_CMD"
echo ""

# Test imports first
echo "📦 Checking dependencies..."
$PYTHON_CMD -c "import pandas, numpy, xgboost, sklearn, joblib; print('✅ Core libraries OK')" 2>&1 | head -5

echo ""
echo "1️⃣  Running Feature Importance Analysis..."
echo "=================================================="
$PYTHON_CMD feature_importance.py

echo ""
echo "2️⃣  Generating Performance Report..."
echo "=================================================="
$PYTHON_CMD visualize_performance.py

echo ""
echo "3️⃣  Testing Prediction System..."
echo "=================================================="
$PYTHON_CMD predict_churn.py

echo ""
echo "✅ All programs completed!"
echo ""
echo "🎯 Next Steps:"
echo "   - Run 'python test_custom_customer.py' for interactive mode"
echo "   - Review outputs above for insights"
