#!/bin/bash

echo "=================================================="
echo "Running All Customer Churn Prediction Programs"
echo "=================================================="

# Set library path for XGBoost
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH

echo ""
echo "1️⃣  Running Feature Importance Analysis..."
echo "=================================================="
python feature_importance.py

echo ""
echo "2️⃣  Running Performance Report (Text-Based)..."
echo "=================================================="
python visualize_performance.py

echo ""
echo "3️⃣  Running Prediction System Demo..."
echo "=================================================="
python predict_churn.py

echo ""
echo "✅ All programs completed!"
echo ""
echo "📊 Text-based performance report generated above"
echo "🎯 Use 'test_custom_customer.py' to test individual customers interactively"
echo ""
echo "💡 Note: For graphical charts, install matplotlib separately:"
echo "   pip install matplotlib"
