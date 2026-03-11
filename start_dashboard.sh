#!/bin/bash

echo "🚀 Starting Customer Churn Prediction Web Dashboard..."
echo ""
echo "📊 Opening dashboard in your browser..."
echo ""
echo "⏳ Please wait while Streamlit loads..."
echo ""
echo "💡 The dashboard will open at: http://localhost:8501"
echo ""
echo "Features:"
echo "  ✓ Executive Dashboard with KPIs"
echo "  ✓ Individual Customer Churn Prediction"
echo "  ✓ Bulk CSV Upload & Prediction"
echo "  ✓ AI Insights & Campaign Simulator"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run streamlit
streamlit run web_dashboard.py --server.port 8501 --browser.gatherUsageStats false
