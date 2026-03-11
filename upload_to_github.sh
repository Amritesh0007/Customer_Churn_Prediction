#!/bin/bash

echo "🚀 Uploading Customer Churn Prediction to GitHub..."
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install git first."
    exit 1
fi

# Initialize git repo if not already done
if [ ! -d .git ]; then
    echo "📦 Initializing git repository..."
    git init
fi

# Add all files
echo "📝 Adding files to git..."
git add .

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo "⚠️  No changes to commit."
else
    # Commit changes
    echo "💾 Committing changes..."
    git commit -m "Initial commit: Complete Customer Churn Prediction Dashboard"
fi

# Check if remote is set
if ! git remote get-url origin &> /dev/null; then
    echo "🔗 Adding GitHub remote..."
    git remote add origin https://github.com/Amritesh0007/Customer_Churn_Prediction.git
fi

# Push to GitHub
echo "⬆️  Pushing to GitHub..."
git branch -M main
git push -u origin main

echo ""
echo "✅ Upload complete!"
echo ""
echo "🌐 View your repository at:"
echo "   https://github.com/Amritesh0007/Customer_Churn_Prediction"
echo ""
echo "📋 Next Steps:"
echo "   1. Visit the GitHub repository URL above"
echo "   2. Add a license file (optional)"
echo "   3. Share your project!"
echo ""
