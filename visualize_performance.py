import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

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

print("\n" + "="*70)
print("CUSTOMER CHURN PREDICTION MODEL - PERFORMANCE REPORT")
print("="*70)

# 1. Overall Metrics
print("\n" + "="*70)
print("📊 OVERALL PERFORMANCE METRICS")
print("="*70)
accuracy = accuracy_score(y, y_pred)
fpr, tpr, _ = roc_curve(y, y_pred_proba)
roc_auc = auc(fpr, tpr)

print(f"\n✅ Overall Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"✅ ROC-AUC Score:     {roc_auc:.4f} ({roc_auc*100:.2f}%)")
print(f"📈 Total Samples:     {len(y):,}")
print(f"   - Correct Predictions: {(y_pred == y).sum():,}")
print(f"   - Incorrect Predictions: {(y_pred != y).sum():,}")

# 2. Confusion Matrix (Text Version)
print("\n" + "="*70)
print("📋 CONFUSION MATRIX")
print("="*70)
cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n                    ACTUAL")
print(f"                  No      Yes")
print(f"                ┌─────────────┐")
print(f"PREDICTED  No   │ {tn:6d} │ {fn:6d} │")
print(f"         Yes   │ {fp:6d} │ {tp:6d} │")
print(f"                └─────────────┘")
print(f"\nTrue Negatives (TN):  {tn:,} - Correctly predicted Stay")
print(f"False Positives (FP): {fp:,} - Incorrectly predicted Churn")
print(f"False Negatives (FN): {fn:,} - Incorrectly predicted Stay")
print(f"True Positives (TP):  {tp:,} - Correctly predicted Churn")

# 3. Class-wise Metrics
print("\n" + "="*70)
print("📈 CLASS-WISE PERFORMANCE")
print("="*70)
from sklearn.metrics import precision_score, recall_score, f1_score

print(f"\n{'Metric':<15} {'Non-Churn (0)':<20} {'Churn (1)':<20}")
print("-" * 55)
print(f"{'Precision':<15} {precision_score(y, y_pred):<20.4f} {precision_score(y, y_pred, pos_label=1):<20.4f}")
print(f"{'Recall':<15} {recall_score(y, y_pred):<20.4f} {recall_score(y, y_pred, pos_label=1):<20.4f}")
print(f"{'F1-Score':<15} {f1_score(y, y_pred):<20.4f} {f1_score(y, y_pred, pos_label=1):<20.4f}")
print(f"{'Support':<15} {(y==0).sum():<20,} {(y==1).sum():<20,}")

# 4. Feature Importance
print("\n" + "="*70)
print("🎯 TOP 20 FEATURE IMPORTANCE")
print("="*70)
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n{'Rank':<6} {'Importance':<12} {'Feature Name'}")
print("-" * 55)
for idx, row in importance_df.head(20).iterrows():
    rank = list(importance_df.index).index(idx) + 1
    bar = "█" * int(row['Importance'] * 100)
    print(f"{rank:<6} {row['Importance']:<12.4f} {row['Feature']} {bar}")

# 5. Category-wise Analysis
print("\n" + "="*70)
print("📊 FEATURE INFLUENCE BY BUSINESS CATEGORY")
print("="*70)

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

print(f"\n1️⃣  Customer Demographics:")
print(f"    Expected: 10-15% | Actual: {demographic_imp/total_imp*100:.2f}%")
print(f"    Score: {demographic_imp:.4f}")

print(f"\n2️⃣  Transaction Behaviour ⭐:")
print(f"    Expected: 30-40% | Actual: {transaction_imp/total_imp*100:.2f}%")
print(f"    Score: {transaction_imp:.4f}")

print(f"\n3️⃣  Digital Engagement:")
print(f"    Expected: 20-30% | Actual: {digital_imp/total_imp*100:.2f}%")
print(f"    Score: {digital_imp:.4f}")

print(f"\n4️⃣  Customer Support Signals:")
print(f"    Expected: 15-25% | Actual: {support_imp/total_imp*100:.2f}%")
print(f"    Score: {support_imp:.4f}")

# 6. Key Insights
print("\n" + "="*70)
print("💡 KEY INSIGHTS & RECOMMENDATIONS")
print("="*70)

top_5_features = importance_df.head(5)['Feature'].tolist()
print(f"\n🎯 Top 5 Churn Drivers:")
for i, feat in enumerate(top_5_features, 1):
    imp_val = importance_df[importance_df['Feature'] == feat]['Importance'].values[0]
    print(f"   {i}. {feat}: {imp_val:.4f}")

print(f"\n✅ Validation:")
if 'payment_failures' in top_5_features or 'last_transaction_days' in top_5_features:
    print("   ✓ Transaction Behaviour features are dominant - matches expectation!")

if 'complaints' in top_5_features or 'support_calls' in top_5_features:
    print("   ✓ Customer Support signals are influential - validates hypothesis!")

print(f"\n💡 Strategic Recommendations:")
print("   1. Focus retention efforts on customers with high payment failures")
print("   2. Monitor customers with long gaps since last transaction (>60 days)")
print("   3. Implement proactive support for customers with multiple complaints")
print("   4. Improve digital engagement features to reduce churn risk")
print("   5. Early warning system: track app_login_frequency and email_open_rate")

# 7. Probability Distribution Stats
print("\n" + "="*70)
print("📊 PREDICTION PROBABILITY STATISTICS")
print("="*70)

churn_proba = y_pred_proba[y == 1]
stay_proba = y_pred_proba[y == 0]

print(f"\nFor Customers Who Actually Churned:")
print(f"   Mean Probability: {churn_proba.mean():.4f}")
print(f"   Std Deviation:    {churn_proba.std():.4f}")
print(f"   Min:              {churn_proba.min():.4f}")
print(f"   Max:              {churn_proba.max():.4f}")
print(f"   Median:           {np.median(churn_proba):.4f}")

print(f"\nFor Customers Who Stayed:")
print(f"   Mean Probability: {stay_proba.mean():.4f}")
print(f"   Std Deviation:    {stay_proba.std():.4f}")
print(f"   Min:              {stay_proba.min():.4f}")
print(f"   Max:              {stay_proba.max():.4f}")
print(f"   Median:           {np.median(stay_proba):.4f}")

print("\n" + "="*70)
print("✅ TEXT-BASED REPORT COMPLETE")
print("="*70)

# GENERATE GRAPHICAL VISUALIZATIONS
print("\n" + "="*70)
print("📊 GENERATING GRAPHICAL VISUALIZATIONS")
print("="*70)

try:
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Customer Churn Prediction Model - Performance Dashboard', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Confusion Matrix
    ax1 = plt.subplot(2, 3, 1)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                annot_kws={'size': 14, 'weight': 'bold'})
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('Actual', fontsize=12)
    
    # 2. ROC Curve
    ax2 = plt.subplot(2, 3, 2)
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc="lower right", fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. Churn Distribution
    ax3 = plt.subplot(2, 3, 3)
    churn_counts = y.value_counts()
    colors = ['#2ecc71', '#e74c3c']
    bars = ax3.bar(['Stay', 'Churn'], churn_counts.values, color=colors, 
                   edgecolor='black', linewidth=2)
    ax3.set_title('Churn Distribution', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_xlabel('Class', fontsize=12)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height):,}', ha='center', va='bottom', 
                 fontsize=11, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Feature Importance (Top 15)
    ax4 = plt.subplot(2, 3, 4)
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)
    
    colors_gradient = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_df)))
    bars = ax4.barh(importance_df['Feature'], importance_df['Importance'], 
                    color=colors_gradient, edgecolor='black', linewidth=1)
    ax4.set_title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Importance Score', fontsize=12)
    ax4.invert_yaxis()
    ax4.grid(axis='x', alpha=0.3)
    
    # 5. Prediction Probability Distribution
    ax5 = plt.subplot(2, 3, 5)
    churn_proba = y_pred_proba[y == 1]
    stay_proba = y_pred_proba[y == 0]
    ax5.hist(churn_proba, bins=30, alpha=0.6, color='#e74c3c', 
             label='Churn Customers', edgecolor='black')
    ax5.hist(stay_proba, bins=30, alpha=0.6, color='#2ecc71', 
             label='Stay Customers', edgecolor='black')
    ax5.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Probability of Churn', fontsize=12)
    ax5.set_ylabel('Frequency', fontsize=12)
    ax5.legend(fontsize=10)
    ax5.grid(axis='y', alpha=0.3)
    ax5.axvline(x=0.5, color='blue', linestyle='--', linewidth=2, 
                label='Decision Threshold')
    
    # 6. Classification Report Heatmap
    ax6 = plt.subplot(2, 3, 6)
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    metrics_data = [
        [precision_score(y, y_pred), precision_score(y, y_pred, pos_label=1)],
        [recall_score(y, y_pred), recall_score(y, y_pred, pos_label=1)],
        [f1_score(y, y_pred), f1_score(y, y_pred, pos_label=1)]
    ]
    
    metrics_labels = ['Precision', 'Recall', 'F1-Score']
    class_labels = ['Non-Churn (0)', 'Churn (1)']
    
    im = ax6.imshow(metrics_data, cmap='YlOrRd', aspect='auto')
    ax6.set_xticks(np.arange(len(class_labels)))
    ax6.set_yticks(np.arange(len(metrics_labels)))
    ax6.set_xticklabels(class_labels)
    ax6.set_yticklabels(metrics_labels)
    ax6.set_title('Performance Metrics by Class', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(metrics_labels)):
        for j in range(len(class_labels)):
            text = ax6.text(j, i, f'{metrics_data[i][j]:.3f}',
                           ha="center", va="center", color="black", 
                           fontsize=13, fontweight='bold')
    
    plt.colorbar(im, ax=ax6, shrink=0.8)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    plt.savefig('model_performance_dashboard.png', dpi=300, bbox_inches='tight')
    print("\n✅ Dashboard saved as 'model_performance_dashboard.png'")
    print("   Opening image viewer...")
    
    # Try to open the image (works on Mac, Windows, Linux)
    import subprocess
    import platform
    system = platform.system()
    try:
        if system == 'Darwin':  # macOS
            subprocess.run(['open', 'model_performance_dashboard.png'])
        elif system == 'Windows':
            subprocess.run(['start', 'model_performance_dashboard.png'], shell=True)
        else:  # Linux
            subprocess.run(['xdg-open', 'model_performance_dashboard.png'])
    except Exception as e:
        print(f"   Note: Could not auto-open image. Please open 'model_performance_dashboard.png' manually.")
    
except Exception as e:
    print(f"\n⚠️  Could not generate graphical visualization: {e}")
    print("   Text-based report is still available above.")

print("\n🎯 Next Steps:")
print("   - Run 'python predict_churn.py' to test predictions")
print("   - Run 'python test_custom_customer.py' for interactive mode")
print("   - Review feature importance to understand churn drivers")
print("   - Check 'model_performance_dashboard.png' for visual charts")
