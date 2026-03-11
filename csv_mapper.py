import streamlit as st
import pandas as pd

st.set_page_config(page_title="CSV Column Mapper", layout="wide")

st.title("🔄 CSV Column Name Converter")
st.markdown("""
Upload your customer CSV file and map your column names to the format expected by the churn prediction model.
""")

# Expected columns
EXPECTED_COLS = [
    'customer_id', 'age', 'gender', 'city', 'occupation', 'dependents', 'income',
    'transaction_frequency', 'transaction_amount', 'last_transaction_days', 
    'account_balance', 'payment_failures', 'app_login_frequency', 'email_open_rate',
    'feature_usage', 'website_visits', 'session_duration', 'complaints', 
    'support_calls', 'refund_requests', 'service_tickets'
]

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(f"**Your columns:** {', '.join(df.columns)}")
    
    # Create mapping
    st.subheader("Map Your Columns")
    st.write("Select which of your columns corresponds to each expected column:")
    
    mapping = {}
    cols1, cols2 = st.columns(2)
    
    for i, expected_col in enumerate(EXPECTED_COLS):
        col_options = list(df.columns) + ["<SKIP>"]
        
        # Try to auto-match
        default_idx = 0
        for j, your_col in enumerate(df.columns):
            if your_col.lower().replace(' ', '_').replace('-', '_') == expected_col:
                default_idx = j
                break
        
        with (cols1 if i < len(EXPECTED_COLS)//2 else cols2):
            mapping[expected_col] = st.selectbox(
                f"{expected_col}",
                options=col_options,
                index=default_idx if default_idx < len(col_options)-1 else len(col_options)-1,
                key=expected_col
            )
    
    if st.button("🔄 Convert CSV"):
        # Check for skipped required columns
        skipped = [k for k, v in mapping.items() if v == "<SKIP>"]
        if skipped:
            st.warning(f"⚠️ These columns will be skipped: {', '.join(skipped)}")
        
        # Create new dataframe
        new_df = pd.DataFrame()
        
        for expected_col, your_col in mapping.items():
            if your_col != "<SKIP>":
                new_df[expected_col] = df[your_col]
        
        st.success("✅ Conversion complete!")
        st.write(f"**New columns:** {', '.join(new_df.columns)}")
        st.dataframe(new_df.head())
        
        # Download button
        csv = new_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Converted CSV",
            data=csv,
            file_name="converted_customer_data.csv",
            mime="text/csv"
        )
