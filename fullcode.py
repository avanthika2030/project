import pandas as pd
import numpy as np
import pickle

def process_dataset(file_path):
    # Read dataset
    data = pd.read_csv(file_path)

    # Ensure required columns exist
    required_columns = ['Customer Reference ID', 'Purchase Amount (USD)', 'Date Purchase']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataset.")

    # Data cleaning
    data['Date Purchase'] = pd.to_datetime(data['Date Purchase'], errors='coerce')
    data['Purchase Amount (USD)'] = data['Purchase Amount (USD)'].fillna(data['Purchase Amount (USD)'].median())
    data = data.dropna(subset=['Date Purchase', 'Customer Reference ID'])

    # Calculate Days Since Last Purchase and Churn Risk
    data['Days Since Last Purchase'] = data.groupby('Customer Reference ID')['Date Purchase'].transform(
        lambda x: (data['Date Purchase'].max() - x.max()).days)
    data['Churn Risk'] = np.where(data['Days Since Last Purchase'] > 90, 1, 0)

    # Load XGBoost model
    with open('xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)

    # Predict churn risk
    features = ['Purchase Amount (USD)', 'Days Since Last Purchase']
    X = data[features]
    data['Churn Prediction'] = xgb_model.predict(X)

    # Calculate Days Between Purchases and Predict Next Purchase Date
    data['Days Between Purchases'] = data.groupby('Customer Reference ID')['Date Purchase'].transform(
        lambda x: x.diff().dt.days)
    data['Days Between Purchases'] = data['Days Between Purchases'].fillna(data['Days Between Purchases'].mean())

    with open('lr_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)

    lr_features = ['Purchase Amount (USD)', 'Days Between Purchases']
    data['Predicted Days Until Next Purchase'] = lr_model.predict(data[lr_features])
    data['Predicted Next Purchase Date'] = data['Date Purchase'] + pd.to_timedelta(
        data['Predicted Days Until Next Purchase'].round(), unit='D')

    # Calculate Retention Rate
    data['Purchase Count'] = data.groupby('Customer Reference ID')['Customer Reference ID'].transform('count')
    retained_customers = data[data['Purchase Count'] > 5]['Customer Reference ID'].nunique()
    total_customers = data['Customer Reference ID'].nunique()
    retention_rate = (retained_customers / total_customers) * 100

    # Compute CLV
    data['Total Revenue'] = data.groupby('Customer Reference ID')['Purchase Amount (USD)'].transform('sum')
    data['Customer Lifetime Value (CLV)'] = data['Total Revenue'] * (retention_rate / 100)

    # Compute Best Engagement Time
    data['Hour of Purchase'] = data['Date Purchase'].dt.hour
    data['Minute of Purchase'] = data['Date Purchase'].dt.minute
    data['Hour-Minute'] = data['Hour of Purchase'].astype(str).str.zfill(2) + ':' + data['Minute of Purchase'].astype(str).str.zfill(2)
    best_engagement_time = data['Hour-Minute'].mode()[0]

    # Prepare output
    clv = data[['Customer Reference ID', 'Customer Lifetime Value (CLV)']].drop_duplicates().to_dict(orient='records')
    next_purchase_date = data[['Customer Reference ID', 'Predicted Next Purchase Date']].drop_duplicates().to_dict(orient='records')
    churn_rate = data[['Customer Reference ID', 'Churn Prediction']].drop_duplicates().to_dict(orient='records')

    return {
        "clv": clv,
        "best_engagement_time": best_engagement_time,
        "next_purchase_date": next_purchase_date,
        "churn_rate": churn_rate,
    }
