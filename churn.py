import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
import pickle

# Function to process dataset and generate predictions
def process_dataset(file_path):
    data = pd.read_csv(file_path)

    # EDA
    required_columns = ['Customer Reference ID', 'Purchase Amount (USD)', 'Date Purchase']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataset.")

    data['Date Purchase'] = pd.to_datetime(data['Date Purchase'], errors='coerce')

    data['Purchase Amount (USD)'] = data['Purchase Amount (USD)'].fillna(data['Purchase Amount (USD)'].median())

    data = data.dropna(subset=['Date Purchase', 'Customer Reference ID'])

    # Predicting Churn Risk using XGBoost
    data['Days Since Last Purchase'] = data.groupby('Customer Reference ID')['Date Purchase'].transform(
        lambda x: (data['Date Purchase'].max() - x.max()).days)
    data['Churn Risk'] = np.where(data['Days Since Last Purchase'] > 90, 1, 0)

    features = ['Purchase Amount (USD)', 'Days Since Last Purchase']
    X = data[features]
    y = data['Churn Risk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)

    # Save the XGBoost model
    with open('xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)

    y_pred_churn = xgb_model.predict(X_test)
    print("Churn Risk Prediction Accuracy:", accuracy_score(y_test, y_pred_churn))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_churn))

process_dataset('Fashion_Retail_Sales.csv')