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
   # Predict Next Purchase Date using Linear Regression
    data['Days Between Purchases'] = data.groupby('Customer Reference ID')['Date Purchase'].transform(
        lambda x: x.diff().dt.days)
    data['Days Between Purchases'] = data['Days Between Purchases'].fillna(data['Days Between Purchases'].mean())

    lr_features = ['Purchase Amount (USD)', 'Days Between Purchases']
    X_lr = data[lr_features]
    y_lr = data['Days Between Purchases']

    X_lr_train, X_lr_test, y_lr_train, y_lr_test = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)

    lr_model = LinearRegression()
    lr_model.fit(X_lr_train, y_lr_train)

    # Save the Linear Regression model
    with open('lr_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)

    y_pred_date = lr_model.predict(X_lr_test)
    print("\nNext Purchase Date Prediction RMSE:", np.sqrt(mean_squared_error(y_lr_test, y_pred_date)))

process_dataset('Fashion_Retail_Sales.csv')