import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression


file_path = 'Fashion_Retail_Sales.csv'
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

xgb_model = XGBClassifier( eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

y_pred_churn = xgb_model.predict(X_test)

print("Churn Risk Prediction Accuracy:", accuracy_score(y_test, y_pred_churn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_churn))


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

y_pred_date = lr_model.predict(X_lr_test)
print("\nNext Purchase Date Prediction RMSE:", np.sqrt(mean_squared_error(y_lr_test, y_pred_date)))

data['Predicted Days Until Next Purchase'] = lr_model.predict(X_lr)
data['Predicted Next Purchase Date'] = data['Date Purchase'] + pd.to_timedelta(data['Predicted Days Until Next Purchase'].round(), unit='D')



# Calculate Retention Rate
data['Purchase Count'] = data.groupby('Customer Reference ID')['Customer Reference ID'].transform('count')
retained_customers = data[data['Purchase Count'] > 5]['Customer Reference ID'].nunique()
total_customers = data['Customer Reference ID'].nunique()
retention_rate = (retained_customers / total_customers) * 100

print(f"\nRetention Rate: {retention_rate:.2f}%")

# CLV
data['Total Revenue'] = data.groupby('Customer Reference ID')['Purchase Amount (USD)'].transform('sum')
data['Customer Lifetime Value (CLV)'] = data['Total Revenue'] * (retention_rate / 100)

#  Best Engagement Time
data['Hour of Purchase'] = data['Date Purchase'].dt.hour
data['Minute of Purchase'] = data['Date Purchase'].dt.minute
data['Hour-Minute'] = data['Hour of Purchase'].astype(str).str.zfill(2) + ':' + data['Minute of Purchase'].astype(str).str.zfill(2)
best_engagement_time = data['Hour-Minute'].mode()[0]

print(f"Best Engagement Time: {best_engagement_time}")

# Outputs
print("\nChurn Risk Prediction:")
print(data[['Customer Reference ID', 'Churn Risk']].drop_duplicates().head())

print("\nNext Purchase Date Predictions:")
print(data[['Customer Reference ID', 'Predicted Next Purchase Date']].drop_duplicates().head())


updated_dataset_path = 'updated_dataset.csv'
data.to_csv(updated_dataset_path, index=False)
print(f"\nUpdated dataset saved to: {updated_dataset_path}")
