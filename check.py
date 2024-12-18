from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, classification_report
file_path = 'Fashion_Retail_Sales.csv'
data = pd.read_csv(file_path)
required_columns = ['Customer Reference ID', 'Purchase Amount (USD)', 'Date Purchase']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"Required column '{col}' is missing from the dataset.")

data['Date Purchase'] = pd.to_datetime(data['Date Purchase'], errors='coerce')

data['Purchase Amount (USD)'] = data['Purchase Amount (USD)'].fillna(data['Purchase Amount (USD)'].median())

data = data.dropna(subset=['Date Purchase', 'Customer Reference ID'])

# Predict Churn Risk using XGBoost
# Calculate days since last purchase
data['Days Since Last Purchase'] = data.groupby('Customer Reference ID')['Date Purchase'].transform(
    lambda x: (data['Date Purchase'].max() - x.max()).days)
data['Churn Risk'] = np.where(data['Days Since Last Purchase'] > 90, 1, 0)
# Initialize models
rf_model = RandomForestClassifier(random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
features = ['Purchase Amount (USD)', 'Days Since Last Purchase']
X = data[features]
y = data['Churn Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and test Random Forest
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Train and test XGBoost
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Compare Metrics
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Random Forest ROC-AUC:", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]))

print("\nXGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))
print("XGBoost ROC-AUC:", roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]))

# Perform Cross-Validation for Random Forest
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='roc_auc')
print("\nRandom Forest CV AUC:", rf_cv_scores.mean())

# Perform Cross-Validation for XGBoost
xgb_cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='roc_auc')
print("XGBoost CV AUC:", xgb_cv_scores.mean())
