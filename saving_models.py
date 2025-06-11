
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score
import joblib

# IMPORTING DATASETS
data1 = pd.read_csv('data1.csv')
data2 = pd.read_csv('data2.csv')
data3 = pd.read_csv('data3.csv')

# Define models
model1 = LogisticRegression(C=10, solver='liblinear', max_iter=1000)
model2 = XGBClassifier(max_depth=3, n_estimators=100, use_label_encoder=False, eval_metric='logloss')
model3 = LogisticRegression(C=0.1, solver='liblinear', max_iter=1000)

######################################
# MODEL 1
print(data1.columns)
columns_to_remove = [
    'Ethnicity_Asian', 'Ethnicity_African', 'Ethnicity_European',
    'Ethnicity_middle_eastern', 'BP(Systolic)', 'BP(Diastolic)',
    'DiabetesPedigreeFunction', 'No. of Pregnancy', 'Skin Thickness(mm)'
]
data1_clean = data1.drop(columns=columns_to_remove)
print(data1_clean.columns)

print("\n===== Training with Data1 (Logistic Regression) =====")
X1 = data1_clean.drop(columns=['Diabetes'])
y1 = data1_clean['Diabetes']

# Save feature order
joblib.dump(list(X1.columns), 'model1_features.pkl')

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42, stratify=y1)

scaler1 = StandardScaler()
X_train1_scaled = scaler1.fit_transform(X_train1)
X_test1_scaled = scaler1.transform(X_test1)

model1.fit(X_train1_scaled, y_train1)

print("\n===== Evaluation for Model 1 =====")
y_pred_lr1 = model1.predict(X_test1_scaled)
accuracy_lr1 = accuracy_score(y_test1, y_pred_lr1)
print(f"Accuracy Score for Model 1: {accuracy_lr1:.4f}")
y_proba_lr1 = model1.predict_proba(X_test1_scaled)[:, 1]
auc_lr1 = roc_auc_score(y_test1, y_proba_lr1)
print(f"AUC Score for Model 1: {auc_lr1:.4f}")
print(f"\nClassification Report for Model 1:\n{classification_report(y_test1, y_pred_lr1)}")

joblib.dump(model1, 'model1.pkl')
joblib.dump(scaler1, 'scaler1.pkl')

############################################
# MODEL 2
print(data2.columns)
columns_to_remove_2 = [
    'Ethnicity_Asian', 'Ethnicity_African',
    'Ethnicity_European', 'Ethnicity_middle_eastern'
]
data2_clean = data2.drop(columns=columns_to_remove_2)
print(data2_clean.columns)

print("\n===== Training with Data2 (XGBoost) =====")
X2 = data2_clean.drop(columns=['Diabetes'])
y2 = data2_clean['Diabetes']

# Save feature order
joblib.dump(list(X2.columns), 'model2_features.pkl')

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)

model2.fit(X_train2, y_train2)

print("\n===== Evaluation for Model 2 =====")
y_pred_xgb = model2.predict(X_test2)
y_proba_xgb = model2.predict_proba(X_test2)[:, 1]
acc_xgb = accuracy_score(y_test2, y_pred_xgb)
print(f"Accuracy Score for Model 2: {acc_xgb:.4f}")
auc_xgb = roc_auc_score(y_test2, y_proba_xgb)
print(f"AUC Score for Model 2: {auc_xgb:.4f}")
print(f"\nClassification Report for Model 2:\n{classification_report(y_test2, y_pred_xgb)}")

joblib.dump(model2, 'model2.pkl')

###################################################
# MODEL 3
print(data3.columns)
columns_to_remove_3 = [
    'Ethnicity_Asian', 'Ethnicity_African', 'Ethnicity_European',
    'Ethnicity_middle_eastern', 'No_Pation', 'ID'
]
data3_clean = data3.drop(columns=columns_to_remove_3)
print(data3_clean.columns)

print("\n===== Training with Data3 (Logistic Regression) =====")
X3 = data3_clean.drop(columns=['Diabetes'])
y3 = data3_clean['Diabetes']

# Save feature order
joblib.dump(list(X3.columns), 'model3_features.pkl')

X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.2, random_state=42, stratify=y3)

scaler3 = StandardScaler()
X_train3_scaled = scaler3.fit_transform(X_train3)
X_test3_scaled = scaler3.transform(X_test3)

model3.fit(X_train3_scaled, y_train3)

print("\n===== Evaluation for Model 3 =====")
y_pred_lr2 = model3.predict(X_test3_scaled)
y_proba_lr2 = model3.predict_proba(X_test3_scaled)[:, 1]
acc_lr2 = accuracy_score(y_test3, y_pred_lr2)
print(f"Accuracy Score for Model 3: {acc_lr2:.4f}")
auc_lr2 = roc_auc_score(y_test3, y_proba_lr2)
print(f"AUC Score for Model 3: {auc_lr2:.4f}")
print(f"\nClassification Report for Model 3:\n{classification_report(y_test3, y_pred_lr2)}")

joblib.dump(model3, 'model3.pkl')
joblib.dump(scaler3, 'scaler3.pkl')
