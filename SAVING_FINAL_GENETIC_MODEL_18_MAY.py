
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # imblearn pipeline to handle sampling inside CV
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_auc_score, classification_report


df = pd.read_csv("dataset_labeled_70th.csv")

# Rename columns for cleaner access
df.columns = df.columns.str.replace(" ", "_").str.replace("-", "_").str.replace("/", "_")
from sklearn.model_selection import train_test_split

X = df.drop(columns=['labels'])
y = df['labels']

# Step 1: Split off the test set (10%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.10, stratify=y, random_state=42
)

# Step 2: Split the remaining 90% into train (70%) and validation (20%)
# Since temp size is 90%, train = 70/90 = ~0.78, val = 20/90 = ~0.22
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.22, stratify=y_temp, random_state=42
)

print(f"Train label distribution:\n{y_train.value_counts(normalize=True)}")
print(f"Validation label distribution:\n{y_val.value_counts(normalize=True)}")
print(f"Test label distribution:\n{y_test.value_counts(normalize=True)}")

# Define numeric and categorical features
numeric_features = [
    'CHR_ID', 'CHR_POS', 'SNPS', 'SNP_ID_CURRENT', 'INTERGENIC',
    'RISK_ALLELE_FREQUENCY', 'P_VALUE', 'PVALUE_MLOG', 'OR_or_BETA', 'PRS_scaled'
]
categorical_features = ['Ethnicity_African', 'Ethnicity_Asian', 'Ethnicity_European']

# Preprocessing transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', 'passthrough', categorical_features)  # no scaling for binary features
    ]
)

from sklearn.linear_model import LogisticRegression

pipe_lr = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(solver='liblinear', random_state=42))
])

param_grid_lr = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],  # inverse of regularization strength
    'classifier__penalty': ['l1', 'l2'],      # regularization type
    'classifier__class_weight': [None, 'balanced']
}

search_lr = RandomizedSearchCV(
    pipe_lr,
    param_grid_lr,
    n_iter=10,
    scoring='roc_auc',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

search_lr.fit(X_train, y_train)
print("Best parameters:", search_lr.best_params_)
print("Best CV AUC:", search_lr.best_score_)
print("Best CV AUC:", search_lr.best_score_)

y_val_pred_lr = search_lr.predict(X_val)
y_val_proba_lr = search_lr.predict_proba(X_val)[:, 1]

print("Validation ROC-AUC:", roc_auc_score(y_val, y_val_proba_lr))
print(classification_report(y_val, y_val_pred_lr))


y_test_pred_lr = search_lr.predict(X_test)
y_test_proba_lr = search_lr.predict_proba(X_test)[:, 1]

print("Test ROC-AUC:", roc_auc_score(y_test, y_test_proba_lr))
print(classification_report(y_test, y_test_pred_lr))

###saving model
import joblib
joblib.dump(search_lr.best_estimator_, 'final_genetic_model_18_may.pkl')

