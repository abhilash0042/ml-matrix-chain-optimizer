import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_squared_error, r2_score)

# ✅ NEW: Load data from JSON
from data.dataloader import load_data

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
X, y = load_data()

print("Shape :", X.shape)

# ─────────────────────────────────────────
# 2. TRAIN TEST SPLIT
# ─────────────────────────────────────────

# 👉 For regression (use y directly if numeric)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────
# 3A. REGRESSION
# ─────────────────────────────────────────
print("\n" + "="*50)
print("  PART A: REGRESSION")
print("="*50)

xgb_reg = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='rmse',
    random_state=42,
    n_jobs=-1
)

xgb_reg.fit(
    X_train_r, y_train_r,
    eval_set=[(X_test_r, y_test_r)],
    verbose=50
)

y_pred_r = xgb_reg.predict(X_test_r)

rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
r2   = r2_score(y_test_r, y_pred_r)

print(f"\nRMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# Plot
plt.scatter(y_test_r, y_pred_r, alpha=0.4)
plt.plot([y_test_r.min(), y_test_r.max()],
         [y_test_r.min(), y_test_r.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("XGBoost Regression")
plt.show()

# ─────────────────────────────────────────
# 3B. CLASSIFICATION
# ─────────────────────────────────────────
print("\n" + "="*50)
print("  PART B: CLASSIFICATION")
print("="*50)

# ⚠️ Only if labels are categorical
# If y is already numeric classes → this works directly

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y, test_size=0.2, random_state=42
)

xgb_cls = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)

xgb_cls.fit(
    X_train_c, y_train_c,
    eval_set=[(X_test_c, y_test_c)],
    verbose=50
)

y_pred_c = xgb_cls.predict(X_test_c)

print(f"\nAccuracy : {accuracy_score(y_test_c, y_pred_c):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_c, y_pred_c))

# Confusion Matrix
cm = confusion_matrix(y_test_c, y_pred_c)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()