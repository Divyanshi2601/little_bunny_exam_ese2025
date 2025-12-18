# little_bunny_exam_ese2025
Ese
# MultiColumn (0.6 log loss-Nitish)

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, log_loss

# ======================
# LOAD DATA
# ======================
train = pd.read_csv("/kaggle/input/mle-ese-mock/train (5).csv")
test  = pd.read_csv("/kaggle/input/mle-ese-mock/test (4).csv")

# ======================
# TARGET CLEANING
# ======================
y = train["quality_grade"].replace(r'^\s*$', np.nan, regex=True)
mask = y.notna()

X = train.loc[mask].drop(columns=["quality_grade"])
y = y.loc[mask]

# ======================
# DROP ID
# ======================
test_id = test["id"]
X = X.drop(columns=["id"], errors="ignore")
test = test.drop(columns=["id"], errors="ignore")

# ======================
# HANDLE MISSING VALUES
# ======================
X = X.fillna(X.mean(numeric_only=True)).fillna(X.mode().iloc[0])
test = test.fillna(test.mean(numeric_only=True)).fillna(test.mode().iloc[0])

# ❌ OUTLIER HANDLING REMOVED (RF doesn't need it)

# ======================
# FEATURE TYPES
# ======================
cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(include="number").columns

# ======================
# PREPROCESSOR
# ======================
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

# ======================
# TRAIN / VALIDATION SPLIT
# ======================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================
# PIPELINE (AGGRESSIVE RF)
# ======================
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=800,        # 🔥 more trees
        max_depth=20,            # 🔥 deeper trees
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight=None,       # 🔥 accuracy boost
        random_state=42
    ))
])

# ======================
# GRID SEARCH (ACCURACY-FOCUSED)
# ======================
param_grid = {
    "classifier__n_estimators": [500, 800],
    "classifier__max_depth": [15, 20, None]
}

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=3,
    scoring="accuracy",   # 🔥 focus on accuracy
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

print("Best Params:", grid.best_params_)

# ======================
# VALIDATION METRICS
# ======================
y_val_pred  = best_model.predict(X_val)
y_val_proba = best_model.predict_proba(X_val)

print("\nValidation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation ROC-AUC:",
      roc_auc_score(y_val, y_val_proba, multi_class="ovr"))

print("Validation Log Loss:",
      log_loss(
          y_val,
          y_val_proba,
          labels=best_model.named_steps["classifier"].classes_
      ))

print("\nClassification Report:\n")
print(classification_report(y_val, y_val_pred))

# ======================
# TRAIN ON FULL DATA
# ======================
best_model.fit(X, y)

# ======================
# TEST PREDICTIONS
# ======================
test_proba = best_model.predict_proba(test)

# ======================
# SUBMISSION
# ======================
submission = pd.DataFrame({"id": test_id})

for i, cls in enumerate(best_model.named_steps["classifier"].classes_):
    submission[f"Status_{cls}"] = test_proba[:, i]
submission.to_csv("submis.csv", index=False)
submission.head()
#Multiclass(Samveg-For TargetNull)
