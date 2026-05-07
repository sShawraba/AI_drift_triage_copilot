import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from preprocess import build_preprocessor
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn
import joblib


df = pd.read_csv("backend/ml_system/ml/data/bank-additional-full.csv", sep=";")

'''clean data'''
# 1. Drop duration (data leakage)
df = df.drop(columns=["duration"])

# 2. Handle pdays
df["pdays_missing"] = (df["pdays"] == 999).astype(int)

# (optional but good)
df.loc[df["pdays"] == 999, "pdays"] = -1

'''target'''
df["y"] = df["y"].map({"yes": 1, "no": 0})

'''split data'''
X = df.drop(columns=["y"])
y = df["y"]

# First split: train (60%) vs temp (40%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)

# Second split: val (20%) vs test (20%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

'''save data'''

os.makedirs("data/processed", exist_ok=True)

X_train.to_csv("data/processed/X_train.csv", index=False)
X_val.to_csv("data/processed/X_val.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)

y_train.to_csv("data/processed/y_train.csv", index=False)
y_val.to_csv("data/processed/y_val.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

'''build preprocessor and model pipeline'''
preprocessor = build_preprocessor(num_cols, cat_cols)
model = LogisticRegression()
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# # train model
# pipeline.fit(X_train, y_train)

# # evaluate model
# val_preds = pipeline.predict(X_val)
# val_probs = pipeline.predict_proba(X_val)[:, 1]

# #print(classification_report(y_val, val_preds))
# print("AUC:", roc_auc_score(y_val, val_probs))
# print("F1:", f1_score(y_val, val_preds))
# print("Recall:", recall_score(y_val, val_preds))

# '''finding the best threshold-highest threshold where recall ≥ 0.75'''
# val_probs = pipeline.predict_proba(X_val)[:, 1] # get probabilities
# precision, recall, thresholds = precision_recall_curve(y_val, val_probs) # compute curve
# # find valid threshold
# best_threshold = 0.5
# for p, r, t in zip(precision, recall, thresholds):
#     if r >= 0.75:
#         best_threshold = t

# print("Selected threshold:", best_threshold)
# #print(classification_report(y_val, val_preds))

# # compute metrics at the selected threshold
# val_preds = (val_probs >= best_threshold).astype(int)

# #ml flow logging
# mlflow.set_experiment("bank-marketing-classifier")

mlflow.set_tracking_uri('http://localhost:5000')

with mlflow.start_run():

    # 1. train model (ONLY ONCE)
    pipeline.fit(X_train, y_train)

    # 2. predictions
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    val_preds_default = pipeline.predict(X_val)

    # 3. metrics (default threshold baseline)
    auc = roc_auc_score(y_val, val_probs)
    f1_default = f1_score(y_val, val_preds_default)

    # 4. threshold tuning (correct version)
    precision, recall, thresholds = precision_recall_curve(y_val, val_probs)

    best_threshold = 0.5
    for p, r, t in zip(precision, recall, thresholds):
        if r >= 0.75:
            best_threshold = t

    # 5. apply tuned threshold
    val_preds = (val_probs >= best_threshold).astype(int)

    recall_final = recall_score(y_val, val_preds)
    f1_final = f1_score(y_val, val_preds)

    # 6. logging
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("threshold", best_threshold)

    mlflow.log_metric("auc", auc)
    mlflow.log_metric("f1_default", f1_default)
    mlflow.log_metric("f1_tuned", f1_final)
    mlflow.log_metric("recall_tuned", recall_final)

    # 7. log model
    mlflow.sklearn.log_model(pipeline, "model")

    # 8. save threshold
    joblib.dump(best_threshold, "threshold.pkl")
    mlflow.log_artifact("threshold.pkl")