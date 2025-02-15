#!/usr/bin/env python
# coding: utf-8

import os
import boto3
import pickle
import pandas as pd
import numpy as np
import re
from io import StringIO
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ---- AWS CONFIGURATIONS ----
S3_BUCKET = "refocus-storage"
RAW_DATA_PREFIX = "insurance-data-raw/"
PROCESSED_DATA_PREFIX = "insurance-data-processed/"
MODEL_REGISTRY_PATH = "model-registry/"
BENCHMARKS_PATH = "benchmarks/benchmarks.csv"
BEST_MODEL_PATH = f"{MODEL_REGISTRY_PATH}best_model.pkl"

s3 = boto3.client("s3")

def get_latest_file_from_s3(prefix):
    """Retrieve the latest file based on timestamp in the filename from an S3 folder."""
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)

    # Extract only valid filenames that match the expected pattern dynamically using prefix
    pattern = rf"{re.escape(prefix)}insurance_data_\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}}\.csv"
    
    files = [obj['Key'] for obj in response.get('Contents', []) if re.search(pattern, obj['Key'])]

    if not files:
        raise FileNotFoundError(f"No valid timestamped files found in S3 under {prefix}")

    # Sort files by timestamp (most recent first)
    files.sort(reverse=True, key=lambda x: re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", x).group(1))

    # Get the latest file and extract its timestamp
    latest_file = files[0]
    timestamp_match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", latest_file)
    
    if not timestamp_match:
        raise ValueError(f"Timestamp extraction failed for {latest_file}")

    return latest_file, timestamp_match.group(1)


# ---- STEP 1: LOAD & PREPROCESS DATA ----
latest_file, timestamp = get_latest_file_from_s3(RAW_DATA_PREFIX)
print(f"Loading latest data file: {latest_file}")

obj = s3.get_object(Bucket=S3_BUCKET, Key=latest_file)
df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))

# Handle missing values
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert categorical features
df = pd.get_dummies(df, columns=["policy_type"], drop_first=True)

# Normalize numerical features
scaler = MinMaxScaler()
df[["age", "annual_premium", "claims_count"]] = scaler.fit_transform(df[["age", "annual_premium", "claims_count"]])

# Save processed data to S3 with the same timestamp
processed_data_path = f"{PROCESSED_DATA_PREFIX}insurance_data_{timestamp}_processed.csv"
csv_buffer = StringIO()
df.to_csv(csv_buffer, index=False)
s3.put_object(Bucket=S3_BUCKET, Key=processed_data_path, Body=csv_buffer.getvalue())

print("Data preprocessing complete!")

# ---- STEP 2: TRAINING MODELS ----
X = df.drop(columns=["churn", "customer_id"])
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define hyperparameter grids
param_grids = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["liblinear", "lbfgs"]
    },
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "xgboost": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 6, 9],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }
}


models = {
    "Logistic_Regression": LogisticRegression(max_iter=500, random_state=42),
    "Random_Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
}

best_models = {}
model_performance = []

for model_name, model in models.items():
    print(f"Training {model_name}...")
    # model.fit(X_train_resampled, y_train_resampled)
    grid_search = GridSearchCV(model, param_grids[model_name.lower().replace(" ", "_")], cv=StratifiedKFold(n_splits=5), scoring="f1", n_jobs=-1)
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    best_model = grid_search.best_estimator_
    best_models[model_name] = best_model
    
    
    # y_pred = model.predict(X_test)
    # y_pred_proba = model.predict_proba(X_test)[:, 1]
        # Make predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
   
    metrics = {
        "Timestamp": timestamp,
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_pred_proba),
    }

    model_performance.append(metrics)

performance_df = pd.DataFrame(model_performance)

# ---- STEP 3: COMPARE WITH EXISTING BEST MODEL ----
try:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=BENCHMARKS_PATH)
    current_benchmarks_df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))
except:
    current_benchmarks_df = pd.DataFrame()

updated_benchmarks_df = pd.concat([current_benchmarks_df, performance_df], ignore_index=True)

benchmarks_buffer = StringIO()
updated_benchmarks_df.to_csv(benchmarks_buffer, index=False)
s3.put_object(Bucket=S3_BUCKET, Key=BENCHMARKS_PATH, Body=benchmarks_buffer.getvalue())

best_new_model = performance_df.sort_values(by="F1 Score", ascending=False).iloc[0]

if not current_benchmarks_df.empty:
    latest_benchmark = current_benchmarks_df.sort_values(by="Timestamp", ascending=False).iloc[0]
else:
    latest_benchmark = None

if latest_benchmark is None or best_new_model["F1 Score"] > latest_benchmark["F1 Score"]:
    best_model_name = best_new_model["Model"]
    #best_model_file = f"{MODEL_REGISTRY_PATH}insurance_data_{timestamp}_{best_model_name.lower().replace(' ', '_')}.pkl"
    best_model_file = f"{MODEL_REGISTRY_PATH}best_model.pkl"

    #best_model_file = 'best_model.pkl'
    best_model = best_models[best_model_name]  # Get the trained model
    model_buffer = pickle.dumps(best_model)
    s3.put_object(Bucket=S3_BUCKET, Key=best_model_file, Body=model_buffer)

    print(f"New best model saved: {best_model_file}")

else:
    best_model_name = best_new_model["Model"]
    best_model_file = f"{MODEL_REGISTRY_PATH}{timestamp}_{best_model_name.lower().replace(' ', '_')}.pkl"
    best_model = best_models[best_model_name]  # Get the trained model
    model_buffer = pickle.dumps(best_model)
    s3.put_object(Bucket=S3_BUCKET, Key=best_model_file, Body=model_buffer)
    
    print("No model update needed. Current best model remains.")

print("Training, evaluation, and deployment process completed!")
