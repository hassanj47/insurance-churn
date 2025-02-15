import os
import pandas as pd
import boto3
# import joblib

# AWS S3 Configurations
S3_BUCKET = "model-registry-refocus"
S3_MODELS_PATH = "models/"  # S3 folder for model storage

# Read benchmarks file
benchmark_file = "../reports/benchmarks.csv"
benchmarks_df = pd.read_csv(benchmark_file)

# Get the best model details
best_model_row = benchmarks_df.sort_values(by="F1 Score", ascending=False).iloc[-1]
best_model_name = best_model_row["Model"]
best_model_file = best_model_name.lower().replace(" ", "_") + "_model.pkl"
best_model_path = os.path.join("../data/outputs/", best_model_file)

print(f"Deploying Best Model: {best_model_name}")

# Upload model to S3
s3_client = boto3.client("s3")
s3_key = f"{S3_MODELS_PATH}{best_model_file}"

s3_client.upload_file(best_model_path, S3_BUCKET, s3_key)
print(f"Model uploaded to S3: s3://{S3_BUCKET}/{s3_key}")
