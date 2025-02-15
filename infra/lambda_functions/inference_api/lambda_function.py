import json
import boto3
import pickle
import os
import numpy as np

# AWS S3 Configurations
# S3_BUCKET = "model-registry-refocus"
# S3_MODELS_PATH = "models/logistic_regression_model.pkl"
# LOCAL_MODEL_PATH = "/tmp/model.pkl"

S3_BUCKET = "refocus-storage"
S3_MODELS_PATH = "model-registry/best_model.pkl"
LOCAL_MODEL_PATH = "/tmp/model.pkl"

def lambda_handler(event, context):
    print("Lambda handler invoked...")
    

    try:
        # Download model from S3
        if not os.path.exists(LOCAL_MODEL_PATH):
            print("Downloading model from S3...")
            s3_client = boto3.client("s3")
            s3_client.download_file(S3_BUCKET, S3_MODELS_PATH, LOCAL_MODEL_PATH)
            print("Model downloaded successfully.")
    except Exception as e:
        print(f"S3 Access Failed: {e}")
    
    try:
        # Load model
        print("Loading model using pickle...")
        with open(LOCAL_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully.")

    except Exception as e:
        print(f"Error loading model: {e}")

    try:
        # Log event input
        print(f"Received event: {event}")

        # Check if request comes from API Gateway (body as string)
        if "body" in event:
            input_data = json.loads(event["body"])  # Parse JSON string
        else:
            input_data = event  # Direct invocation (from AWS Lambda console)

        print(f"Parsed input data: {input_data}")

        # Validate input
        if "features" not in input_data:
            print("Error: No 'features' in input data")
            return {"statusCode": 400, "body": json.dumps({"error": "Invalid request. 'features' key missing."})}

        features = np.array(input_data["features"]).reshape(1, -1)
        print(f"Features converted to numpy array: {features}")

        # Make prediction
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)[:, 1].tolist()
        print(f"Prediction: {prediction}, Probability: {prediction_proba}")

        return {
            "statusCode": 200,
            "body": json.dumps({"prediction": int(prediction[0]), "probability": prediction_proba[0]})
        }

    except Exception as e:
        print(f"Exception: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }