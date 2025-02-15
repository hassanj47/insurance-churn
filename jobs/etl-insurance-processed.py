import sys
import pandas as pd
import boto3
import re
from datetime import datetime
from awsglue.utils import getResolvedOptions
from sqlalchemy import create_engine

def get_latest_s3_file(s3_bucket, s3_prefix):
    """Finds the latest CSV file in the given S3 folder based on timestamp in the filename."""
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)

    # Extract filenames matching the expected pattern
    pattern = re.compile(r"insurance_data_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.csv")
    files = [
        obj["Key"] for obj in response.get("Contents", [])
        if pattern.search(obj["Key"])
    ]

    if not files:
        raise ValueError("No valid CSV files found in S3 folder!")

    # Extract timestamps and sort files by datetime
    latest_file = max(files, key=lambda x: datetime.strptime(pattern.search(x).group(1), "%Y-%m-%d_%H-%M-%S"))
    print(f"Latest file selected: {latest_file}")

    return latest_file

def extract_processed_data_from_s3(s3_path):
    """Extract: Reads the latest CSV from an S3 folder into a Pandas DataFrame."""
    s3_bucket, s3_prefix = s3_path.replace("s3://", "").split("/", 1)

    # Get latest file from S3
    latest_file_key = get_latest_s3_file(s3_bucket, s3_prefix)
    file_url = f"s3://{s3_bucket}/{latest_file_key}"

    print("Extracting processed data from:", file_url)
    df = pd.read_csv(file_url)
    print("Successfully extracted CSV with", len(df), "rows and", len(df.columns), "columns")
    return df

def transform_processed_data(df):
    """Transform: Renames all columns except 'customer_id' with '_proc' suffix and adds a timestamp column."""
    df = df.rename(columns={col: f"{col}_proc" for col in df.columns if col != "customer_id"})  # Rename columns except customer_id
    df["timestamp_proc"] = datetime.now()  # Add current timestamp
    print("Transformation applied: Renamed columns (except 'customer_id') and added 'timestamp_proc'")
    return df

def load_processed_data_to_rds(df, host, port, database, username, password, table_name):
    """Load: Writes a DataFrame to MySQL RDS using SQLAlchemy."""
    db_url = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
    engine = create_engine(db_url)

    try:
        with engine.connect() as connection:
            tables_result = connection.execute("SHOW TABLES;")
            tables = [row[0] for row in tables_result.fetchall()]
            print("Available tables in", database, ":", tables)

        df.to_sql(table_name, con=engine, if_exists="append", index=False)
        print("Successfully loaded", len(df), "rows into", table_name)

    except Exception as e:
        print("Error loading data to database:", e)

    finally:
        engine.dispose()

if __name__ == "__main__":
    required_args = ["host", "database", "username", "password", "s3_path", "table"]
    optional_args = {"port": "3306"}

    args = getResolvedOptions(sys.argv, required_args + list(optional_args.keys()))

    host = args["host"]
    database = args["database"]
    username = args["username"]
    password = args["password"]
    s3_path = args["s3_path"]
    table_name = args["table"]
    port = args.get("port", optional_args["port"])

    # Extract: Read processed data from S3
    df = extract_processed_data_from_s3(s3_path)

    # Transform: Rename columns (except 'customer_id') and add timestamp column
    df = transform_processed_data(df)

    # Load: Write transformed data to RDS
    load_processed_data_to_rds(df, host, port, database, username, password, table_name)
