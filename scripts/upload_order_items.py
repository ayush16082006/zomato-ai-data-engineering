import pandas as pd
from google.cloud import bigquery


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ID = "zomato-ai-data-engineering"
DATASET_ID = "zomato"
TABLE_ID = "order_items"

CSV_FILE = r"C:\Users\shand\Downloads\order_items.csv"

# Number of rows processed at one time
CHUNK_SIZE = 50000


# --------------------------------------------------
# BIGQUERY CLIENT
# --------------------------------------------------

client = bigquery.Client(project=PROJECT_ID)

table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"


# --------------------------------------------------
# LOAD CSV IN CHUNKS
# --------------------------------------------------

first_chunk = True
total_rows = 0

print("===================================")
print("STARTING BIGQUERY UPLOAD")
print("===================================")

print(f"Source file : {CSV_FILE}")
print(f"Destination : {table_ref}")
print(f"Chunk size  : {CHUNK_SIZE}")


try:

    for chunk_number, chunk in enumerate(
        pd.read_csv(CSV_FILE, chunksize=CHUNK_SIZE),
        start=1
    ):

        rows = len(chunk)
        total_rows += rows

        print(
            f"\nUploading chunk {chunk_number} "
            f"({rows} rows)..."
        )

        # --------------------------------------------------
        # BIGQUERY LOAD CONFIGURATION
        # --------------------------------------------------

        job_config = bigquery.LoadJobConfig(
            write_disposition=(
                bigquery.WriteDisposition.WRITE_TRUNCATE
                if first_chunk
                else bigquery.WriteDisposition.WRITE_APPEND
            ),
            autodetect=True
        )

        # --------------------------------------------------
        # UPLOAD CHUNK
        # --------------------------------------------------

        job = client.load_table_from_dataframe(
            chunk,
            table_ref,
            job_config=job_config
        )

        # Wait for BigQuery job to complete
        job.result()

        print(
            f"Chunk {chunk_number} uploaded successfully."
        )

        print(
            f"Total rows uploaded so far: {total_rows}"
        )

        first_chunk = False


    # --------------------------------------------------
    # FINAL CHECK
    # --------------------------------------------------

    table = client.get_table(table_ref)

    print("\n===================================")
    print("UPLOAD COMPLETED SUCCESSFULLY")
    print("===================================")

    print(f"Table       : {table_ref}")
    print(f"Rows        : {table.num_rows}")
    print(f"Columns     : {len(table.schema)}")
    print(f"Rows loaded : {total_rows}")

    print("\nSchema:")
    for field in table.schema:
        print(f"  {field.name} -> {field.field_type}")


except FileNotFoundError:
    print("\nERROR: CSV file was not found.")
    print(f"Check this path:\n{CSV_FILE}")

except Exception as e:
    print("\n===================================")
    print("UPLOAD FAILED")
    print("===================================")
    print(f"Error: {e}")