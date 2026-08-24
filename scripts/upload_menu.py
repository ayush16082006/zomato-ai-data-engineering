import pandas as pd
from google.cloud import bigquery


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ID = "zomato-ai-data-engineering"
DATASET_ID = "zomato"
TABLE_ID = "menu"

CSV_FILE = r"C:/Users/shand/Downloads/menu.csv"

# Number of rows processed at one time
CHUNK_SIZE = 50000

# Start uploading from this chunk
START_CHUNK = 18


# --------------------------------------------------
# BIGQUERY CLIENT
# --------------------------------------------------

client = bigquery.Client(project=PROJECT_ID)

table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"


# --------------------------------------------------
# LOAD CSV IN CHUNKS
# --------------------------------------------------

total_rows = 0

print("===================================")
print("STARTING BIGQUERY UPLOAD")
print("===================================")

print(f"Source file : {CSV_FILE}")
print(f"Destination : {table_ref}")
print(f"Chunk size  : {CHUNK_SIZE}")
print(f"Starting from chunk : {START_CHUNK}")


try:

    for chunk_number, chunk in enumerate(
        pd.read_csv(CSV_FILE, chunksize=CHUNK_SIZE),
        start=1
    ):

        # --------------------------------------------------
        # SKIP ALREADY UPLOADED CHUNKS
        # --------------------------------------------------

        if chunk_number < START_CHUNK:
            continue


        # --------------------------------------------------
        # CHUNK INFORMATION
        # --------------------------------------------------

        rows = len(chunk)
        total_rows += rows

        print(
            f"\nUploading chunk {chunk_number} "
            f"({rows} rows)..."
        )


        # --------------------------------------------------
        # FIX PRICE COLUMN
        # --------------------------------------------------

        chunk["price"] = pd.to_numeric(
            chunk["price"],
            errors="coerce"
        )


        # --------------------------------------------------
        # BIGQUERY LOAD CONFIGURATION
        # --------------------------------------------------

        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            autodetect=False
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


        # --------------------------------------------------
        # SUCCESS MESSAGE
        # --------------------------------------------------

        print(
            f"Chunk {chunk_number} uploaded successfully."
        )

        print(
            f"Rows uploaded in this chunk : {rows}"
        )

        print(
            f"Rows uploaded in this run   : {total_rows}"
        )


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


    # --------------------------------------------------
    # SCHEMA
    # --------------------------------------------------

    print("\nSchema:")

    for field in table.schema:
        print(
            f"  {field.name} -> {field.field_type}"
        )


except FileNotFoundError:

    print("\n===================================")
    print("ERROR: CSV FILE NOT FOUND")
    print("===================================")

    print(f"Check this path:")
    print(CSV_FILE)


except Exception as e:

    print("\n===================================")
    print("UPLOAD FAILED")
    print("===================================")

    print(f"Error: {e}")