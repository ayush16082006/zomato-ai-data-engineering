import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

ENV_FILE = BASE_DIR / "airflow" / ".env"


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

if not ENV_FILE.exists():
    raise FileNotFoundError(
        f"Environment file not found:\n{ENV_FILE}"
    )

load_dotenv(ENV_FILE)


# ============================================================
# READ QDRANT CREDENTIALS
# ============================================================

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


if not QDRANT_URL:
    raise RuntimeError(
        "QDRANT_URL is missing from airflow/.env"
    )


if not QDRANT_API_KEY:
    raise RuntimeError(
        "QDRANT_API_KEY is missing from airflow/.env"
    )


# ============================================================
# CONNECT TO QDRANT
# ============================================================

print()
print("=" * 60)
print("QDRANT CONNECTION TEST")
print("=" * 60)

print()
print("Cluster: Zomato")
print(f"Endpoint: {QDRANT_URL}")
print()
print("Connecting to Qdrant Cloud...")


client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


# ============================================================
# TEST CONNECTION
# ============================================================

collections = client.get_collections()


# ============================================================
# SUCCESS
# ============================================================

print()
print("=" * 60)
print("QDRANT CONNECTION SUCCESSFUL")
print("=" * 60)

print()
print("Cluster name       : Zomato")
print(
    "Collections found  : "
    f"{len(collections.collections)}"
)

if collections.collections:

    print()
    print("Existing collections:")

    for collection in collections.collections:
        print(
            f"  - {collection.name}"
        )

else:

    print()
    print(
        "No collections found."
    )

    print(
        "This is expected because "
        "we have not created a collection yet."
    )

print()
print("=" * 60)
print("STEP 3 CONNECTION TEST PASSED")
print("=" * 60)