import os, io, json, argparse
import pandas as pd
from azure.storage.blob import BlobServiceClient

# Default Azurite connection string (same format as Azure Storage)
DEFAULT_CONN = (
    "DefaultEndpointsProtocol=http;"
    "AccountName=devstoreaccount1;"
    "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/"
    "K1SZFPTOtr/KBHBeksoGMGw==;"
    "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
    "QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;"
    "TableEndpoint=http://127.0.0.1:10002/devstoreaccount1;"
)

def process_nutritional_data_from_azurite(
    container_name="datasets",
    blob_name="All_Diets.csv",
    out_json="simulated_nosql/results.json",
    connection_string=None
):
    # Connect to Azurite Blob Storage (local emulator)
    conn = connection_string or os.environ.get("AZURE_STORAGE_CONNECTION_STRING", DEFAULT_CONN)
    blob_service_client = BlobServiceClient.from_connection_string(conn)

    container_client = blob_service_client.get_container_client(container_name)
    try:
        container_client.create_container()
    except Exception:
        pass  # Container already exists

    blob_client = container_client.get_blob_client(blob_name)

    # Download the CSV data from Azurite
    print(f"📦 Downloading '{blob_name}' from container '{container_name}'...")
    data = blob_client.download_blob().readall()
    df = pd.read_csv(io.BytesIO(data))

    # Normalize possible column variations
    df = df.rename(columns={
        "Protein (g)": "Protein(g)",
        "Carbs (g)": "Carbs(g)",
        "Fat (g)": "Fat(g)"
    })

    # Ensure numeric values and fill missing data
    for c in ["Protein(g)", "Carbs(g)", "Fat(g)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.fillna(df.mean(), inplace=True)

    # Compute average macronutrients per diet type
    avg_macros = (
        df.groupby("Diet_type")[["Protein(g)", "Carbs(g)", "Fat(g)"]]
        .mean()
        .reset_index()
        .sort_values("Diet_type")
    )

    # Save processed results as JSON (simulating NoSQL storage)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(avg_macros.to_dict(orient="records"), f, indent=2)

    print(f"Processed rows: {len(df)}")
    print(f"Diet types: {df['Diet_type'].nunique()}")
    print(f"Results saved to: {out_json}")
    return out_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulated serverless Azure Function using Azurite")
    parser.add_argument("--container", default="datasets")
    parser.add_argument("--blob", default="All_Diets.csv")
    parser.add_argument("--out", default="simulated_nosql/results.json")
    args = parser.parse_args()

    process_nutritional_data_from_azurite(
        container_name=args.container,
        blob_name=args.blob,
        out_json=args.out,
        connection_string=os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    )
