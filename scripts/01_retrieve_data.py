import requests
import json
import os
import pandas as pd

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT  = "https://api.gdc.cancer.gov/data"
OUTPUT_DIR         = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def query_by_sample_type(sample_type: str, max_files: int = 10):
    """Query GDC for a specific sample type separately."""

    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": ["TCGA-BRCA"]
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "files.data_type",
                    "value": ["Gene Expression Quantification"]
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "files.analysis.workflow_type",
                    "value": ["STAR - Counts"]
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "files.data_format",
                    "value": ["TSV"]
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "cases.samples.sample_type",
                    "value": [sample_type]        # ← filter by type
                }
            }
        ]
    }

    params = {
        "filters": json.dumps(filters),
        "fields":  "file_id,file_name,cases.submitter_id,"
                   "cases.samples.sample_type",
        "format":  "JSON",
        "size":    str(max_files)
    }

    response = requests.get(GDC_FILES_ENDPOINT, params=params)
    response.raise_for_status()
    hits = response.json()["data"]["hits"]
    print(f"  → Found {len(hits)} [{sample_type}] files")
    return hits


def download_files(file_hits):
    """Download files using GDC bulk download endpoint."""
    file_ids = [f["file_id"] for f in file_hits]

    payload  = {"ids": file_ids}
    response = requests.post(
        GDC_DATA_ENDPOINT,
        data    = json.dumps(payload),
        headers = {"Content-Type": "application/json"},
        stream  = True
    )
    response.raise_for_status()

    archive_path = os.path.join(OUTPUT_DIR, "tcga_brca_raw.tar.gz")
    with open(archive_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)

    print(f"Downloaded archive → {archive_path}")
    return archive_path


def save_metadata(file_hits):
    """Save metadata CSV."""
    records = []
    for hit in file_hits:
        sample_type = "Unknown"
        try:
            sample_type = hit["cases"][0]["samples"][0]["sample_type"]
        except (KeyError, IndexError):
            pass
        records.append({
            "file_id":     hit["file_id"],
            "file_name":   hit["file_name"],
            "case_id":     hit["cases"][0]["submitter_id"],
            "sample_type": sample_type
        })

    df = pd.DataFrame(records)
    meta_path = os.path.join(OUTPUT_DIR, "metadata.csv")
    df.to_csv(meta_path, index=False)

    print(f"\nMetadata saved → {meta_path}")
    print(f"   Class distribution:")
    print(df["sample_type"].value_counts().to_string())
    return df


if __name__ == "__main__":
    print("🔍 Querying TCGA-BRCA — fetching BALANCED samples...")

    # ── Fetch equal numbers of Tumor and Normal ──────────────
    tumor_hits  = query_by_sample_type("Primary Tumor",       max_files=15)
    normal_hits = query_by_sample_type("Solid Tissue Normal", max_files=15)

    all_hits = tumor_hits + normal_hits
    print(f"\nTotal files to download: {len(all_hits)}")

    archive  = download_files(all_hits)
    metadata = save_metadata(all_hits)
