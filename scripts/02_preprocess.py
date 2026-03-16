import os
import glob
import tarfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ─── Step 1: Extract archive ──────────────────────────────────
def extract_archive():
    archive = os.path.join(RAW_DIR, "tcga_brca_raw.tar.gz")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(RAW_DIR)
    print("Archive extracted")


# ─── Step 2: Load & merge expression files ────────────────────
def load_expression_files():
    tsv_files = glob.glob(
        os.path.join(RAW_DIR, "**/*.tsv"), recursive=True
    )
    print(f"Found {len(tsv_files)} TSV files")

    dfs = []
    for f in tsv_files:
        try:
            df = pd.read_csv(f, sep="\t", comment="#",
                             index_col=0, header=0)
            if "unstranded" in df.columns:
                col_name = os.path.basename(f)
                df = df[["unstranded"]].rename(
                    columns={"unstranded": col_name}
                )
                dfs.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if not dfs:
        raise RuntimeError("No valid TSV files found after extraction!")

    merged = pd.concat(dfs, axis=1)
    print(f"Merged matrix shape: {merged.shape}  (genes × samples)")
    return merged.T   # → (samples × genes)


# ─── Step 3: Load labels ──────────────────────────────────────
def load_labels(expression_df):
    meta = pd.read_csv(os.path.join(RAW_DIR, "metadata.csv"))

    label_map = {
        "Primary Tumor":       1,
        "Solid Tissue Normal": 0,
        "Metastatic":          1
    }
    meta["label"] = meta["sample_type"].map(label_map)
    meta = meta.dropna(subset=["label"])

    common = expression_df.index.intersection(meta["file_name"])
    X = expression_df.loc[common]
    y = meta.set_index("file_name").loc[common, "label"].astype(int)

    print(f"\nLabel distribution after alignment:")
    print(f"   Tumor  (1): {sum(y == 1)}")
    print(f"   Normal (0): {sum(y == 0)}")
    print(f"   Total     : {len(y)}")
    return X, y


# ─── Step 4: Normalize & filter ──────────────────────────────
def preprocess_features(X):
    # Log1p normalize
    X_log = np.log1p(X.values.astype(float))

    # Variance filter
    selector  = VarianceThreshold(threshold=0.1)
    X_filtered = selector.fit_transform(X_log)
    genes_kept = X.columns[selector.get_support()]
    print(f"Genes after variance filter: {X_filtered.shape[1]}")

    # Standard scale
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)

    return X_scaled, genes_kept, scaler, selector


# ─── Step 5: Smart split ──────────────────────────────────────
def save_processed(X_scaled, y, genes_kept):

    # Save full feature matrix
    pd.DataFrame(X_scaled, columns=genes_kept).to_csv(
        f"{PROCESSED_DIR}/X_features.csv", index=False
    )
    y.to_csv(f"{PROCESSED_DIR}/y_labels.csv", index=False)

    class_counts  = pd.Series(y).value_counts()
    min_count     = class_counts.min()
    n_total       = len(y)

    print(f"\n🔎 Checking split safety...")
    print(f"   Min class count : {min_count}")
    print(f"   Total samples   : {n_total}")

    # ── Case 1: Enough samples → normal stratified split ─────
    if min_count >= 4:
        test_size    = 0.2
        use_stratify = y
        print("Strategy: Stratified 80/20 split")

    # ── Case 2: Very few minority → smaller test set ─────────
    elif min_count == 3:
        test_size    = 0.15
        use_stratify = y
        print("Strategy: Stratified 85/15 split (small minority class)")

    # ── Case 3: Only 2 minority samples → no stratify ────────
    elif min_count == 2:
        test_size    = 0.2
        use_stratify = None
        print("Strategy: Random 80/20 split (minority class too small)")

    # ── Case 4: Only 1 minority sample → manual split ────────
    else:
        print("Strategy: Manual split (only 1 minority sample)")
        print("   Placing the 1 Normal sample in training set.")

        minority_idx = y[y == 0].index
        majority_idx = y[y == 1].index

        # Put the single normal in train, split majority 80/20
        maj_train_idx, maj_test_idx = train_test_split(
            majority_idx,
            test_size    = 0.2,
            random_state = 42
        )

        train_idx = maj_train_idx.tolist() + minority_idx.tolist()
        test_idx  = maj_test_idx.tolist()

        # Convert y index to positional for numpy
        all_idx   = list(y.index)
        train_pos = [all_idx.index(i) for i in train_idx]
        test_pos  = [all_idx.index(i) for i in test_idx]

        X_train = X_scaled[train_pos]
        X_test  = X_scaled[test_pos]
        y_train = y.iloc[train_pos].values
        y_test  = y.iloc[test_pos].values

        np.save(f"{PROCESSED_DIR}/X_train.npy", X_train)
        np.save(f"{PROCESSED_DIR}/X_test.npy",  X_test)
        np.save(f"{PROCESSED_DIR}/y_train.npy", y_train)
        np.save(f"{PROCESSED_DIR}/y_test.npy",  y_test)

        print(f"\nTrain: {X_train.shape}  "
              f"(Tumor={sum(y_train==1)}, Normal={sum(y_train==0)})")
        print(f"Test : {X_test.shape}   "
              f"(Tumor={sum(y_test==1)},  Normal={sum(y_test==0)})")
        return   # ← exit early, already saved

    # ── Normal sklearn split (Cases 1-3) ─────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size    = test_size,
        random_state = 42,
        stratify     = use_stratify
    )

    np.save(f"{PROCESSED_DIR}/X_train.npy", X_train)
    np.save(f"{PROCESSED_DIR}/X_test.npy",  X_test)
    np.save(f"{PROCESSED_DIR}/y_train.npy", y_train)
    np.save(f"{PROCESSED_DIR}/y_test.npy",  y_test)

    print(f"\nTrain: {X_train.shape}  "
          f"(Tumor={sum(y_train==1)}, Normal={sum(y_train==0)})")
    print(f"Test : {X_test.shape}   "
          f"(Tumor={sum(y_test==1)},  Normal={sum(y_test==0)})")


# ─── Main ──────
if __name__ == "__main__":
    extract_archive()
    expr_matrix              = load_expression_files()
    X, y                     = load_labels(expr_matrix)
    X_scaled, genes, sc, sel = preprocess_features(X)
    save_processed(X_scaled, y, genes)
    print("\nPreprocessing complete!")
