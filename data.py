"""
data.py — Dataset download and preprocessing for UJIIndoorLoc.

Download strategy (UCI changed their API in 2024; direct CSV links return 403):
  1. Try the zip archive at archive.ics.uci.edu/static/public/310/ujiindoorloc.zip
  2. On failure, print manual-download instructions and raise RuntimeError.
  3. If CSVs are already present in data_dir, skip download entirely.
"""

import os
import io
import zipfile
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# UCI dataset identifiers
ZIP_URL = "https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip"
TRAIN_FILENAME = "trainingData.csv"
VAL_FILENAME   = "validationData.csv"

# WAP column range
N_WAP = 520
NO_SIGNAL_RAW     = 100.0
NO_SIGNAL_REPLACE = -105.0


def download_data(data_dir: str = "data/") -> None:
    """Download and extract UJIIndoorLoc CSVs if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, TRAIN_FILENAME)
    val_path   = os.path.join(data_dir, VAL_FILENAME)

    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"[data] CSVs already present in '{data_dir}', skipping download.")
        return

    print(f"[data] Downloading dataset from {ZIP_URL} ...")
    response = None
    try:
        response = requests.get(ZIP_URL, timeout=60)
        response.raise_for_status()
    except requests.exceptions.SSLError:
        # UCI's certificate occasionally expires; retry without verification
        print("[data] SSL cert error — retrying with verification disabled (safe for this public dataset).")
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            response = requests.get(ZIP_URL, timeout=60, verify=False)
            response.raise_for_status()
        except requests.RequestException as exc2:
            _print_manual_instructions(data_dir)
            raise RuntimeError(
                f"Failed to download dataset: {exc2}\n"
                "See manual instructions above."
            ) from exc2
    except requests.RequestException as exc:
        _print_manual_instructions(data_dir)
        raise RuntimeError(
            f"Failed to download dataset: {exc}\n"
            "See manual instructions above."
        ) from exc

    # Extract only the two CSVs we need
    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            extracted = []
            for member in zf.namelist():
                basename = os.path.basename(member)
                if basename in (TRAIN_FILENAME, VAL_FILENAME):
                    # Write to data_dir directly, stripping any subdirectory
                    target = os.path.join(data_dir, basename)
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                    extracted.append(basename)
            if TRAIN_FILENAME not in extracted or VAL_FILENAME not in extracted:
                raise RuntimeError(
                    f"Expected {TRAIN_FILENAME} and {VAL_FILENAME} inside zip, "
                    f"but found: {zf.namelist()}"
                )
    except zipfile.BadZipFile as exc:
        _print_manual_instructions(data_dir)
        raise RuntimeError(f"Downloaded file is not a valid zip: {exc}") from exc

    print(f"[data] Extracted '{TRAIN_FILENAME}' and '{VAL_FILENAME}' to '{data_dir}'.")


def load_and_preprocess(data_dir: str = "data/"):
    """
    Load and preprocess UJIIndoorLoc.

    Returns
    -------
    X_train, X_val              : np.ndarray, shape (N, 520), float32
    y_building_train, y_building_val : np.ndarray, int64, values in {0, 1, 2}
    y_floor_train,   y_floor_val    : np.ndarray, int64, values in {0, 1, 2, 3, 4}
    """
    download_data(data_dir)

    train_df = pd.read_csv(os.path.join(data_dir, TRAIN_FILENAME))
    val_df   = pd.read_csv(os.path.join(data_dir, VAL_FILENAME))

    # WAP columns are the first 520 columns
    wap_cols = [f"WAP{i:03d}" for i in range(1, N_WAP + 1)]

    # Replace "no signal" sentinel with minimum detectable dBm value
    train_df[wap_cols] = train_df[wap_cols].replace(NO_SIGNAL_RAW, NO_SIGNAL_REPLACE)
    val_df[wap_cols]   = val_df[wap_cols].replace(NO_SIGNAL_RAW, NO_SIGNAL_REPLACE)

    X_train_raw = train_df[wap_cols].values.astype(np.float32)
    X_val_raw   = val_df[wap_cols].values.astype(np.float32)

    # Fit scaler on training data only
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val   = scaler.transform(X_val_raw).astype(np.float32)

    y_building_train = train_df["BUILDINGID"].values.astype(np.int64)
    y_building_val   = val_df["BUILDINGID"].values.astype(np.int64)
    y_floor_train    = train_df["FLOOR"].values.astype(np.int64)
    y_floor_val      = val_df["FLOOR"].values.astype(np.int64)

    # Print summary
    print(f"[data] X_train: {X_train.shape}  X_val: {X_val.shape}")
    _print_distribution("building train", y_building_train)
    _print_distribution("building val",   y_building_val)
    _print_distribution("floor train",    y_floor_train)
    _print_distribution("floor val",      y_floor_val)

    return X_train, X_val, y_building_train, y_building_val, y_floor_train, y_floor_val


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_distribution(name: str, labels: np.ndarray) -> None:
    unique, counts = np.unique(labels, return_counts=True)
    dist = ", ".join(f"{c}={n}" for c, n in zip(unique, counts))
    print(f"[data] {name} distribution: {dist}  (total {len(labels)})")


def _print_manual_instructions(data_dir: str) -> None:
    print(
        "\n[data] === MANUAL DOWNLOAD INSTRUCTIONS ===\n"
        "UCI's direct download requires a browser session. To download manually:\n"
        "  1. Visit https://archive.ics.uci.edu/dataset/310/ujiindoorloc\n"
        "  2. Click 'Download' to get ujiindoorloc.zip\n"
        f"  3. Extract 'trainingData.csv' and 'validationData.csv' into '{data_dir}'\n"
        "  4. Re-run the script.\n"
    )
