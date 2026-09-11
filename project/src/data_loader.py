"""
data_loader.py
--------------
Loads the raw NASA C-MAPSS FD001 text files and attaches the correct
column names. This module does NOT rename sensors or fabricate any
domain-specific labels -- the original C-MAPSS sensor IDs (s1..s21)
are preserved everywhere in the codebase.

SENSOR_INFO below is only a *reference/description* dictionary used to
show the technical C-MAPSS name and description next to the original
sensor ID in the dashboard (e.g. "s3 -- T30 -- Total temperature at
HPC outlet"). This is standard, published C-MAPSS documentation and
is NOT a claim that this is UAV piston-engine telemetry.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Original C-MAPSS column layout: unit, cycle, 3 operating settings, s1..s21
COLUMN_NAMES = (
    ["unit", "cycle", "setting1", "setting2", "setting3"]
    + [f"s{i}" for i in range(1, 22)]
)

# Reference info for the 7 sensors selected for this prototype.
# Names/descriptions are the standard published C-MAPSS sensor descriptions.
SENSOR_INFO = {
    "s3":  {"name": "T30",  "desc": "Total temperature at HPC outlet"},
    "s4":  {"name": "T50",  "desc": "Total temperature at LPT outlet"},
    "s11": {"name": "Ps30", "desc": "Static pressure at HPC outlet"},
    "s12": {"name": "Phi",  "desc": "Ratio of fuel flow to Ps30"},
    "s15": {"name": "BPR",  "desc": "Bypass ratio"},
    "s20": {"name": "W31",  "desc": "HPT coolant bleed"},
    "s21": {"name": "W32",  "desc": "LPT coolant bleed"},
}

SELECTED_SENSORS = list(SENSOR_INFO.keys())


def _read_whitespace_file(path: Path, expected_cols: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required data file not found: {path}. "
            f"Please place the NASA C-MAPSS FD001 files inside the 'data/' folder."
        )
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # C-MAPSS raw txt files sometimes have trailing blank columns from
    # trailing whitespace -- drop fully-empty columns defensively.
    df = df.dropna(axis=1, how="all")
    if df.shape[1] > expected_cols:
        df = df.iloc[:, :expected_cols]
    if df.shape[1] < expected_cols:
        raise ValueError(
            f"File {path} has {df.shape[1]} columns, expected {expected_cols}. "
            "Please verify this is an unmodified NASA C-MAPSS FD001 file."
        )
    return df


def load_raw_file(path) -> pd.DataFrame:
    """Load train_FD001.txt / test_FD001.txt and assign column names."""
    path = Path(path)
    df = _read_whitespace_file(path, len(COLUMN_NAMES))
    df.columns = COLUMN_NAMES
    # Basic sanity / type cleanup
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    df = df.dropna(subset=SELECTED_SENSORS)
    return df.reset_index(drop=True)


def load_rul_file(path) -> pd.DataFrame:
    """Load RUL_FD001.txt -- one true RUL value per test engine, in order."""
    path = Path(path)
    df = _read_whitespace_file(path, 1)
    df.columns = ["RUL"]
    df["unit"] = np.arange(1, len(df) + 1)
    return df[["unit", "RUL"]]


def load_train(data_dir="data") -> pd.DataFrame:
    return load_raw_file(Path(data_dir) / "train_FD001.txt")


def load_test(data_dir="data") -> pd.DataFrame:
    return load_raw_file(Path(data_dir) / "test_FD001.txt")


def load_test_rul(data_dir="data") -> pd.DataFrame:
    return load_rul_file(Path(data_dir) / "RUL_FD001.txt")


def data_files_present(data_dir="data") -> dict:
    """Check which of the 3 required files exist, without raising."""
    data_dir = Path(data_dir)
    files = {
        "train_FD001.txt": (data_dir / "train_FD001.txt").exists(),
        "test_FD001.txt": (data_dir / "test_FD001.txt").exists(),
        "RUL_FD001.txt": (data_dir / "RUL_FD001.txt").exists(),
    }
    return files
