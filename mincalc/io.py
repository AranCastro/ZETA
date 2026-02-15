from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def read_uploaded_table(filename: str, data: bytes) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}. Use CSV or Excel.")

    buffer = BytesIO(data)
    if suffix == ".csv":
        return pd.read_csv(buffer)
    return pd.read_excel(buffer)
