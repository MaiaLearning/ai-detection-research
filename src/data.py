"""Shared PERSUADE 2.0 loading/cleaning, used by every experiment script so
cleaning criteria can't silently drift between experiments (CLAUDE.md:
"Log the exact corpus version and filter criteria used for every run.").
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_FIELDS = [
    "ell_clean", "grade_level", "holistic_essay_score", "word_count",
    "prompt_name", "full_text",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_clean(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    n_total = len(df)

    df["ell_clean"] = df["ell_status"].where(df["ell_status"].isin(["Yes", "No"]))
    mask = df[REQUIRED_FIELDS].notna().all(axis=1)
    clean = df.loc[mask].copy()

    manifest = {
        "n_total_rows": n_total,
        "n_after_filter": len(clean),
        "n_dropped_missing_fields": n_total - len(clean),
        "n_ell": int((clean["ell_clean"] == "Yes").sum()),
        "n_non_ell": int((clean["ell_clean"] == "No").sum()),
    }
    return clean, manifest


def add_standardized_covariates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_word_count"] = np.log1p(df["word_count"])
    for col, z_col in [("log_word_count", "z_logwc"), ("holistic_essay_score", "z_holistic")]:
        mean, std = df[col].mean(), df[col].std(ddof=1)
        df[z_col] = (df[col] - mean) / std
    return df
