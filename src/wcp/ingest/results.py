"""International football results 1872-present.

Source: martj42/international_results — the canonical public dataset for
international men's football matches, used by virtually every academic World Cup
prediction paper since 2016. Distributed as plain CSV on GitHub.

Repo: https://github.com/martj42/international_results
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from ..paths import processed_path, raw_dir
from .base import http_get

SOURCE = "international_results"
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
GOALSCORERS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv"
)
SHOOTOUTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
)


def _download(url: str, dest: Path) -> Path:
    body = http_get(url)
    dest.write_bytes(body)
    return dest


def ingest() -> dict[str, Path]:
    """Download all three CSVs, normalise dtypes, and emit parquet."""
    raw = raw_dir(SOURCE)
    csvs = {
        "results": _download(RESULTS_URL, raw / "results.csv"),
        "goalscorers": _download(GOALSCORERS_URL, raw / "goalscorers.csv"),
        "shootouts": _download(SHOOTOUTS_URL, raw / "shootouts.csv"),
    }

    results = pd.read_csv(csvs["results"], parse_dates=["date"])
    goalscorers = pd.read_csv(csvs["goalscorers"], parse_dates=["date"])
    shootouts = pd.read_csv(csvs["shootouts"], parse_dates=["date"])

    out = {
        "results": processed_path("results.parquet"),
        "goalscorers": processed_path("goalscorers.parquet"),
        "shootouts": processed_path("shootouts.parquet"),
    }
    results.to_parquet(out["results"], index=False)
    goalscorers.to_parquet(out["goalscorers"], index=False)
    shootouts.to_parquet(out["shootouts"], index=False)
    return out


def load_results() -> pd.DataFrame:
    p = processed_path("results.parquet")
    if not p.exists():
        ingest()
    return pd.read_parquet(p)
