"""FIFA World Rankings (men's).

Source: a community-maintained CSV mirror of FIFA's official rankings, kept on
GitHub at cashlo/FIFA-Ranking. The official FIFA download is paginated HTML
without a public API, so the mirror is the path of least resistance.

Override URLs via WCP_FIFA_RANK_URL if the mirror moves.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from ..paths import processed_path, raw_dir
from .base import http_get

SOURCE = "fifa_rankings"
URL = os.environ.get(
    "WCP_FIFA_RANK_URL",
    "https://raw.githubusercontent.com/cashlo/FIFA-Ranking/master/fifa_ranking.csv",
)


def ingest() -> Path:
    raw = raw_dir(SOURCE)
    body = http_get(URL)
    csv_path = raw / "fifa_ranking.csv"
    csv_path.write_bytes(body)

    df = pd.read_csv(csv_path)
    out = processed_path("fifa_rankings.parquet")
    df.to_parquet(out, index=False)
    return out


def load() -> pd.DataFrame:
    p = processed_path("fifa_rankings.parquet")
    if not p.exists():
        ingest()
    return pd.read_parquet(p)
