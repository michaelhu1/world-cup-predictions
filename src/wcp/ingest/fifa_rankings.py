"""FIFA World Rankings (men's).

Source: a community-maintained CSV mirror of FIFA's official rankings.

As of 2026-06, the cashlo/FIFA-Ranking mirror is 404 and there is no widely
maintained replacement. Set ``WCP_FIFA_RANK_URL`` to a working mirror to use
this ingester, or skip it — Elo ratings (see ``elo.py``) are the better
strength signal for the predictor anyway.
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
