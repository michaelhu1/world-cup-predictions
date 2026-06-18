"""World Football Elo ratings.

Source: eloratings.net publishes its rating history. The simplest free
machine-readable mirror is the per-team Elo CSV at:

    http://api.clubelo.com/  (clubs only — NOT useful for internationals)

For internationals we use the community CSV mirror maintained at
https://github.com/lsv/fifa-worldcup-2018 / https://github.com/martj42 has stale
mirrors; the most active free source is the eloratings.net per-date snapshot.

This ingester pulls the ranking-history JSON exposed by the unofficial
World Football Elo "API" used by community projects:

    https://www.eloratings.net/World.tsv  (current top-N TSV snapshot)

For full history we instead bundle the matches CSV exposed at:

    https://www.eloratings.net/all_matches.tsv

If the upstream changes, override the URL via the ``WCP_ELO_URL`` env var or
swap to one of the mirrored CSVs in data/external/.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd

from ..paths import processed_path, raw_dir
from .base import http_get

SOURCE = "elo"

CURRENT_URL = os.environ.get(
    "WCP_ELO_CURRENT_URL", "https://www.eloratings.net/World.tsv"
)
HISTORY_URL = os.environ.get(
    "WCP_ELO_HISTORY_URL", "https://www.eloratings.net/all_matches.tsv"
)


def _download_tsv(url: str, dest: Path) -> Path:
    body = http_get(url)
    dest.write_bytes(body)
    return dest


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, low_memory=False)


def ingest() -> dict[str, Path]:
    raw = raw_dir(SOURCE)
    cur = _download_tsv(CURRENT_URL, raw / "current.tsv")
    hist = _download_tsv(HISTORY_URL, raw / "all_matches.tsv")

    df_cur = _read_tsv(cur)
    df_hist = _read_tsv(hist)

    out = {
        "current": processed_path("elo_current.parquet"),
        "history": processed_path("elo_history.parquet"),
    }
    df_cur.to_parquet(out["current"], index=False)
    df_hist.to_parquet(out["history"], index=False)
    return out


def load_current() -> pd.DataFrame:
    p = processed_path("elo_current.parquet")
    if not p.exists():
        ingest()
    return pd.read_parquet(p)
