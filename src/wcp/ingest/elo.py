"""World Football Elo ratings (current snapshot).

Source: eloratings.net's machine-readable current-rankings TSV at
``https://www.eloratings.net/World.tsv``. 244 national teams, 31 columns;
the first few are stable and documented at https://www.eloratings.net/about.

There is no public full-history TSV. eloratings.net used to expose
``all_matches.tsv`` but it 404s as of 2026-06; if a fresh mirror appears,
swap it in via the ``WCP_ELO_HISTORY_URL`` env var.
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

# Stable columns in eloratings.net World.tsv. Trailing columns (4-30) hold
# detailed historical match counts — we keep them as ``raw_*`` for future use.
NAMED_COLUMNS = {
    0: "rank",
    1: "rank_overall",
    2: "country_code",
    3: "elo",
    4: "rank_change",
    5: "elo_peak",
    6: "rank_peak",
    7: "elo_low",
    8: "rank_low",
}


def ingest() -> Path:
    raw = raw_dir(SOURCE)
    body = http_get(CURRENT_URL, use_cache=False)  # always refresh current ratings
    tsv = raw / "current.tsv"
    tsv.write_bytes(body)

    df = pd.read_csv(io.BytesIO(body), sep="\t", header=None, low_memory=False)
    df = df.rename(columns=NAMED_COLUMNS)
    df = df.rename(columns={i: f"raw_{i}" for i in df.columns if isinstance(i, int)})

    out = processed_path("elo_current.parquet")
    df.to_parquet(out, index=False)
    return out


def load_current() -> pd.DataFrame:
    p = processed_path("elo_current.parquet")
    if not p.exists():
        ingest()
    return pd.read_parquet(p)
