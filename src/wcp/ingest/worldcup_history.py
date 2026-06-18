"""Historical FIFA World Cup tournaments.

Source: openfootball/worldcup — a yaml/csv archive of every men's World Cup
since 1930, including group-stage standings, knockout brackets, scorers, and
hosts. Distributed under a permissive licence on GitHub.

The repo's CSV index lives at https://github.com/openfootball/worldcup. We pull
the per-edition JSON the community has compiled at
openfootball/world-cup.json — a single normalised JSON per tournament — to keep
this ingester one HTTP call per edition.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from ..paths import processed_path, raw_dir
from .base import http_get

SOURCE = "worldcup_history"

EDITIONS = [
    "1930", "1934", "1938", "1950", "1954", "1958", "1962", "1966",
    "1970", "1974", "1978", "1982", "1986", "1990", "1994", "1998",
    "2002", "2006", "2010", "2014", "2018", "2022",
]

URL_TEMPLATE = os.environ.get(
    "WCP_WC_JSON_URL_TEMPLATE",
    "https://raw.githubusercontent.com/openfootball/worldcup.json/master/{year}/worldcup.json",
)


def _download_edition(year: str, dest: Path) -> Path | None:
    url = URL_TEMPLATE.format(year=year)
    try:
        body = http_get(url)
    except Exception:  # noqa: BLE001 — older editions sometimes 404 on this mirror
        return None
    dest.write_bytes(body)
    return dest


def _flatten_matches(year: str, doc: dict) -> list[dict]:
    """Flatten a worldcup.json doc into a list of match dicts."""
    rows: list[dict] = []
    for round_ in doc.get("rounds", []):
        round_name = round_.get("name", "")
        for m in round_.get("matches", []):
            rows.append({
                "edition": int(year),
                "round": round_name,
                "date": m.get("date"),
                "team1": (m.get("team1") or {}).get("name"),
                "team2": (m.get("team2") or {}).get("name"),
                "score1": (m.get("score1") if m.get("score1") is not None
                           else (m.get("score") or {}).get("ft", [None, None])[0]),
                "score2": (m.get("score2") if m.get("score2") is not None
                           else (m.get("score") or {}).get("ft", [None, None])[1]),
                "city": m.get("city"),
                "stadium": m.get("stadium"),
            })
    return rows


def ingest() -> Path:
    raw = raw_dir(SOURCE)
    all_rows: list[dict] = []
    for year in EDITIONS:
        ed_dir = raw / year
        ed_dir.mkdir(parents=True, exist_ok=True)
        f = ed_dir / "worldcup.json"
        downloaded = _download_edition(year, f)
        if not downloaded:
            continue
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        all_rows.extend(_flatten_matches(year, doc))

    df = pd.DataFrame(all_rows)
    out = processed_path("worldcup_matches.parquet")
    df.to_parquet(out, index=False)
    return out


def load_matches() -> pd.DataFrame:
    p = processed_path("worldcup_matches.parquet")
    if not p.exists():
        ingest()
    return pd.read_parquet(p)
