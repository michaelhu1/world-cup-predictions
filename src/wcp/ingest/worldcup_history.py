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


def _team_name(v) -> str | None:
    """Team can be a bare string (modern schema) or {name, code, ...} (older)."""
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return v.get("name") or v.get("code")
    return None


def _ft_score(m: dict) -> tuple[int | None, int | None]:
    if m.get("score1") is not None or m.get("score2") is not None:
        return m.get("score1"), m.get("score2")
    score = m.get("score") or {}
    ft = score.get("ft")
    if isinstance(ft, list) and len(ft) == 2:
        return ft[0], ft[1]
    return None, None


def _flatten_matches(year: str, doc: dict) -> tuple[list[dict], list[dict]]:
    """Flatten a worldcup.json doc into (matches, goalscorers) row lists.

    The openfootball/worldcup.json format has drifted across editions:
    older docs nested matches under ``rounds[].matches[]``, newer ones use
    a flat top-level ``matches[]`` and embed per-goal scorer info under
    ``goals1`` / ``goals2``. This handles both.
    """
    matches: list[dict] = []
    goals: list[dict] = []

    def _emit_match(m: dict, round_name: str) -> None:
        s1, s2 = _ft_score(m)
        t1 = _team_name(m.get("team1"))
        t2 = _team_name(m.get("team2"))
        date = m.get("date")
        ground = m.get("ground")
        if isinstance(ground, dict):
            stadium = ground.get("name")
            city = ground.get("city")
        else:
            stadium = ground
            city = m.get("city")
        matches.append({
            "edition": int(year),
            "round": round_name or m.get("round"),
            "group": m.get("group"),
            "date": date,
            "team1": t1,
            "team2": t2,
            "score1": s1,
            "score2": s2,
            "city": city,
            "stadium": stadium,
        })
        for side, team in (("home", t1), ("away", t2)):
            for g in m.get(f"goals{1 if side == 'home' else 2}", []) or []:
                goals.append({
                    "edition": int(year),
                    "date": date,
                    "team": team,
                    "opponent": t2 if side == "home" else t1,
                    "side": side,
                    "minute": g.get("minute"),
                    "scorer": g.get("name") or g.get("scorer"),
                    "penalty": bool(g.get("penalty")),
                    "own_goal": bool(g.get("og") or g.get("own_goal")),
                })

    if "matches" in doc and isinstance(doc["matches"], list):
        for m in doc["matches"]:
            _emit_match(m, m.get("round", ""))
    for round_ in doc.get("rounds", []) or []:
        round_name = round_.get("name", "")
        for m in round_.get("matches", []) or []:
            _emit_match(m, round_name)

    return matches, goals


def ingest() -> dict[str, Path]:
    raw = raw_dir(SOURCE)
    match_rows: list[dict] = []
    goal_rows: list[dict] = []
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
        m, g = _flatten_matches(year, doc)
        match_rows.extend(m)
        goal_rows.extend(g)

    out = {
        "matches": processed_path("worldcup_matches.parquet"),
        "goals": processed_path("worldcup_goals.parquet"),
    }
    pd.DataFrame(match_rows).to_parquet(out["matches"], index=False)
    pd.DataFrame(goal_rows).to_parquet(out["goals"], index=False)
    return out


def load_matches() -> pd.DataFrame:
    p = processed_path("worldcup_matches.parquet")
    if not p.exists():
        ingest()
    return pd.read_parquet(p)


def load_goals() -> pd.DataFrame:
    p = processed_path("worldcup_goals.parquet")
    if not p.exists():
        ingest()
    return pd.read_parquet(p)
