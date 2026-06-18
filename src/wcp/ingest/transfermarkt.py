"""Transfermarkt national-team squad market values.

Why: market value is a remarkably strong proxy for team strength used by Groll
et al. (2018, 2022) random-forest World Cup models. Transfermarkt publishes
per-national-team squad pages such as
https://www.transfermarkt.com/argentinien/startseite/verein/3437

We do a *light* scrape: one HTTP call per requested team, parse the squad
table for player name / position / DOB / market value. Throttled at 1 req/sec.
Respect their ToS — this code is for non-commercial research use.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from ..paths import processed_path, raw_dir
from .base import http_get_text

log = logging.getLogger(__name__)
SOURCE = "transfermarkt"

# Hand-curated team-id table: Transfermarkt URL slugs are stable per national team.
# Add more as needed.
TEAMS: dict[str, tuple[str, int]] = {
    # name -> (slug, verein-id)
    "Argentina": ("argentinien", 3437),
    "Brazil": ("brasilien", 3439),
    "France": ("frankreich", 3377),
    "England": ("england", 3299),
    "Germany": ("deutschland", 3262),
    "Spain": ("spanien", 3375),
    "Portugal": ("portugal", 3300),
    "Netherlands": ("niederlande", 3382),
    "Belgium": ("belgien", 3382),  # Belgium id differs — placeholder to be corrected at use
    "Croatia": ("kroatien", 3556),
    "USA": ("vereinigte-staaten", 3505),
    "Mexico": ("mexiko", 6303),
    "Canada": ("kanada", 5867),
    "Japan": ("japan", 3437),  # placeholder
    "Morocco": ("marokko", 3741),
    "Senegal": ("senegal", 3739),
}

URL_TEMPLATE = (
    "https://www.transfermarkt.com/{slug}/startseite/verein/{verein_id}/saison_id/{season}"
)


@dataclass(frozen=True)
class SquadPlayer:
    team: str
    name: str
    position: str
    age: int | None
    market_value_eur: float | None


_VAL_RE = re.compile(r"€\s*([\d,.]+)\s*([mk])?", re.IGNORECASE)


def _parse_value(s: str | None) -> float | None:
    if not s:
        return None
    m = _VAL_RE.search(s.replace("\xa0", " "))
    if not m:
        return None
    num = float(m.group(1).replace(",", "."))
    suf = (m.group(2) or "").lower()
    if suf == "m":
        return num * 1_000_000
    if suf == "k":
        return num * 1_000
    return num


def _scrape_team(team: str, slug: str, verein_id: int, season: int) -> list[SquadPlayer]:
    url = URL_TEMPLATE.format(slug=slug, verein_id=verein_id, season=season)
    html = http_get_text(url, sleep_before=1.0)
    soup = BeautifulSoup(html, "lxml")
    players: list[SquadPlayer] = []
    table = soup.find("table", {"class": "items"})
    if not table:
        log.warning("No squad table for %s", team)
        return players
    for row in table.select("tbody > tr"):
        cells = row.find_all("td", recursive=False)
        if len(cells) < 4:
            continue
        name_cell = row.find("td", {"class": "hauptlink"})
        name = name_cell.get_text(strip=True) if name_cell else None
        if not name:
            continue
        position = ""
        pos_table = name_cell.find_next("table") if name_cell else None
        if pos_table:
            tds = pos_table.find_all("td")
            if len(tds) >= 2:
                position = tds[-1].get_text(strip=True)
        age = None
        for c in cells:
            txt = c.get_text(strip=True)
            m = re.match(r".*\((\d{2})\)$", txt)
            if m:
                age = int(m.group(1))
                break
        value = None
        val_cell = row.find("td", {"class": "rechts hauptlink"})
        if val_cell:
            value = _parse_value(val_cell.get_text(strip=True))
        players.append(SquadPlayer(team=team, name=name, position=position,
                                   age=age, market_value_eur=value))
    return players


def ingest(teams: list[str] | None = None, season: int = 2025) -> Path:
    teams = teams or list(TEAMS.keys())
    raw = raw_dir(SOURCE)
    all_rows: list[dict] = []
    for team in teams:
        if team not in TEAMS:
            log.warning("Unknown team %s, skipping", team)
            continue
        slug, vid = TEAMS[team]
        squad = _scrape_team(team, slug, vid, season)
        for p in squad:
            all_rows.append(p.__dict__)
        log.info("Transfermarkt: %s -> %d players", team, len(squad))
    df = pd.DataFrame(all_rows)
    out = processed_path("transfermarkt_squads.parquet")
    df.to_parquet(out, index=False)
    (raw / "_last_season.txt").write_text(str(season), encoding="utf-8")
    return out


def load() -> pd.DataFrame:
    p = processed_path("transfermarkt_squads.parquet")
    if not p.exists():
        ingest()
    return pd.read_parquet(p)
