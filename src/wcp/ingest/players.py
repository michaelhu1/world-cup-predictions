"""Player attribute table.

Source of choice: the SoFIFA / EA Sports FC player-attribute dump that Kaggle
mirrors every release cycle (e.g. "FIFA 23 complete player dataset",
stefanoleone992 series). It's the most-used feature table in academic World Cup
prediction papers (Groll et al. uses video-game attributes as a proxy for
ability).

Kaggle requires auth, so this ingester does NOT pull from Kaggle directly.
Instead, drop the downloaded CSV at:

    data/external/players_22.csv  (or pass --path)

and this module will normalise + emit a parquet at data/processed/players.parquet.

If KAGGLE_USERNAME / KAGGLE_KEY env vars are set, the user can run
``kaggle datasets download stefanoleone992/fifa-22-complete-player-dataset
-p data/external/`` separately; we deliberately don't bake kaggle into the
required deps.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import DATA_EXTERNAL, processed_path

SOURCE = "players"

DEFAULT_FILE_CANDIDATES = [
    DATA_EXTERNAL / "players_23.csv",
    DATA_EXTERNAL / "players_22.csv",
    DATA_EXTERNAL / "players.csv",
]


def _resolve_input(path: str | None) -> Path:
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Player file not found: {p}")
        return p
    for cand in DEFAULT_FILE_CANDIDATES:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "No player CSV found. Place a SoFIFA-style file at "
        f"{DEFAULT_FILE_CANDIDATES[0]} or pass --path explicitly. "
        "Tip: kaggle datasets download stefanoleone992/fifa-23-complete-player-dataset"
    )


KEEP_COLUMNS = [
    "sofifa_id", "short_name", "long_name", "player_positions", "overall",
    "potential", "value_eur", "wage_eur", "age", "dob", "height_cm", "weight_kg",
    "club_name", "league_name", "nationality_name",
    "preferred_foot", "weak_foot", "skill_moves",
    "pace", "shooting", "passing", "dribbling", "defending", "physic",
    "attacking_finishing", "attacking_short_passing", "attacking_crossing",
    "attacking_heading_accuracy", "attacking_volleys", "skill_dribbling",
    "skill_curve", "skill_fk_accuracy", "skill_long_passing", "skill_ball_control",
    "movement_acceleration", "movement_sprint_speed", "movement_agility",
    "movement_reactions", "movement_balance", "power_shot_power",
    "power_jumping", "power_stamina", "power_strength", "power_long_shots",
    "mentality_aggression", "mentality_interceptions", "mentality_positioning",
    "mentality_vision", "mentality_penalties", "mentality_composure",
    "defending_marking_awareness", "defending_standing_tackle",
    "defending_sliding_tackle", "goalkeeping_diving", "goalkeeping_handling",
    "goalkeeping_kicking", "goalkeeping_positioning", "goalkeeping_reflexes",
    "goalkeeping_speed",
]


def ingest(path: str | None = None) -> Path:
    src = _resolve_input(path)
    df = pd.read_csv(src, low_memory=False)
    cols = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df[cols].copy()
    out = processed_path("players.parquet")
    df.to_parquet(out, index=False)
    return out


def load() -> pd.DataFrame:
    p = processed_path("players.parquet")
    if not p.exists():
        ingest()
    return pd.read_parquet(p)
