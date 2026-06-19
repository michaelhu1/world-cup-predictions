"""Canonical project paths.

A single import point for filesystem layout so ingesters and the simulator
agree on where raw vs processed data lives.
"""
from __future__ import annotations

import os
from pathlib import Path


def _project_root() -> Path:
    env = os.environ.get("WCP_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    # src/wcp/paths.py -> project root
    return Path(__file__).resolve().parents[2]


ROOT: Path = _project_root()
DATA: Path = ROOT / "data"
DATA_RAW: Path = DATA / "raw"
DATA_PROCESSED: Path = DATA / "processed"
DATA_CACHE: Path = DATA / "cache"
DATA_EXTERNAL: Path = DATA / "external"


def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_PROCESSED, DATA_CACHE, DATA_EXTERNAL):
        d.mkdir(parents=True, exist_ok=True)


def raw_dir(source: str) -> Path:
    p = DATA_RAW / source
    p.mkdir(parents=True, exist_ok=True)
    return p


def processed_path(name: str) -> Path:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    return DATA_PROCESSED / name
