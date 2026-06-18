# world-cup-predictions

A 2026 FIFA World Cup match-outcome predictor. Two halves:

1. **Ingestion (`wcp.ingest.*`)** — pull historical matches, ratings, player attributes, market values, and World Cup history from public datasets and light scrapers into a single `data/` lake.
2. **Simulation (`wcp.sim.*`)** — given two teams (and their squads), simulate a match: scoreline, scorers, and downstream tournament probabilities. The simulator family is pluggable; a Dixon-Coles + scorer-sampling baseline is included, with hooks for learned/RL-based simulators added once the research workflow lands.

## Quick start

```bash
# create venv (Python 3.10+ required)
python -m venv .venv
. .venv/Scripts/activate    # on Windows bash; use .venv/bin/activate on POSIX

pip install -e ".[sim,dev]"

# pull every dataset that has a free, no-key path
wcp ingest all

# pull a single source
wcp ingest results       # martj42 international results 1872-present
wcp ingest elo           # World Football Elo ratings
wcp ingest fifa-rank     # FIFA World Rankings
wcp ingest worldcup      # openfootball/world-cup all editions
wcp ingest players       # player attributes (Kaggle SoFIFA dump path)
wcp ingest transfermarkt # squad market values (light scrape)

# run a baseline match simulation
wcp simulate --home Argentina --away France --n 10000
```

Data lands under `data/raw/<source>/...` (gitignored); processed parquet files under `data/processed/`.

## Layout

```
src/wcp/
  ingest/          # one module per data source
    results.py
    elo.py
    fifa_rankings.py
    worldcup_history.py
    players.py
    transfermarkt.py
    base.py        # shared HTTP session, caching, paths
  sim/
    base.py        # MatchSimulator interface + result types
    dixon_coles.py # baseline Poisson-family scoreline + scorer sampling
  cli.py           # `wcp` entrypoint
  paths.py         # canonical data dir resolution
```

## Status

- [x] Project skeleton, packaging, CLI scaffold
- [x] Ingestion modules for: international results, Elo, FIFA rankings, World Cup history
- [x] Player attributes ingester (Kaggle SoFIFA path; configurable file)
- [x] Transfermarkt light squad-value scraper
- [ ] Simulation: Dixon-Coles + scorer sampling baseline (stub in place)
- [ ] Learned simulator (RL / event-level) — pending research workflow output

## Notes

- Scrapers throttle and respect `robots.txt`; cached HTTP responses live under `data/cache/`.
