"""`wcp` command-line entrypoint.

Two top-level groups:

    wcp ingest <source>     run a single ingester or all of them
    wcp simulate ...        run a baseline match simulation
"""
from __future__ import annotations

import logging

import click

from .paths import ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@click.group()
def main() -> None:
    """world-cup-predictions CLI."""
    ensure_dirs()


# ---------- ingest ----------
@main.group()
def ingest() -> None:
    """Pull a data source into data/processed/*.parquet."""


@ingest.command("results")
def _ingest_results() -> None:
    from .ingest import results
    out = results.ingest()
    for k, p in out.items():
        click.echo(f"{k}: {p}")


@ingest.command("elo")
def _ingest_elo() -> None:
    from .ingest import elo
    p = elo.ingest()
    click.echo(str(p))


@ingest.command("fifa-rank")
def _ingest_fifa_rank() -> None:
    from .ingest import fifa_rankings
    p = fifa_rankings.ingest()
    click.echo(str(p))


@ingest.command("worldcup")
def _ingest_worldcup() -> None:
    from .ingest import worldcup_history
    out = worldcup_history.ingest()
    for k, p in out.items():
        click.echo(f"{k}: {p}")


@ingest.command("players")
@click.option("--path", type=click.Path(exists=True), default=None,
              help="Path to a SoFIFA-style player CSV (Kaggle dump). "
                   "Defaults to data/external/players_*.csv.")
def _ingest_players(path: str | None) -> None:
    from .ingest import players
    p = players.ingest(path=path)
    click.echo(str(p))


@ingest.command("transfermarkt")
@click.option("--team", "teams", multiple=True, default=None,
              help="One or more team names; default = all known teams.")
@click.option("--season", type=int, default=2025)
def _ingest_transfermarkt(teams: tuple[str, ...], season: int) -> None:
    from .ingest import transfermarkt
    p = transfermarkt.ingest(teams=list(teams) if teams else None, season=season)
    click.echo(str(p))


@ingest.command("all")
def _ingest_all() -> None:
    """Run every ingester that has a free, no-key, no-input path."""
    from .ingest import results, elo, fifa_rankings, worldcup_history
    for fn, name in [
        (results.ingest, "results"),
        (elo.ingest, "elo"),
        (fifa_rankings.ingest, "fifa-rank"),
        (worldcup_history.ingest, "worldcup"),
    ]:
        try:
            click.echo(f"== {name} ==")
            fn()
        except Exception as e:  # noqa: BLE001
            click.echo(f"  failed: {e}", err=True)


# ---------- simulate ----------
@main.command()
@click.option("--home", required=True)
@click.option("--away", required=True)
@click.option("--n", type=int, default=10_000)
@click.option("--home-rating", type=float, default=None)
@click.option("--away-rating", type=float, default=None)
@click.option("--seed", type=int, default=None)
def simulate(home: str, away: str, n: int,
             home_rating: float | None, away_rating: float | None,
             seed: int | None) -> None:
    """Baseline Dixon-Coles simulation between two teams."""
    from .sim.base import TeamSpec
    from .sim.dixon_coles import DixonColesSimulator

    sim = DixonColesSimulator(seed=seed)
    h = TeamSpec(name=home, rating=home_rating)
    a = TeamSpec(name=away, rating=away_rating)
    dist = sim.simulate_distribution(h, a, n=n)
    click.echo(f"{home} vs {away}  (n={n})")
    click.echo(f"  P({home} win) = {dist.home_win_prob:.3f}")
    click.echo(f"  P(draw)       = {dist.draw_prob:.3f}")
    click.echo(f"  P({away} win) = {dist.away_win_prob:.3f}")
    click.echo(f"  E[goals]      = {dist.expected_home_goals:.2f} - {dist.expected_away_goals:.2f}")
    click.echo("  Top scorelines:")
    for (h_g, a_g), p in dist.top_scorelines[:5]:
        click.echo(f"    {h_g}-{a_g}: {p:.3f}")


if __name__ == "__main__":
    main()
