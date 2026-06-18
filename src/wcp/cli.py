"""`wcp` command-line entrypoint.

Top-level commands:

    wcp ingest <source>     run a single ingester or all of them
    wcp fit dixon-coles     MLE-fit Dixon-Coles strengths on results.parquet
    wcp simulate ...        Dixon-Coles simulation between two teams
"""
from __future__ import annotations

import logging

import click

from .paths import ensure_dirs, processed_path

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


# ---------- fit ----------
DEFAULT_DC_PATH = processed_path("dc_strengths.json")


@main.group()
def fit() -> None:
    """Fit model parameters."""


@fit.command("dixon-coles")
@click.option("--half-life-days", type=float, default=1825.0,
              help="Time-decay half-life in days (default ~5 years).")
@click.option("--min-team-matches", type=int, default=50,
              help="Drop teams with fewer than this many matches.")
@click.option("--out", type=click.Path(), default=str(DEFAULT_DC_PATH))
@click.option("--max-iter", type=int, default=500)
def _fit_dc(half_life_days: float, min_team_matches: int,
            out: str, max_iter: int) -> None:
    """Maximum-likelihood fit of Dixon-Coles strengths on results.parquet."""
    from .sim.dixon_coles import fit_strengths

    fitted = fit_strengths(
        half_life_days=half_life_days,
        min_team_matches=min_team_matches,
        max_iter=max_iter,
    )
    p = fitted.save(out)
    m = fitted.meta
    click.echo(f"saved {p}")
    click.echo(f"  teams={m['n_teams']}  matches={m['n_matches']}  "
               f"converged={m['converged']}  nll={m['nll']:.1f}")
    click.echo(f"  intercept={fitted.params.intercept:.3f}  "
               f"home_adv={fitted.params.home_advantage:.3f}  "
               f"rho={fitted.params.rho:.3f}")
    click.echo("  top 10 by attack:")
    top_atk = sorted(fitted.strengths.items(),
                     key=lambda kv: -kv[1].attack)[:10]
    for name, s in top_atk:
        click.echo(f"    {name:<22} attack={s.attack:+.3f}  defence={s.defence:+.3f}")


# ---------- simulate ----------
@main.command()
@click.option("--home", required=True)
@click.option("--away", required=True)
@click.option("--n", type=int, default=10_000)
@click.option("--home-rating", type=float, default=None)
@click.option("--away-rating", type=float, default=None)
@click.option("--neutral", is_flag=True, default=False,
              help="Treat the venue as neutral (no home advantage).")
@click.option("--model", type=click.Path(), default=None,
              help=f"Path to dc_strengths.json (default: {DEFAULT_DC_PATH} if it exists).")
@click.option("--seed", type=int, default=None)
def simulate(home: str, away: str, n: int,
             home_rating: float | None, away_rating: float | None,
             neutral: bool, model: str | None, seed: int | None) -> None:
    """Dixon-Coles simulation between two teams.

    Auto-loads ``data/processed/dc_strengths.json`` if it exists; otherwise
    falls back to Elo-style rating-derived expected goals.
    """
    from .sim.base import TeamSpec
    from .sim.dixon_coles import DixonColesSimulator, FittedModel

    chosen_model = model or (str(DEFAULT_DC_PATH) if DEFAULT_DC_PATH.exists() else None)
    if chosen_model:
        from .sim.dixon_coles import TEAM_ALIASES
        fitted = FittedModel.load(chosen_model)
        sim = DixonColesSimulator.from_fit(fitted, seed=seed)
        m = fitted.meta
        click.echo(f"loaded fitted model: teams={m.get('n_teams')} "
                   f"matches={m.get('n_matches')} asof={m.get('asof', '?')}")

        def _resolve_label(name: str) -> str | None:
            if name in fitted.strengths:
                return name
            alias = TEAM_ALIASES.get(name)
            if alias and alias in fitted.strengths:
                return alias
            return None

        for label, name in (("home", home), ("away", away)):
            resolved = _resolve_label(name)
            if resolved is None:
                click.echo(f"  ! {name} not in fitted strengths; falling back to rating",
                           err=True)
            elif resolved != name:
                click.echo(f"  resolved {name} -> {resolved}")
    else:
        sim = DixonColesSimulator(seed=seed)
        click.echo("no fitted model found — using rating-only fallback "
                   "(run `wcp fit dixon-coles` first)")

    h = TeamSpec(name=home, rating=home_rating)
    a = TeamSpec(name=away, rating=away_rating)
    # Stash neutral flag onto sim by closure: the interface takes neutral via
    # simulate_match kwarg, but simulate_distribution drives that internally;
    # patch by wrapping.
    orig = sim.simulate_match
    sim.simulate_match = lambda H, A: orig(H, A, neutral=neutral)  # type: ignore[assignment]

    dist = sim.simulate_distribution(h, a, n=n)
    click.echo(f"{home} vs {away}  (n={n}{', neutral' if neutral else ''})")
    click.echo(f"  P({home} win) = {dist.home_win_prob:.3f}")
    click.echo(f"  P(draw)       = {dist.draw_prob:.3f}")
    click.echo(f"  P({away} win) = {dist.away_win_prob:.3f}")
    click.echo(f"  E[goals]      = {dist.expected_home_goals:.2f} - {dist.expected_away_goals:.2f}")
    click.echo("  Top scorelines:")
    for (h_g, a_g), p in dist.top_scorelines[:5]:
        click.echo(f"    {h_g}-{a_g}: {p:.3f}")


if __name__ == "__main__":
    main()
