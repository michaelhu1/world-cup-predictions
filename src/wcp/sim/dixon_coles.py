"""Baseline Dixon-Coles match simulator.

Dixon & Coles (1997, JRSS-C) extended the bivariate Poisson goal model with a
low-score correlation correction (rho) plus an exponential time-decay weight on
historical matches. This module ships a deliberately small implementation:

- Goal totals come from a bivariate-Poisson-with-rho draw.
- Conditional on goals, each goal is attributed to a player by sampling from
  ``TeamSpec.squad`` with weights proportional to that player's per-match
  scoring rate (or uniformly at random if no squad given).

Fitting team strengths (alpha/beta/gamma) lives in :func:`fit_strengths`,
which currently does *no* fitting and returns a neutral default — the real
fitter will land once results.parquet is materialised by the ingester.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .base import MatchResult, MatchSimulator, TeamSpec


@dataclass(frozen=True)
class TeamStrength:
    """Per-team latent strengths (Dixon-Coles parameterisation)."""
    attack: float
    defence: float


@dataclass(frozen=True)
class ModelParams:
    home_advantage: float = 0.30
    rho: float = -0.05  # Dixon-Coles low-score correlation
    base_rate: float = 1.30  # average goals per side, league-/era-specific


def fit_strengths(
    matches_df=None, decay_half_life_days: float = 1825,  # ~5 years
) -> dict[str, TeamStrength]:
    """Placeholder fitter. Returns an empty dict until a real fit is wired.

    To wire it: maximise the Dixon-Coles likelihood over per-team alpha/beta
    plus a global gamma (home advantage), with exponential time decay weights.
    """
    return {}


def _dc_correction(home_goals: int, away_goals: int, lam_h: float, lam_a: float, rho: float) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1 - lam_h * lam_a * rho
    if home_goals == 0 and away_goals == 1:
        return 1 + lam_h * rho
    if home_goals == 1 and away_goals == 0:
        return 1 + lam_a * rho
    if home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


def _sample_poisson(rng: random.Random, lam: float) -> int:
    """Knuth's Poisson sampler — fine for small lambda we deal with here."""
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def _sample_bivariate_poisson_dc(
    rng: random.Random, lam_h: float, lam_a: float, rho: float, max_goals: int = 12
) -> tuple[int, int]:
    """Sample (home_goals, away_goals) from independent Poissons reweighted by
    the Dixon-Coles low-score correction. Rejection sample at most a few times."""
    for _ in range(20):
        h = _sample_poisson(rng, lam_h)
        a = _sample_poisson(rng, lam_a)
        if h > max_goals or a > max_goals:
            continue
        accept_p = max(0.0, _dc_correction(h, a, lam_h, lam_a, rho))
        if accept_p >= 1.0 or rng.random() < accept_p:
            return h, a
    return _sample_poisson(rng, lam_h), _sample_poisson(rng, lam_a)


def _attribute_goals(
    rng: random.Random, n_goals: int, squad: list[tuple[str, float, float]]
) -> tuple[str, ...]:
    if n_goals == 0:
        return ()
    if not squad:
        return tuple(f"unknown_{i+1}" for i in range(n_goals))
    weights = [max(1e-6, m * r) for (_, m, r) in squad]
    total = sum(weights)
    weights = [w / total for w in weights]
    names = [n for (n, _, _) in squad]
    out: list[str] = []
    for _ in range(n_goals):
        x = rng.random()
        acc = 0.0
        chosen = names[-1]
        for n, w in zip(names, weights):
            acc += w
            if x <= acc:
                chosen = n
                break
        out.append(chosen)
    return tuple(out)


class DixonColesSimulator(MatchSimulator):
    """Baseline simulator. Uses team ``rating`` (e.g. Elo) as a strength proxy
    when fitted strengths aren't available yet."""

    def __init__(self, params: ModelParams | None = None,
                 strengths: dict[str, TeamStrength] | None = None,
                 seed: int | None = None) -> None:
        self.params = params or ModelParams()
        self.strengths = strengths or {}
        self._rng = random.Random(seed)

    def _expected_goals(self, home: TeamSpec, away: TeamSpec) -> tuple[float, float]:
        p = self.params
        # If we have fitted strengths use them; otherwise fall back to rating-derived
        # expected goals via a logistic squash on the rating diff.
        sh = self.strengths.get(home.name)
        sa = self.strengths.get(away.name)
        if sh and sa:
            lam_h = math.exp(p.home_advantage + sh.attack - sa.defence) * p.base_rate
            lam_a = math.exp(sa.attack - sh.defence) * p.base_rate
            return lam_h, lam_a

        rh = home.rating if home.rating is not None else 1500
        ra = away.rating if away.rating is not None else 1500
        diff = (rh - ra) / 400.0  # standard Elo scaling
        lam_h = p.base_rate * math.exp(p.home_advantage + 0.3 * diff)
        lam_a = p.base_rate * math.exp(-0.3 * diff)
        return lam_h, lam_a

    def simulate_match(self, home: TeamSpec, away: TeamSpec) -> MatchResult:
        lam_h, lam_a = self._expected_goals(home, away)
        h, a = _sample_bivariate_poisson_dc(self._rng, lam_h, lam_a, self.params.rho)
        return MatchResult(
            home=home.name, away=away.name,
            home_goals=h, away_goals=a,
            scorers_home=_attribute_goals(self._rng, h, home.squad),
            scorers_away=_attribute_goals(self._rng, a, away.squad),
        )
