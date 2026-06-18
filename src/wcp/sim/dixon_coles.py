"""Dixon-Coles match simulator + MLE fitter.

Implements the Dixon & Coles (1997) model:

    lambda_home = exp(intercept + home_adv * h_flag + attack[home] - defence[away])
    lambda_away = exp(intercept             + attack[away] - defence[home])

with the low-score correlation correction tau(h, a, lambda_h, lambda_a, rho)
on cells (0,0), (0,1), (1,0), (1,1). Likelihood is a product of independent
Poissons multiplied by tau, with each match weighted by:

    w(t) = exp(-xi * (today - match_date in days)) * importance(tournament)

Identifiability: average attack across teams is constrained to 0 each step.

The simulator class (`DixonColesSimulator`) keeps the original interface from
the project skeleton — fall back to Elo-derived expected goals when no fitted
strengths are loaded for a team, otherwise use the MLE-fitted ones.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from .base import MatchResult, MatchSimulator, TeamSpec


# ---------------------------------------------------------------------------
# Parameter containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeamStrength:
    attack: float
    defence: float


@dataclass(frozen=True)
class ModelParams:
    """Global / fitted parameters."""

    intercept: float = math.log(1.30)        # base log-rate ~ league avg goals/team
    home_advantage: float = 0.30             # log multiplier for non-neutral home side
    rho: float = -0.05                       # DC low-score correlation correction


@dataclass
class FittedModel:
    params: ModelParams
    strengths: dict[str, TeamStrength]
    meta: dict

    # ---- persistence -------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "params": asdict(self.params),
            "strengths": {k: asdict(v) for k, v in self.strengths.items()},
            "meta": self.meta,
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "FittedModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            params=ModelParams(**d["params"]),
            strengths={k: TeamStrength(**v) for k, v in d["strengths"].items()},
            meta=d.get("meta", {}),
        )


# ---------------------------------------------------------------------------
# DC correction + samplers
# ---------------------------------------------------------------------------


def _dc_correction(h: int, a: int, lam_h: float, lam_a: float, rho: float) -> float:
    if h == 0 and a == 0:
        return 1 - lam_h * lam_a * rho
    if h == 0 and a == 1:
        return 1 + lam_h * rho
    if h == 1 and a == 0:
        return 1 + lam_a * rho
    if h == 1 and a == 1:
        return 1 - rho
    return 1.0


def _sample_poisson(rng: random.Random, lam: float) -> int:
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
    for _ in range(20):
        h = _sample_poisson(rng, lam_h)
        a = _sample_poisson(rng, lam_a)
        if h > max_goals or a > max_goals:
            continue
        accept = max(0.0, _dc_correction(h, a, lam_h, lam_a, rho))
        if accept >= 1.0 or rng.random() < accept:
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


# ---------------------------------------------------------------------------
# MLE fit
# ---------------------------------------------------------------------------


# Tournament importance multipliers — friendlies count less, WC counts more.
# Non-FIFA / sub-confederation tournaments get near-zero weight to keep
# minor-island sides from dominating the fit when they pile up wins in
# regional games against very weak opponents.
_TOURNAMENT_WEIGHT: dict[str, float] = {
    "Friendly": 0.4,
    "FIFA World Cup": 1.5,
    "FIFA World Cup qualification": 1.0,
    "UEFA Euro": 1.3,
    "UEFA Euro qualification": 0.9,
    "UEFA Nations League": 1.0,
    "Copa América": 1.2,
    "African Cup of Nations": 1.0,
    "African Cup of Nations qualification": 0.8,
    "AFC Asian Cup": 1.0,
    "AFC Asian Cup qualification": 0.8,
    "CONCACAF Nations League": 1.0,
    "CONCACAF Gold Cup": 1.0,
    "Confederations Cup": 1.0,
    # Near-zero weight: regional / non-FIFA-rated tournaments.
    "Island Games": 0.0,
    "Inter Games": 0.0,
    "Gulf Cup": 0.2,
    "CECAFA Cup": 0.2,
    "WAFF Cup": 0.2,
    "Pacific Games": 0.0,
    "Indian Ocean Island Games": 0.0,
    "Mediterranean Games": 0.0,
}


def _tournament_weight(name: str | float) -> float:
    if not isinstance(name, str):
        return 0.5
    return _TOURNAMENT_WEIGHT.get(name, 0.5)


# User-facing aliases for common short names. Apply to both home/away when
# resolving a fitted strength — keeps `--home USA` working when the dataset
# stores "United States".
TEAM_ALIASES: dict[str, str] = {
    "USA": "United States",
    "UAE": "United Arab Emirates",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "South Korea": "South Korea",
    "Korea Republic": "South Korea",
    "North Korea": "North Korea",
    "Korea DPR": "North Korea",
    "Ivory Coast": "Côte d'Ivoire",
    "Cape Verde": "Cape Verde",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
}


def _build_design(
    df,
    half_life_days: float,
    min_team_matches: int,
    asof: datetime,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Project results.parquet rows into model arrays."""
    import pandas as pd

    df = df.dropna(subset=["home_score", "away_score", "home_team", "away_team", "date"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Drop teams with too few matches — their strengths are unidentified.
    counts = (
        df["home_team"].value_counts().add(df["away_team"].value_counts(), fill_value=0)
    )
    keep = set(counts[counts >= min_team_matches].index)
    df = df[df["home_team"].isin(keep) & df["away_team"].isin(keep)]

    teams = sorted(set(df["home_team"]).union(df["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}

    home_idx = df["home_team"].map(idx).to_numpy()
    away_idx = df["away_team"].map(idx).to_numpy()
    home_g = df["home_score"].to_numpy()
    away_g = df["away_score"].to_numpy()
    h_flag = (~df["neutral"].astype(bool)).to_numpy().astype(float)

    age_days = (asof - df["date"]).dt.days.to_numpy().astype(float)
    age_days = np.clip(age_days, 0, None)
    xi = math.log(2.0) / max(half_life_days, 1.0)
    time_w = np.exp(-xi * age_days)
    tour_w = df["tournament"].map(_tournament_weight).to_numpy()
    weights = time_w * tour_w

    return home_idx, away_idx, home_g, away_g, h_flag, weights, teams


def _neg_log_likelihood(
    theta: np.ndarray,
    home_idx: np.ndarray, away_idx: np.ndarray,
    home_g: np.ndarray, away_g: np.ndarray,
    h_flag: np.ndarray, weights: np.ndarray,
    n_teams: int,
) -> float:
    attack = theta[:n_teams]
    defence = theta[n_teams:2 * n_teams]
    intercept, home_adv, rho = theta[-3], theta[-2], theta[-1]

    # Identifiability: zero-mean attack
    attack = attack - attack.mean()

    log_lam_h = intercept + home_adv * h_flag + attack[home_idx] - defence[away_idx]
    log_lam_a = intercept + attack[away_idx] - defence[home_idx]
    lam_h = np.exp(log_lam_h)
    lam_a = np.exp(log_lam_a)

    # Poisson log-pmf using gammaln for stability
    from scipy.special import gammaln
    lp_h = home_g * log_lam_h - lam_h - gammaln(home_g + 1)
    lp_a = away_g * log_lam_a - lam_a - gammaln(away_g + 1)
    lp = lp_h + lp_a

    # Dixon-Coles correction (log) on low-score cells only
    h0a0 = (home_g == 0) & (away_g == 0)
    h0a1 = (home_g == 0) & (away_g == 1)
    h1a0 = (home_g == 1) & (away_g == 0)
    h1a1 = (home_g == 1) & (away_g == 1)
    correction = np.ones_like(lam_h)
    correction = np.where(h0a0, 1 - lam_h * lam_a * rho, correction)
    correction = np.where(h0a1, 1 + lam_h * rho, correction)
    correction = np.where(h1a0, 1 + lam_a * rho, correction)
    correction = np.where(h1a1, 1 - rho, correction)
    correction = np.maximum(correction, 1e-9)
    lp = lp + np.log(correction)

    # L2 ridge on team params keeps optimisation numerically stable
    ridge = 0.01 * (np.sum(attack ** 2) + np.sum(defence ** 2))
    return -float(np.sum(weights * lp)) + ridge


def fit_strengths(
    matches_df=None,
    *,
    half_life_days: float = 1825.0,
    min_team_matches: int = 50,
    asof: datetime | None = None,
    init: FittedModel | None = None,
    max_iter: int = 500,
) -> FittedModel:
    """Maximum-likelihood fit of Dixon-Coles team strengths on
    ``data/processed/results.parquet`` (or the dataframe passed in).

    Returns a ``FittedModel`` with per-team (attack, defence) and global
    (intercept, home_advantage, rho).
    """
    import pandas as pd
    from scipy.optimize import minimize

    if matches_df is None:
        from ..ingest import results
        matches_df = results.load_results()

    asof = asof or datetime.utcnow()
    home_idx, away_idx, home_g, away_g, h_flag, weights, teams = _build_design(
        matches_df, half_life_days=half_life_days,
        min_team_matches=min_team_matches, asof=asof,
    )
    n = len(teams)
    if n == 0:
        raise ValueError("No teams left after min_team_matches filter")

    # Init from prior fit when available, otherwise zeros + sensible globals.
    theta0 = np.zeros(2 * n + 3)
    theta0[-3] = math.log(1.30)
    theta0[-2] = 0.25
    theta0[-1] = -0.05
    if init is not None:
        for i, t in enumerate(teams):
            s = init.strengths.get(t)
            if s is None:
                continue
            theta0[i] = s.attack
            theta0[n + i] = s.defence
        theta0[-3] = init.params.intercept
        theta0[-2] = init.params.home_advantage
        theta0[-1] = init.params.rho

    # Bounds: rho in (-1, 1); home_adv non-negative-ish; attack/defence bounded.
    bounds = (
        [(-3, 3)] * n              # attack
        + [(-3, 3)] * n            # defence
        + [(-2, 3)]                # intercept
        + [(0, 1.5)]               # home advantage
        + [(-0.99, 0.99)]          # rho
    )

    res = minimize(
        _neg_log_likelihood,
        theta0,
        args=(home_idx, away_idx, home_g, away_g, h_flag, weights, n),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "disp": False},
    )

    theta = res.x
    attack = theta[:n] - theta[:n].mean()
    defence = theta[n:2 * n]
    intercept, home_adv, rho = theta[-3], theta[-2], theta[-1]

    fitted = FittedModel(
        params=ModelParams(intercept=float(intercept),
                           home_advantage=float(home_adv),
                           rho=float(rho)),
        strengths={t: TeamStrength(float(attack[i]), float(defence[i]))
                   for i, t in enumerate(teams)},
        meta={
            "n_matches": int(home_idx.size),
            "n_teams": int(n),
            "half_life_days": float(half_life_days),
            "min_team_matches": int(min_team_matches),
            "asof": asof.isoformat(),
            "converged": bool(res.success),
            "nll": float(res.fun),
            "iterations": int(res.nit),
        },
    )
    return fitted


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class DixonColesSimulator(MatchSimulator):
    def __init__(
        self,
        params: ModelParams | None = None,
        strengths: dict[str, TeamStrength] | None = None,
        seed: int | None = None,
    ) -> None:
        self.params = params or ModelParams()
        self.strengths = strengths or {}
        self._rng = random.Random(seed)

    @classmethod
    def from_fit(cls, fit: FittedModel, seed: int | None = None) -> "DixonColesSimulator":
        return cls(params=fit.params, strengths=fit.strengths, seed=seed)

    def _resolve(self, name: str) -> TeamStrength | None:
        s = self.strengths.get(name)
        if s is not None:
            return s
        alias = TEAM_ALIASES.get(name)
        if alias:
            return self.strengths.get(alias)
        return None

    def _expected_goals(
        self, home: TeamSpec, away: TeamSpec, neutral: bool = False
    ) -> tuple[float, float]:
        p = self.params
        h_flag = 0.0 if neutral else 1.0
        sh = self._resolve(home.name)
        sa = self._resolve(away.name)
        if sh and sa:
            log_lam_h = p.intercept + p.home_advantage * h_flag + sh.attack - sa.defence
            log_lam_a = p.intercept + sa.attack - sh.defence
            return math.exp(log_lam_h), math.exp(log_lam_a)

        # Fallback: use Elo-like ratings if provided, otherwise neutral 1.3 each.
        rh = home.rating if home.rating is not None else 1500.0
        ra = away.rating if away.rating is not None else 1500.0
        diff = (rh - ra) / 400.0
        base = math.exp(p.intercept)
        lam_h = base * math.exp(p.home_advantage * h_flag + 0.3 * diff)
        lam_a = base * math.exp(-0.3 * diff)
        return lam_h, lam_a

    def simulate_match(
        self, home: TeamSpec, away: TeamSpec, neutral: bool = False
    ) -> MatchResult:
        lam_h, lam_a = self._expected_goals(home, away, neutral=neutral)
        h, a = _sample_bivariate_poisson_dc(self._rng, lam_h, lam_a, self.params.rho)
        return MatchResult(
            home=home.name, away=away.name,
            home_goals=h, away_goals=a,
            scorers_home=_attribute_goals(self._rng, h, home.squad),
            scorers_away=_attribute_goals(self._rng, a, away.squad),
        )
