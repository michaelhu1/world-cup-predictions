"""Match simulator interface and shared result types.

Anything that can simulate a single match for the predictor must conform to
``MatchSimulator``. Multiple backends (Dixon-Coles, learned event model, RL
policy rollout) plug in by implementing ``simulate_match`` and optionally
``simulate_tournament``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class TeamSpec:
    """Minimal info needed to simulate a team in a match.

    ``squad`` is a list of (player_name, minutes_per_match_estimate, scoring_rate)
    tuples; if absent, the simulator falls back to anonymous goal sampling.
    """
    name: str
    rating: float | None = None
    squad: list[tuple[str, float, float]] = field(default_factory=list)


@dataclass(frozen=True)
class MatchResult:
    home: str
    away: str
    home_goals: int
    away_goals: int
    scorers_home: tuple[str, ...] = ()
    scorers_away: tuple[str, ...] = ()

    @property
    def winner(self) -> str | None:
        if self.home_goals > self.away_goals:
            return self.home
        if self.away_goals > self.home_goals:
            return self.away
        return None


@dataclass(frozen=True)
class MatchDistribution:
    """Aggregate of ``n`` simulated matches between the same two teams."""
    home: str
    away: str
    n: int
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    expected_home_goals: float
    expected_away_goals: float
    top_scorelines: list[tuple[tuple[int, int], float]]
    top_scorers: list[tuple[str, float]]


class MatchSimulator(ABC):
    """Abstract simulator. Implementations: Dixon-Coles, learned, RL."""

    @abstractmethod
    def simulate_match(self, home: TeamSpec, away: TeamSpec) -> MatchResult:
        """Single sample from the match outcome distribution."""

    def simulate_distribution(
        self, home: TeamSpec, away: TeamSpec, n: int = 10_000
    ) -> MatchDistribution:
        """Default Monte Carlo aggregation; override if a closed form is cheaper."""
        from collections import Counter

        score_counts: Counter[tuple[int, int]] = Counter()
        scorer_counts: Counter[str] = Counter()
        h_wins = d = a_wins = 0
        h_goals_total = 0
        a_goals_total = 0
        for _ in range(n):
            r = self.simulate_match(home, away)
            score_counts[(r.home_goals, r.away_goals)] += 1
            if r.home_goals > r.away_goals:
                h_wins += 1
            elif r.away_goals > r.home_goals:
                a_wins += 1
            else:
                d += 1
            h_goals_total += r.home_goals
            a_goals_total += r.away_goals
            for s in r.scorers_home + r.scorers_away:
                scorer_counts[s] += 1
        top_scorelines = [(k, v / n) for k, v in score_counts.most_common(10)]
        top_scorers = [(k, v / n) for k, v in scorer_counts.most_common(20)]
        return MatchDistribution(
            home=home.name, away=away.name, n=n,
            home_win_prob=h_wins / n, draw_prob=d / n, away_win_prob=a_wins / n,
            expected_home_goals=h_goals_total / n,
            expected_away_goals=a_goals_total / n,
            top_scorelines=top_scorelines,
            top_scorers=top_scorers,
        )
