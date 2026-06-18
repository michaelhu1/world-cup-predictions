"""Match simulation subpackage.

Three-layer hybrid (see ``docs/architecture.md``):

1. ``dixon_coles.DixonColesSimulator`` — bivariate-Poisson backbone for
   per-fixture (lambda_home, lambda_away) and scoreline distribution.
2. ``scorer.ScorerAllocator`` (TODO) — hierarchical Dirichlet-Multinomial
   conditioned on sampled team goals; player priors from socceraction VAEP/xG.
3. ``tournament.TournamentSimulator`` (TODO) — Monte Carlo over the 2026
   48-team format with extra-time and shootout sub-models.

Research (RL/MARL/GRF/trajectory/learned-event-models) was reviewed and
rejected for the solo-dev WC 2026 use case: every candidate was missing at
least one of {team identity, player identity, scoreline output, international
data, feasible compute}.
"""
