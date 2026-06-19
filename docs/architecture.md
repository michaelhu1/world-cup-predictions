# Architecture Decision Record: simulator family

Date: 2026-06-18

## Decision

Three-layer hybrid for the 2026 FIFA World Cup predictor:

1. **Goal-totals backbone** — Dixon-Coles bivariate Poisson with time-decayed
   weights, fitted on public international results (1872–present). Optional
   Bayesian shrinkage toward World Football Elo / SPI for low-data nations.
   Produces per-fixture (λ_home, λ_away) → calibrated scoreline distribution.
2. **Scorer allocator** — hierarchical Dirichlet-Multinomial conditioned on
   sampled team goals. Per-player goal rate priors derived via socceraction
   (VAEP/90, xG/90) on StatsBomb Open Data (men's WC 1958–2022 + Euros +
   qualifiers), combined with lineup probability, penalty-taker designation,
   set-piece role, and opponent defensive strength. Outputs scorer list per
   simulated match; aggregates give Golden Boot probabilities.
3. **Tournament Monte Carlo** — 50k–100k iterations of the 2026 format
   (48 teams, 12 groups of 4, 32-team knockout). Stage-specific λ adjustments
   for extra-time; logistic shootout sub-model fitted on every WC/Euro/Copa
   shootout since 1982 (~80 instances; wide priors).

Plus a calibration layer (isotonic recalibration on held-out WC 2022 + Euro 2024,
reliability diagrams + Brier on every release).

## Why this and not RL / learned simulators

Adversarial-verified review of 35 candidates from 14 research lenses (RL/MARL,
learned event models, trajectory transformers, GNN tactical models, foundation
models). Findings:

- **Google Research Football MARL family** (TiZero, TiKick, MAPPO, CDS,
  Light-MALib, Diversity-is-Strength, hierarchical QMIX, TLeague, NeuPL,
  SEED RL): anonymous interchangeable agents, no team/player identity, no
  path from real international results, compute budgets 24 GPUs × 9000 CPUs
  × 70 days for state-of-the-art entries.
- **Embodied-physics RL** (DeepMind 2v2 humanoid soccer, OP3 bipeds, Emergent
  Coordination): TPU-weeks of compute, anonymous walkers, irrelevant to outcome
  prediction.
- **Learned event simulators on club data** (Seq2Event, NMSTPP/HPUS, LEM,
  Unified-LEM, OpenSTARLab): produce W/D/L and scoreline distributions but
  lack team identity AND player identity in their published form; trained on
  Wyscout/StatsBomb club leagues, distribution-shifted from internationals;
  errors compound over ~1500–2000 events per match.
- **Tracking-based trajectory models** (SoccerMap/EPV, TacticAI, FootBots,
  TranSPORTmer, UniTraj, MC Pass Search): require synchronized 10–25 Hz
  player+ball tracking, not publicly available for international fixtures
  (Hawk-Eye / FIFA feeds are licensed). No validated 90-minute coherent
  rollouts. No team identity conditioning.

Net: every learned-simulator option is wrong-output, wrong-data, or
wrong-compute for the solo-dev WC 2026 use case. A statistical scoreline
backbone with a lightweight learned scorer-allocation head is dominant.

## Build order

| Phase | Days | Deliverable |
|------|------|-------------|
| Week 1–2 | 10 | Dixon-Coles fitter + tournament MC end-to-end (functional) |
| Week 3–4 | 10 | socceraction VAEP/xG player priors + Dirichlet-Multinomial scorer head |
| Week 5–6 | 10 | Calibration on WC 2022 + Euro 2024; reliability diagrams; Brier |
| Week 7+  | ongoing | Live-tournament tooling: lineup/injury overrides, daily refit, dashboard |

## Key risks

1. Sparse data for new 48-team format qualifiers → confederation-level
   shrinkage priors, Elo blending.
2. Lineup uncertainty dominates scorer error → publish player distributions
   conditional on starting XI; update at lineup announcement (~75 min pre-kickoff).
3. Calibration drift on emerging teams (Morocco-style 2022 surprises) →
   tune time-decay ξ on WC 2018 + WC 2022 holdouts.
4. Extra-time and shootouts are structurally different → dedicated submodels.
5. StatsBomb international coverage uneven → role-based fallbacks for
   under-covered players.
6. Multi-tournament holdout (WC 2018, Euro 2020, Copa 2021, WC 2022, Euro 2024)
   to avoid over-fitting to 2022 patterns.
7. Freeze architecture before group stage; only update parameters during the
   tournament — resist swapping in a learned simulator mid-event.

## Bibliography (selected, full list in research transcript)

- Dixon & Coles 1997 — bivariate Poisson with low-score correction
- Karlis & Ntzoufras 2003 — bivariate Poisson goal models
- Baio & Blangiardo 2010 — Bayesian hierarchical football
- Decroos et al. 2019 — VAEP / socceraction
- Groll et al. 2018, 2022 — random-forest WC models
- Mendes-Neves et al. 2024, 2025 — Large Events Model (cross-validation only)
- Kurach et al. 2019 — Google Research Football (background only)
- StatsBomb Open Data — men's WC 1958–2022, Euros, qualifiers (primary feature source)
