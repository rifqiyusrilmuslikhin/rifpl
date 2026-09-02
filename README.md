# FPL Model

Python research scaffold for a Fantasy Premier League modeling project. Notebooks only orchestrate
package code and display results.

## Requirements

- Python 3.12 (the development version is pinned in `.python-version`)
- PowerShell 7 or Windows PowerShell
- Git and VSCode with the Python extension (recommended)

## Local setup (PowerShell / VSCode)

From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

In VSCode, run **Python: Select Interpreter** and choose `.venv\Scripts\python.exe`. Verify the
environment with:

```powershell
python -m ruff format --check .
python -m ruff check .
python -m pytest
python -c "import fpl_model; print(fpl_model.__version__)"
```

To apply formatting, run `python -m ruff format .`.

## Configuration

Project defaults live in `config/project.toml`. Paths are relative to the repository root unless
an absolute path is supplied. `FPL_CONFIG` may point at an alternative TOML file:

```powershell
$env:FPL_CONFIG = "config/project.toml"
python -c "from fpl_model.config import load_config; print(load_config())"
```

The canonical timezone is UTC. Timestamps used by future pipeline stages must be timezone-aware
and normalized to UTC before storage or comparison.

## Google Colab

Clone or upload this repository, change into its root, then open
`notebooks/00_colab_smoke_test.ipynb`. Its first cell installs the editable package and pinned
development dependencies; its second cell runs the same import smoke test used locally. Restart
the Colab runtime if pip reports that an already-imported dependency changed.

The notebook intentionally contains no acquisition, transformation, feature, or model logic.

## Repository layout

```text
config/                 project defaults (season, UTC, paths)
notebooks/              thin experiment orchestration
src/fpl_model/data/     data-layer modules
src/fpl_model/features/ feature-layer modules
src/fpl_model/models/   modeling modules
src/fpl_model/evaluation/ evaluation modules
src/fpl_model/decision/ decision-layer modules
tests/                  unit and smoke tests
data/                   ignored local data tiers
artifacts/              ignored model artifacts
reports/                ignored generated metrics and predictions
```

## Raw-data foundation

Sprint 1 adds versioned contracts for `player_fixture_fact`, `deadline_snapshot`,
`player_gameweek_model`, and the player identity registry. The data layer also provides:

- an immutable content-addressed raw artifact store with SHA-256 verification;
- JSON manifests containing source URL, pinned revision or content hash, UTC retrieval time,
  season, schema version, media type, byte count, and license note;
- historical CSV and current-season FPL API loaders with minimum source-schema checks;
- explicit states for genuine zero, source-unavailable fields, and acquisition failures.

Network calls are isolated in `fpl_model.data.acquisition`. Tests inject local fixture fetchers and
never require live network access. A live FPL endpoint smoke check is intentionally pending and is
not part of the deterministic test suite.

## Deadline snapshots and player identity

Sprint 2 adds a strict point-in-time boundary and a versioned identity registry:

- `DeadlineCalendar` stores canonical UTC deadlines and `SnapshotGate` accepts a bootstrap payload
  only when its one `is_next` event is the target GW, its embedded deadline is canonical, and its
  capture time is strictly pre-deadline;
- selection operates only on already accepted snapshots, so a missing historical snapshot remains
  missing instead of being backfilled from future data;
- `PlayerIdentityRegistry` assigns one immutable internal key from stable cross-season `fpl_code`,
  while retaining FPL element IDs as season aliases and team assignments as GW validity intervals;
- Understat mappings remain nullable until explicitly audited. Name normalization creates candidate
  proposals only, and reverse uniqueness prevents one external ID from attaching to two players.

## Canonical fixture and player-GW dataset

Sprint 3 adds `CanonicalDatasetBuilder`, which deliberately freezes accepted deadline-snapshot and
fixture-schedule context before attaching completed outcomes:

- `player_fixture_fact` preserves one raw outcome per player and EPL fixture, with the player's team
  resolved from the identity interval valid for that fixture's GW and the opponent derived from the
  canonical schedule;
- pre-deadline fixture context requires an explicit `available_at_utc < deadline_utc` and gives every
  fixture in a player-DGW the same `feature_cutoff_utc` and `context_anchor_id`;
- `player_gameweek_model` has exactly one `(season, gameweek, player_key)` row, sums DGW points and
  minutes, and derives the participation/points labels only after context is frozen;
- the decision dataset retains every player in the accepted snapshot. A player with no scheduled
  fixture is an explicit blank row with `fixture_count=0`, `is_blank=true`, and zero outcome targets.
  Later ranking-training code may exclude blanks, but it must filter this flag consistently for every
  model and evaluation window;
- `ReconciliationReport` fails construction on duplicate/lost rows or mismatched per-player and
  global totals, and can retain the passing result with `write_json(...)`.

The fixture schedule input accepts canonical field names or FPL aliases, but additionally requires
`season`, `available_at_utc`, and `source_artifact_id` so the schedule itself has point-in-time
provenance. Fixture difficulty remains nullable and is aggregated only when all fixtures in the GW
provide it. Model training, evaluation logic, and decision logic remain deferred to later sprints.

## Baseline feature contract

Sprint 4 adds a fixed, machine-readable 46-feature contract and a causal feature builder. The
contract is retained in `src/fpl_model/features/baseline_contract.json`; every feature declares its
dtype, source, availability cutoff, window, aggregation, missing-data rule, and hypothesis.

`BaselineFeatureBuilder.build(...)` consumes canonical player-GW rows and fixture contexts plus
player-fixture and team-fixture history. It returns keys and provenance followed by exactly the 46
contracted features. Player histories are filtered and sorted independently at every deadline
before last-3/5/10 aggregation. Team and opponent histories use prior completed EPL fixtures only.
All fixtures in a DGW must share one pre-deadline anchor, and opponent rates are averaged across
the target fixtures.

Player-history input supports the canonical fixture fields plus FPL statistics (`goals_scored`,
`assists`, `starts`, `bps`, `bonus`, and `yellow_cards`) and nullable audited Understat fields (`xg`,
`xa`, `shots`, and `key_passes`). Team history may be supplied either as one team-perspective row or
as a home/away fixture row. If `completed_at_utc` is absent, completion is conservatively treated as
three hours after kickoff; if `available_at_utc` is absent, it defaults to completion. Both must be
strictly before the target deadline.

Per-90 rates use summed window values and minutes, a fixed `1e-6` minute epsilon, and a 90-minute
minimum. Partial rolling windows are allowed, but missing Understat values or unresolved mappings
remain missing. No imputation, normalization, model training, or feature selection is performed in
this layer. `build_coverage_report(...)`, `write_coverage_report(...)`, and
`write_spot_check_report(...)` retain the required per-season missingness and representative-player
audit artifacts.

## Leakage release gate

Sprint 5 adds fail-fast point-in-time and fold-scope checks in
`fpl_model.evaluation.leakage_check`, without starting the walk-forward evaluation harness:

- feature and source-provenance timestamps must be strictly before each row's deadline;
- target labels, unaggregated current-GW outcomes, duplicate canonical keys, and suspicious direct
  target proxies are rejected from model feature lists;
- DGW fixtures must share one deadline anchor and identical history-derived values;
- training rows must be strictly earlier than test rows, with no later-season contamination;
- learned preprocessing must retain fit-row keys and prove that they match the training fold only.

The baseline feature frame now retains `snapshot_captured_at_utc` and validates it independently of
`feature_cutoff_utc`. Snapshot status/chance value states are also checked so genuine numeric zero
cannot be confused with source-unavailable or acquisition-failure missingness. Adversarial tests
deliberately inject target-GW minutes, first-fixture DGW knowledge, test-period preprocessing rows,
later-season rows, and a renamed perfect target proxy. A broad mutation test changes post-deadline
labels, player outcomes, team outcomes, and provenance while requiring the pre-deadline feature
frame to remain exactly identical.

## Walk-forward evaluation

Sprint 6 adds the model-agnostic evaluation harness in `fpl_model.evaluation`:

- `config/evaluation_windows.toml` freezes warm-up, discovery, calibration, confirmation, and
  2026-27 prospective boundaries. Only discovery windows carry selection permission;
- `ExpandingWindowSplitter` uses a distinct immediately preceding calibration block and constructs
  training data only from earlier configured windows, dropping locally present future seasons;
- ranking and error metrics are computed inside each `(season, gameweek)` before their unweighted
  mean is reported. Duplicate player-GW rows are rejected so a canonical DGW row is scored once;
- probability reports include Brier score, log loss, ROC-AUC, PR-AUC, reliability bins,
  calibration intercept/slope, ECE, and per-GW top-k precision/lift/recall diagnostics;
- required eligibility, participation, position, SGW/DGW, season-phase, and availability cohorts
  are formed after prediction. The outcome-derived played and 60+ cohorts are explicitly marked
  diagnostic-only;
- paired comparisons require identical canonical rows, targets, eligibility, and GWs. Confidence
  intervals resample whole GWs rather than player rows;
- `WalkForwardHarness` passes target-free test inputs to predictors and verifies their retained
  fit-row provenance. Its immutable Parquet artifact retains keys, fold, seed, eligibility,
  targets, predictions, and baseline columns with a SHA-256 metadata sidecar, allowing reports to
  be regenerated without fitting again.

## Simple baselines

Sprint 7 adds one fold-local baseline runner in `fpl_model.models`. It sends always-zero,
last-appearance, Last-5, position-by-recent-minutes, price ranking, historical participation and
minutes, Logistic Regression, and Ridge predictions through the Sprint 6 walk-forward harness.
Categorical encoding, numeric imputation, and scaling are fitted separately inside every training
fold. Ridge retains raw diagnostics and clipped inference outputs; price is reported only with
ranking metrics.

The feature builder now retains `points_last_appearance` and the nullable official `ep_next` as
baseline inputs outside the fixed 46-feature contract. Official xPts is accepted only when its
value state agrees with a strictly pre-deadline snapshot, remains missing when unavailable, and is
reported with per-season coverage plus a same-row Last-5 comparator. Historical cold-start
fallbacks retain explicit availability indicators in the OOF artifact.

`run_simple_baselines(...)` returns the same immutable `RetainedPredictions` artifact used by the
evaluation harness. `regenerate_baseline_report(...)` reconstructs points/ranking metrics,
minutes errors, Brier/log-loss/AUC results, reliability bins, coverage, sanity checks, and a
Sprint 8 stop/go recommendation without retraining. `FrozenBaselineReport.write_json(...)`
refuses to overwrite a previously frozen report.
