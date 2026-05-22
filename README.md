# march-madness-2026

NCAA tournament outcome predictions for the Kaggle [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026) competition.

**Finished 65 of 3,462 teams (12,252 entrants) - top 2%.** Final Brier score 0.1229, vs 0.1097 for 1st place. My first Kaggle.

The long-form write-up of how I got there is in [METHODOLOGY.md](METHODOLOGY.md). This README is the project tour.

## The problem

Predict the probability of every possible matchup in the 2026 NCAA Men's and Women's tournaments. You submit one probability for each (lower_team_id, higher_team_id) pair for both brackets, about 132,000 predictions in total. Only the ~134 games that actually happen during the tournament score, and you're graded on **Brier score** - mean squared error between your predicted probability and the actual 0/1 outcome. Lower is better.

Two things make this hard:

1. The metric punishes confident wrong predictions heavily. If your model says 15% and the underdog wins, you eat 0.72 of Brier on a single game. So calibration matters more than raw accuracy.
2. Six rounds of single elimination amplify luck. One bracket-busting upset can swing you 100+ places on the leaderboard.

The mental model that follows from this: you're not trying to pick winners. You're trying to output **calibrated probabilities**. A 12 seed beats a 5 seed roughly 35% of the time historically. Your model should say 35% on those games, not 15% and not 50%.

## What's in this repo

```
src/
  data_loader.py         load the 35 Kaggle CSVs into a single DataStore container
  data_cleaner.py        pace-adjust box-score counts to per-100-possessions,
                         normalize OT games, compute Four Factors,
                         Pythagorean expectation and luck, conference-tourney stats,
                         flag the 2020 COVID season
  ratings.py             Elo, Glicko-2, TrueSkill, Bradley-Terry,
                         plus Massey Ordinal features from Kaggle's third-party rankings
  features.py            39 (lower_team - higher_team) difference features,
                         plus game-flip data augmentation
  model.py               the 6-model stacking ensemble (see below),
                         OOF ensemble-weight optimization,
                         and gender-specific isotonic calibration
  generate_submission.py vectorized inference over 132K matchups
  pipeline.py            end-to-end driver
  test_all_phases.py     sanity tests across every phase of the pipeline

submissions/
  submission_v1.csv      final submission, 132,133 rows of [matchup_id, predicted_prob]

METHODOLOGY.md           long-form write-up: what worked, what didn't,
                         and the bug list from my pre-submission code audit
requirements.txt
```

## Methodology

### 1. Multiple rating systems, not just one

Everyone in this competition uses Elo. I used four systems in parallel: **Elo**, **Glicko-2**, **TrueSkill**, and **Bradley-Terry**. Each measures team strength in a slightly different way:

- Elo is a moving average of point-spread-adjusted results with K-factor decay and conference-based offseason regression.
- Glicko-2 adds an explicit rating-deviation parameter that tracks how confident the rating is.
- TrueSkill is Bayesian and gives you (mu, sigma) for each team.
- Bradley-Terry is a logistic regression over team-indicator pairs - it solves the consistency problem analytically per season.

The pairwise correlation across teams in a given season is 0.93-0.97. Strongly related, but not collinear. The residual disagreement is what the ensemble needs. Two of the systems explicitly track uncertainty, which turned out to matter for early-round upset prediction.

I also use Kaggle's provided **Massey Ordinals** (third-party power rankings from systems like POM, MOR). I take the Day-133 snapshot (just before the tournament starts) and compress with a log-rank transform so rank 1 is ~97 and rank 350 is ~62.

### 2. Pace adjustment is non-negotiable

Raw box-score stats are useless without pace adjustment. A team scoring 80 in 65 possessions is doing something completely different from a team scoring 80 in 80 possessions. Every counting stat (FGM, FGA, TOV, ORB, etc.) gets converted to per-100-possessions. Estimated possessions per Dean Oliver's formula:

```
POSS = 0.5 * ((FGA + 0.475*FTA - ORB + TOV) + opp_equivalents)
```

I also normalize overtime games (multiply counting stats by 40 / (40 + 5*NumOT)) so a triple-OT game isn't misread as a 60-minute outburst of activity.

### 3. Game-flip augmentation

Every training row is "Team A vs Team B, A won". For each one I create a mirror row "Team B vs Team A, B won" with all difference features negated and the label flipped. This doubles the training set and forces the model to satisfy P(A > B) = 1 - P(B > A) by construction.

Caveat: features that are products of two negated values (`seed_product`, `seed_x_elo`) don't actually flip - because `(-a)(-b) = ab`. I had a bug here for about an hour where I was negating things I shouldn't have.

### 4. Out-of-fold everything

This is the single most important methodological choice in the whole project. Two leakage holes that beginners commonly leave open:

1. Optimizing ensemble weights inside CV by picking the weights that minimize Brier on the test fold.
2. Fitting the isotonic calibrator on training predictions.

Both make CV scores look better while making real-world predictions worse. The fix is:

- Train each base model with **expanding-window CV** (train on years [start..N], test on N+1).
- Collect each model's predictions on its held-out fold - these are **out-of-fold (OOF) predictions**.
- Fit ensemble weights against the OOF predictions, not the training-set predictions.
- Fit the isotonic calibrator against the OOF predictions, not the training-set predictions.

The labels were never seen by the model that produced those OOF predictions, so optimizing weights or a calibrator against them is honest.

Closing these holes moved my "honest" CV from 0.1628 to 0.1660. Uglier number, but real - the leaderboard came in at 0.1229, in line with the OOF estimate. The previous 0.1628 was a number I'd been lying to myself with.

### 5. The 6-model stacking ensemble

Without diversity, ensembling is pointless. The weight optimizer just picks the best single model and zeros out the rest. So the design deliberately forces disagreement:

| Model | Features it sees | Role |
|---|---|---|
| **XGBoost** (Optuna-tuned) | all 39 | the generalist |
| **LightGBM** | rating-system features only | specialist for ratings disagreement |
| **CatBoost** | performance / efficiency features only | specialist for box-score signal |
| **Logistic regression** | all 39 (median-imputed, scaled) | linear baseline, won't overfit |
| **TabPFN** | all 39 | pretrained tabular transformer |
| **Seed-difference lookup** | seed_diff only | non-parametric anchor for extreme matchups |

OOF-optimized final weights ended up roughly: **TabPFN 52%, XGB 31%, seed baseline 10%, LR 4%, LGB 3%, CB ~0%**.

The seed-difference lookup is just a table mapping `round(seed_diff)` to historical win rate. Contributes nothing on close matchups but acts as a strong anchor on 1 vs 16, 8 vs 9, 12 vs 5 - matchups where the trees can otherwise get overconfident.

TabPFN is a transformer pretrained on millions of synthetic tabular datasets. For datasets under 10K rows it routinely beats hand-tuned XGBoost out of the box. It became the single highest-leverage model in my stack.

### 6. Gender-specific calibration

The women's tournament is dramatically chalkier than the men's. Women's 1-seeds have lost to 16-seeds exactly once in history. Men's 1-seeds drop to 16-seeds about once a decade. A single calibrator trained on combined data is too cautious on the women's bracket (giving favorites only 90% when 99% is correct) and too confident on the men's.

I train one model on combined data with a gender flag as a feature, but the isotonic calibration step runs separately per gender, and the probability clipping bounds differ: men get [0.02, 0.98], women get [0.01, 0.99].

### 7. Pre-submission code audit

Before submitting I read every source file end to end looking for bugs. I found twelve. The major ones:

- Home court advantage was being applied to both teams instead of only the home team - effectively doubling the 82-point Elo swing.
- Coach experience features were counting future seasons (temporal leakage).
- Two feature columns held identical data (last14 Elo trend was a duplicate of last14 win rate).
- Defensive efficiency was being used with the wrong sign convention in some places.
- Ensemble weights were being fit on training predictions instead of OOF.
- The isotonic calibrator was being fit on training predictions instead of OOF.

Fixing just the leakage bugs moved CV by about 30 basis points. The audit also gave me much more confidence in the final number, which mattered more than the score itself.

## How to reproduce

The Kaggle competition data is not redistributed in this repo (the competition license restricts redistribution). To run:

```bash
pip install -r requirements.txt
# Download the 35 CSV files from
#   https://www.kaggle.com/competitions/march-machine-learning-mania-2026
# Put them at the project root.
python -m src.pipeline                  # full pipeline, ~40min with TabPFN
python -m src.pipeline --evaluate-only  # CV only, no submission, ~5min
python -m src.test_all_phases           # smoke tests across all phases
```

The final submission for the live competition is checked in as `submissions/submission_v1.csv`.

## Results

| Metric | Value |
|---|---|
| Final rank | **65 of 3,462 teams (12,252 entrants)** - top 2% |
| Final Brier (private leaderboard) | **0.1229** |
| 1st place Brier | 0.1097 |
| Gap to 1st | 0.013 |
| OOF CV Brier, equal weights | 0.1669 |
| OOF CV Brier, optimized weights | 0.1660 |
| Best CV fold | 0.1280 (2025) |
| Worst CV fold | 0.1911 (2023) |
| Training rows (post game-flip) | 4,820 |
| Predictions generated | 132,133 |

The 0.013 gap to 1st sounds small, but in this competition it represents fundamentally better information - almost certainly Vegas closing lines, paid analytics services, or hand-tuned bracket adjustments. Not closeable in a week with public data alone.

## What I'd do differently next year

1. **Vegas closing lines.** The top of the leaderboard has them. Closing lines encode sharp money, injury reports, and information no public model has access to. Probably most of the 0.013 gap.
2. **Player-level data.** Roster changes, transfers, season-ending injuries. None of this is in the Kaggle bundle.
3. **A stacked logistic meta-learner** on OOF predictions, instead of just convex weight optimization. With seven moderately-diverse base models a real second-stage model should outperform weighted averaging.
4. **Bayesian search across the full ensemble**, not per-model.
5. **Start more than a week before the deadline.** Top finishers have months.

## Tech stack

Python, NumPy, pandas, scikit-learn, XGBoost, LightGBM, CatBoost, TabPFN, Optuna, trueskill, glicko2, scipy.

- Hemanth
