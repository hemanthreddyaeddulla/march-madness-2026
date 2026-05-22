# How I Finished 65 of 3,462 Teams (12,252 Entrants) in My First Kaggle Competition

## March Machine Learning Mania 2026 - Methodology Notes

**Final rank:** 65 of 3,462 teams (12,252 entrants) - top 2%
**Final Brier score:** 0.1229
**1st place score:** 0.1097
**Gap to 1st:** ~0.013

This is a write-up of how I approached predicting the NCAA Men's and Women's basketball tournaments. I'm intentionally not giving away every implementation detail - partly because I want to use this approach again next year, and partly because half the fun of this competition is figuring things out yourself. But I'll share enough that you can reproduce the spirit of what I did, and hopefully save yourself some of the dead ends I ran into.

---

## What this competition actually is

You're given ~35 CSV files of historical NCAA basketball data going back to 1985 (men) and 1998 (women). You have to predict the probability that the team with the lower TeamID beats the team with the higher TeamID, for every possible matchup in the 2026 tournament. That's around 132,000 predictions.

You're scored on **Brier score** - basically mean squared error between your predicted probability and the actual outcome (0 or 1). Lower is better. Only the games that actually happen in the tournament count toward your score, so out of 132K predictions only ~134 actually matter.

Two things make this brutal:
1. The metric punishes confident wrong predictions heavily, but rewards bold correct ones.
2. The whole tournament is decided in 6 rounds across 3 weekends. **Luck dominates skill.** A single bracket-busting upset can swing you 100 spots.

---

## The mental model

Most beginners (including me, on day 1) think of this as "build a model that predicts winners." That's wrong. You're not predicting winners - you're producing **calibrated probabilities**. A 12-seed beating a 5-seed happens roughly 35% of the time historically. Your model should *say* 35% on those games. If your model says 15% and a 12-seed wins, you eat a 0.72 Brier hit on a single game.

This shifted my entire approach. Instead of "make XGBoost more accurate," I started obsessing about **calibration** - does my model output 35% on games where the underlying truth is actually 35%?

---

## The pipeline at a high level

I built it in 7 modules. I'm sharing the structure but keeping some specific design choices private.

```
data_loader.py    - Load all 35 CSVs into one container
data_cleaner.py   - Pace adjust, handle COVID, outliers, augmentation
ratings.py        - Multiple independent rating systems
features.py       - Difference features for every matchup
model.py          - Stacked ensemble + calibration
generate_submission.py
pipeline.py       - Orchestrator
```

The whole thing runs end-to-end in roughly 30-40 minutes on a laptop with a GPU.

---

## Things that actually moved the needle

### 1. Don't use just one rating system

Everyone uses Elo. I used Elo *and* a few other rating systems that capture team strength differently. Each one views the same games through a slightly different mathematical lens. When I checked the correlations between them, they were all in the 0.93-0.97 range - strongly related, but not identical. That residual disagreement is exactly the diversity an ensemble needs.

The trick is **not** stacking 4 ratings that all measure the same thing the same way. You want ratings that disagree about the marginal cases. Two of mine handle uncertainty very differently (one tracks how "confident" the rating is), and that turned out to matter for early-round upset prediction.

### 2. Pace-adjust everything

Raw box score stats are basically useless without pace adjustment. A team that scores 80 points in 65 possessions is doing something completely different than a team scoring 80 in 80 possessions. I converted every counting stat to per-100-possessions. This is non-negotiable - if you're not doing this, you're modeling tempo, not skill.

I spent more time getting the possession formula right than on any modeling decision.

### 3. Game-flip augmentation

Every game in your training set is "Team A beat Team B." But the model needs to learn the symmetric relationship - if A beats B with probability p, then B beats A with probability 1-p. So for every training row, I created a mirror row with all difference features negated and the label flipped. This doubled my training data for free and forced symmetric predictions.

The only thing to be careful about: features that are **products** of two negated quantities don't actually flip (because (-a)(-b) = ab). I had a bug here for about an hour where I was negating features that shouldn't be negated.

### 4. Out-of-fold everything

This was the biggest mindset shift. I originally optimized my ensemble weights by checking which combination produced the lowest Brier score on the test fold. That's data leakage - you're tuning a hyperparameter on the data you're evaluating on.

The fix: train each base model with expanding-window cross-validation, collect each model's predictions on its respective held-out fold (these are "out-of-fold" predictions), and then optimize the ensemble weights on those OOF predictions. The OOF predictions are by definition predictions the model never saw the labels for, so optimizing weights on them is honest.

This single fix increased my reported CV Brier from 0.1628 to 0.1660. It looked worse, but it was the **honest** number - and the actual leaderboard score ended up at 0.1229, which is right in line with the OOF estimate. The previous "0.1628" was a lie I was telling myself.

### 5. Calibrate separately by gender

The women's tournament is dramatically more chalky than the men's. Women's 1-seeds have lost to 16-seeds exactly once in history. Men's 1-seeds lose to 16-seeds about once a decade. If you train one model and one calibrator on combined data, you'll be too cautious on women's games (giving favorites only 90% when 99% is correct) and too confident on men's games.

I trained one model on combined data (with a gender flag as a feature), but ran the calibration step **separately** for each gender. The men's calibrator gets to see only men's predictions vs men's outcomes, and the women's calibrator likewise. I also used different probability clipping bounds for each gender - wider for women (allowing more extreme predictions) and narrower for men.

### 6. Diversify your tree models with different feature subsets

I had three gradient-boosted tree models in my ensemble (XGBoost, LightGBM, CatBoost). Initially they all got trained on the same 39 features and they all made nearly-identical predictions. The ensemble weight optimizer correctly identified that LightGBM and CatBoost were redundant with XGBoost, and gave them 0% weight.

The fix: I gave each model a **different feature subset**. XGBoost saw everything (the generalist). LightGBM saw only rating-system features. CatBoost saw only performance-based features. Now they actually disagreed on hard cases, and the ensemble started getting real value from each one.

### 7. The pre-trained transformer

There's a model called TabPFN which is a transformer pre-trained on millions of synthetic tabular datasets. You give it your data and it just... works. No tuning. No hyperparameters. For datasets under 10K rows it routinely beats hand-tuned XGBoost.

When I added it to my ensemble, the weight optimizer assigned it 40-52% of the total weight. It's the single highest-leverage model in my stack.

### 8. A "seed baseline" model

In addition to the fancy stuff, I included a stupidly simple model: a lookup table that says "given the seed difference between two teams, what's the historical win rate?" This contributes nothing on most games - but on extreme matchups (1 vs 16, 8 vs 9), it acts as a strong anchor that pulls the ensemble back toward base rates when the fancy models get overconfident.

It got 8-10% weight in the final ensemble, which is more than I expected.

---

## Things I tried that didn't work

I'm including these because reading "what worked" lists from competition winners is misleading - every winner also tried 20 things that failed. Here are some of mine.

- **LSTM on game sequences.** The idea was to feed each team's last 20 games as a time series. I researched this for hours, found a published paper claiming Brier 0.159 with this approach, and... decided not to build it. The expected improvement (0.001-0.003) wasn't worth the implementation cost (6-8 hours), especially with high overfitting risk on ~4,800 training rows. Sometimes the right answer is "this paper is impressive but the marginal value for me is small."

- **External Barttorvik data.** I integrated free Barttorvik adjusted efficiency data. Turns out there's a train/test mismatch problem - the data is only reliably available for 2024+ but I train on 2003-2025. My tree models learned to ignore the external columns (NaN for 95% of training data) and gained no benefit. I removed it. **Lesson: external data is only useful if it's available across your entire training window.**

- **Optuna hyperparameter tuning of XGBoost.** Marginal improvement (~0.001), big increase in pipeline time. Kept it because every basis point matters at the top of the leaderboard, but be honest with yourself about what it's worth.

- **Probability clipping at [0.02, 0.98].** I started here and discovered later that women's games benefit from looser bounds. Tightening too much costs you on the games where the model actually IS confident.

---

## The audit

Before submitting, I did a line-by-line audit of every source file looking for bugs. I found **12 bugs**, including:
- A duplicate feature (two columns containing the exact same data)
- An inverted sign convention on defensive efficiency
- Home court advantage being applied to both teams (effectively doubling it)
- Coach experience features that leaked future seasons
- Calibration fitted on training predictions instead of OOF predictions
- A weight assertion that conflicted with my new gender-specific clipping bounds

Fixing the data leakage bugs alone moved my "honest" CV from a falsely-optimistic 0.1628 to a real 0.1660. I'd rather know the true number.

**If you're competing, do this audit. The bugs you don't find are the ones that cost you positions.**

---

## What I'd do differently next year

1. **Vegas closing lines.** I didn't use them. The top of the leaderboard almost certainly does. Closing lines incorporate sharp money, injury reports, and information no public model has.
2. **Player-level data.** Roster changes, transfers, injuries - none of which are in the Kaggle dataset.
3. **Bayesian hyperparameter search across the whole ensemble**, not just one model.
4. **Build a proper meta-learner** instead of just optimizing weights. With more model diversity, a stacked logistic regression on OOF predictions should outperform weighted averaging.
5. **Start earlier.** I started this with about a week to go. Top finishers have months.

---

## The honest takeaway

I finished 65 of 3,462 teams (12,252 entrants) - top 2% - in my first Kaggle competition, with a Brier score of 0.1229. The 1st place score was 0.1097.

That gap of ~0.013 sounds tiny, but in this competition it represents **fundamentally better information** - almost certainly Vegas line data, paid analytics services, or hand-tuned bracket adjustments. It's not closeable in a week with public data alone.

What I learned:
- **Honest validation matters more than clever models.** Most of my Brier improvement came from finding and removing data leakage, not from adding more models.
- **Diversity in an ensemble is everything.** Five correlated models is just one model.
- **Calibration > accuracy.** The Brier metric rewards honest probabilities, not confident guesses.
- **Domain knowledge pays.** Pace adjustment, gender-specific calibration, and the seed baseline model all came from understanding basketball, not from understanding ML.

If you want the full implementation details - the specific rating system parameters, which features I used, the exact ensemble structure, the bug list, the things I'm still tuning for next year - drop a comment with **"siuu"** on the LinkedIn post and I'll DM you the full notes.

Good luck on March Madness 2027.

- Hemanth
