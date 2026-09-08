# Experimental protocol

## Scope

MotifShift is a synthetic diagnostic suite. The generators deliberately contain
recurring motifs, and representation bins and predictor parameters are hand chosen.
Results describe these particular settings; they are not evidence of broad
real-world superiority or sufficient conditions for general-purpose learning.
There is no pretrained model, large training job, or learned policy in the suite.

## Data and representations

**Trajectories.** Scenes contain multiple agents with recurring, non-identical
motion motifs, variation between scenes, and noise. Position tokens use an absolute
grid. Displacement and motion tokens summarize within-agent transitions. Situation
motion adds crowding context from agents in the same scene and time step. Motion
representations have one fewer token per trajectory than position representations.
Diagnostic sequence pairs never cross agent or scene boundaries.

**Gridworld replay.** Worlds contain reusable local geometries and noisy goal-directed
replay episodes. Raw tokens include the layout ID, position, and action. Other
representations use local walls or discrete situation categories. Situation-transition
tokens describe an observed action and its result. Episode boundaries are preserved
in sequence metrics and next-token fitting/scoring. This is replay analysis, not
policy optimization or a control benchmark.

**Scalar fields.** Synthetic bounded scalar fields evolve with advection/diffusion-like
updates. Tokens describe coordinates and value, local patches, value/change, or a
coarse regime. Bigram diagnostics compare successive transitions at the **same cell**;
they do not treat raster-neighbor cells as consecutive times. Diagnostic tokens that
include change describe a completed transition and may use its endpoint. Predictive
token features instead substitute the previous observed change; the initial frame
uses zero when no previous observation exists. Held-out next values and future
changes are excluded from predictor inputs.

Token bin definitions live in `motifshift/representations/` and
`motifshift/physical/representations.py`. They are fixed in code, not learned on
held-out data. Changes to these definitions change the experiment.

## Splits and predictors

Each evaluation holds out one complete scene, layout, or environment. Training uses
all other contexts; no fitted lookup, global fallback, or nearest-neighbor model
is shared across folds. There is no test-set hyperparameter search in the runner.

Trajectory predictors compare constant velocity, mean displacement, raw-coordinate
k-nearest neighbors, motion-token means, and situation/motion-token means. They
predict one future displacement using current/previous observed context. Errors
are Euclidean displacement errors in synthetic coordinate units.

Gridworld transfer predicts the most frequent next token conditional on the current
token. Unseen tokens use a global training fallback. Accuracy is meaningful within
a representation, but different representations define different prediction targets.
Situation transitions are descriptive replay tokens with overlapping temporal
information; predicting them is not equivalent to forecasting actions from raw observations.

Physical-field predictors compare the training mean next value and token-conditioned
means. Unseen tokens use a training-only global fallback. Errors are absolute scalar
value errors. Inputs use observations available at time t to predict time t+1.

Raw RL tokens contain a context identifier, so zero vocabulary overlap on unseen
layouts is guaranteed by construction. This is a diagnostic contrast, not an empirical
discovery of a failure shared by all raw representations.

## Metrics

| Metric | Definition and interpretation |
| --- | --- |
| Vocabulary size | Number of distinct observed tokens |
| Effective vocabulary size | `exp(H(p))`, with natural-log empirical token entropy |
| Normalized effective vocabulary | Effective vocabulary divided by token count; sensitive to dataset size |
| Top-k coverage | Fraction of observations in the k most frequent tokens; domain tables prefer k=50 if configured, otherwise the largest configured k |
| Token coverage | Fraction of held-out token observations whose token appeared in training |
| Unique-token coverage | Fraction of held-out token types seen in training |
| Top-k Jaccard | Set overlap of the most frequent train/test token types; count ties follow first occurrence |
| Jensen–Shannon divergence | Symmetric divergence of empirical token distributions, base 2, in [0, 1] |
| Bigram coverage | Fraction of valid held-out adjacent token pairs observed within a training sequence |
| Novel recombination rate | Fraction of held-out pairs absent in training, conditional on both component tokens having appeared in training |

A sequence with no valid pairs contributes no pairs; pair metrics return zero if
there are no eligible pairs. Zero novel recombination can therefore mean that no
eligible recombinations exist, not that all combinations were memorized.
Coverage and divergence alone do not measure semantic usefulness. Coarser tokens
can trivially improve reuse while discarding information. Entropy, sample size,
sequence length, and predictive task difficulty must be considered together.

## Aggregation and uncertainty

Context-level metrics are averaged with equal weight per held-out context.
Standard deviations use NumPy's population convention (`ddof=0`). For multi-seed
runs, each seed first produces a context mean; the final mean/std is calculated over
those seed means, retaining all seed outputs. These summaries are neither confidence
intervals nor significance tests. Folds share training contexts and are correlated.
The small configured seed set does not establish statistical power.

The JSON files preserve full floating-point results; tables round for readability.
Figures summarize means and should be interpreted alongside JSON standard deviations.
