# Investigation results

## Outcome

The final simple model correctly classifies **345 of 450 imagined left/right
fist trials (76.67%)**, exceeding the requested 75% overall target in development
cross-validation. All ten subjects are retained. No actual-movement training
data is used. The full implementation is `motor_imagery_simple.py`.

The classifier is StandardScaler followed by LogisticRegression(C=0.01).
Features are the mean voltages of 64 average-referenced electrodes in seven
half-second blocks from 0.5 to 4.0 seconds after the cue. There is no continuous
band-pass filter, so nearby trials cannot enter the feature window.

## Why the approach changed

The original model mostly used EEG near cue onset (-0.25 to 0.5 seconds).
Extending the window helped, but increasingly elaborate motor-band covariance
models still fell short. Longer voltage averages retain information discarded
by the 7-30 Hz power-based models. A strongly regularized linear classifier
can learn that information with only 30 training trials per fold.

This is an empirical finding about this dataset. It does not identify the
physiological source of the useful voltage changes.

## Comparisons

Every reported model below scores all 450 trials with complete recording runs
held out. The extra-subject pooling experiments also borrow other subjects'
training data; they are not identical in training scope to the final model.
The table is an exploratory comparison, not a collection of independent final
test results. Best individual configurations were observed after comparisons.

| Approach | Accuracy |
| --- | ---: |
| Original short-window CSP/LDA | 58.22% |
| First nested CSP/tangent selection | 63.56% |
| Band power with average reference or Laplacian, best fixed model | 64.89% |
| Expanded fixed CSP ensemble | 68.67% |
| CSP with patterns borrowed from other subjects, fixed ensemble | 71.11% |
| Shared tangent-space classifier, best fixed model | 67.33% |
| Continuous low-frequency filtered voltage, best fixed model | 75.33% |
| Shared low-frequency classifier, nested selection | 79.33% |
| **Unfiltered within-trial voltage + logistic regression** | **76.67%** |
| Within-trial voltage/CSP combined ensemble | 77.33% |

Historical scores are summarized in `results/development_comparison.csv`.
Exploratory implementations and large intermediate outputs were removed during
cleanup. The matched baseline predictions and all final-model predictions and
controls remain in `results/imagery_simple/`. Only the final model and its
controls are maintained as reproducible code in this cleaned project.

The continuous low-frequency approaches were not retained: a 0.1 Hz FIR filter
extends well outside the target trial and can blend adjacent cues. Their
higher scores are not sufficient justification for using them. The final
model uses only samples inside the current trial.

The combined within-trial model averages twelve classifiers. Its gain over
one logistic-regression classifier is three correct trials, so the simpler
model better matches the requested coding style.

## Per-subject results

| Subject | Correct / 45 | Accuracy |
| --- | ---: | ---: |
| 1 | 37 | 82.22% |
| 2 | 36 | 80.00% |
| 3 | 39 | 86.67% |
| 4 | 37 | 82.22% |
| 5 | 26 | 57.78% |
| 6 | 27 | 60.00% |
| 7 | 38 | 84.44% |
| 8 | 35 | 77.78% |
| 9 | 42 | 93.33% |
| 10 | 28 | 62.22% |

Overall success does not mean every person reaches 75%. A person contributes
45 test predictions, each made after training a fresh classifier on that
person's other 30 trials. These are not unseen-person results.

Confusion matrix (rows=true, columns=predicted; left then right):

```
185  45
 60 160
```

## Integrity checks and controls

The source audit found 450 valid imagery trials: 230 left and 220 right,
64 channels, 160 Hz, and 4.1-second task annotations. None of the published
malformed-record subject exclusions apply to subjects 1-10. Predicting left
on every trial would score 51.11%.

Three unit checks pass: changing another trial cannot change a trial's
features; a common voltage offset across electrodes is removed; changing
held-out labels cannot change predictions for that held-out run. The verification
script reproduces the final model's saved probabilities, and all
450 saved predictions are unique and reconstruct the reported score.

Matching final-model controls:

| Control | Accuracy |
| --- | ---: |
| Actual post-cue, all electrodes | 76.67% |
| Actual post-cue, 17 predefined motor-region electrodes | 70.22% |
| Entirely before-cue EEG (-4.0 to -0.5 seconds) | 56.44% |
| Mean of 200 within-run shuffled-label evaluations | 50.30% |
| Highest of the 200 shuffled-label evaluations | 57.78% |

No shuffled-label evaluation reached the real score (finite-permutation
p=1/201, about 0.005). This test conditions on the final feature design; it
**does not account for the preceding model search**. The before-cue result
is above 50%, so some recording/schedule-related predictability may exist
before motor imagery begins.

An audit found opposite-class pairs in the experiment's trial sequence.
Using the previous *true* label could give around 76% without reading EEG.
That shortcut is explicitly excluded: trial number, neighboring labels and
cue codes are never input features. This schedule finding is historical audit evidence, not a model feature.

## Interpretation and uncertainty

The [official dataset protocol](https://physionet.org/content/eegmmidb/1.0.0/)
uses a visual target on the corresponding side of the screen. Visual responses
and eye movements can therefore correlate with the left/right labels.
The motor-only and before-cue controls do not establish a purely motor source.
Describe the result as **classifying cued imagined-movement EEG recordings**,
not as demonstrated thought-only control or a validated real-time BCI.

We investigated multiple methods on the same available recordings. Although
each fold excludes its test run from scaling and training, the final design
was selected using development results. An independent recording session
with a frozen model design is needed for confirmation. To test a thought-only
claim, use cues that do not reveal left/right through gaze and record eye
activity for artifact assessment.

A rough bootstrap interval, resampling the ten subject accuracies rather than
pretending every trial is independent, is **69.11%-83.56%** (10,000 resamples,
seed 42 after the permutation sequence). Thus the measured score exceeds 75%,
but these data do not establish that the underlying accuracy is at least 75%
with 95% confidence. Control outputs and parameters are saved in
`results/imagery_simple/verification.json`.

Additional primary-source background:
[PhysioNet data curation](https://pmc.ncbi.nlm.nih.gov/articles/PMC10998040/)
and [processing/artifact study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11674661/).
