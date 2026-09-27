# Imagined left versus right movement

A simple EEG classifier with **76.67% development accuracy: 345/450 trials**
across ten subjects. Each person is tested on complete recording runs excluded
from training. The target is 75% overall, not 75% for every person.

Read [motor_imagery_simple.py](motor_imagery_simple.py) for the commented code,
or open the executed [notebook](motor_imagery.ipynb) for a walkthrough.

## Setup and run

Use Python 3.12. With [uv](https://docs.astral.sh/uv/):

```powershell
uv venv .venv-accuracy --python 3.12
uv pip install --python .venv-accuracy/Scripts/python.exe -r requirements-dev.txt
.\run_accuracy.cmd --download
```

The first run downloads missing public EEG files from PhysioNet. Later runs
can omit `--download`. Downloaded recordings and the Python environment are
local files excluded from Git. For the command-line model without notebook
tools, install `requirements.txt` instead of `requirements-dev.txt`.

On macOS/Linux, use `.venv-accuracy/bin/python` instead of the Windows path:

```bash
uv venv .venv-accuracy --python 3.12
uv pip install --python .venv-accuracy/bin/python -r requirements-dev.txt
.venv-accuracy/bin/python motor_imagery_simple.py --download
```

In VS Code, select this environment as the notebook kernel. The model reads
recordings from `.data` or `~/mne_data`. In this existing workspace the working
environment is already installed; run `.\run_accuracy.cmd`.

## Method

1. Read imagined left/right fist runs 4, 8 and 12 for subjects 1-10.
2. Take 0.5-4.0 seconds of EEG from each trial and subtract the mean across
   electrodes at each time point.
3. Average each electrode in seven half-second blocks: 64 x 7 = 448 features.
4. Fit a training-only scaler and logistic regression with `C=0.01`.
5. Train on two runs, test on the third, and repeat. Every trial is tested once.

No actual-movement recordings, neighboring-trial features, previous labels or
trial-order features are used. No continuous filter mixes nearby trials.

## Results and limitations

The final model scores **76.67%**, versus **58.22%** for the original model on
matched run splits. Three data-isolation tests pass. Two hundred label-shuffle
controls average 50.30%. Full results, historical comparisons and uncertainty
are in [RESULTS.md](RESULTS.md).

These are development cross-validation results: the model was chosen after
comparing approaches on these recordings. Fresh recordings are needed to
confirm generalization. The dataset uses visual cues, so eye-related or visual
activity may contribute; this is not proof of a thought-only BCI. Motor-only
electrodes score 70.22%, and before-cue EEG scores 56.44%.

The [official PhysioNet dataset](https://physionet.org/content/eegmmidb/1.0.0/)
provides the recordings and describes the protocol. Dataset citation:
Schalk, G. (2009), *EEG Motor Movement/Imagery Dataset*, version 1.0.0,
DOI: 10.13026/C28G6P. Raw recordings are not redistributed in this repository.

## Project files

- `motor_imagery_simple.py`: complete classifier and evaluation.
- `motor_imagery.ipynb`: executed explanation and chart.
- `test_motor_imagery_simple.py`: tests for data isolation.
- `verify_results.py`: prediction reproduction and negative controls.
- `results/imagery_simple/`: final predictions, scores, chart and controls.
- `results/development_comparison.csv`: compact historical experiment summary.

Old model implementations, duplicate backups, notebook generators and
exploratory feature caches are excluded from this repository. In the original
workspace they are recoverable from the ignored `.cleanup-archive/` folder.
The final model and its controls remain reproducible.

## Verify

```powershell
$env:_MNE_FAKE_HOME_DIR = "$PWD"
.venv-accuracy/Scripts/python.exe -m unittest test_motor_imagery_simple.py
.venv-accuracy/Scripts/python.exe verify_results.py
```

Run the model first if `predictions.csv` is missing. The verification script
reproduces its saved probabilities, then runs before-cue, motor-channel and
200 shuffled-label controls. It does not depend on deleted experiment files.
