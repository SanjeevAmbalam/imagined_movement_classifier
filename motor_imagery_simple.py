from pathlib import Path
import argparse
import json

import mne
import numpy as np
import pandas as pd
from mne.datasets import eegbci
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parent
SUBJECTS = list(range(1, 11))
RUNS = [4, 8, 12]  
TARGET = 0.75


def find_recording(subject, run, download=False):
    filename = f'S{subject:03d}R{run:02d}.edf'
    relative_path = Path('MNE-eegbci-data/files/eegmmidb/1.0.0')
    relative_path = relative_path / f'S{subject:03d}' / filename

    for folder in [PROJECT / '.data', Path.home() / 'mne_data']:
        path = folder / relative_path
        if path.exists():
            return path

    if not download:
        raise FileNotFoundError(f'{filename} is missing. Run with --download.')

    files = eegbci.load_data(subject, [run], path=str(PROJECT / '.data'),
                             update_path=False, verbose=False)
    return Path(files[0])


def make_features(signals, sampling_rate):
    # signals has shape: number of trials, number of electrodes, time samples.
    # Subtract the average electrode voltage at each time point.
    signals = signals - signals.mean(axis=1, keepdims=True)

    samples_per_block = int(sampling_rate * 0.5)
    trials, channels, samples = signals.shape
    if samples % samples_per_block != 0:
        raise ValueError('Each trial must contain complete half-second blocks.')

    blocks = samples // samples_per_block
    signals = signals.reshape(trials, channels, blocks, samples_per_block)
    averages = signals.mean(axis=3)

    # At 160 Hz: 64 electrodes x 7 time blocks = 448 numbers per trial.
    return averages.reshape(trials, channels * blocks)


def load_subject(subject, download=False, start_time=0.5):
    feature_parts = []
    labels = []
    run_numbers = []

    for run in RUNS:
        path = find_recording(subject, run, download)
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        raw.pick('eeg')
        sampling_rate = raw.info['sfreq']
        events, _ = mne.events_from_annotations(
            raw, event_id={'T1': 0, 'T2': 1}, verbose=False)

        trials = []
        for event in events:
            start = event[0] + int(start_time * sampling_rate)
            stop = start + int(3.5 * sampling_rate)
            if start < 0 or stop > raw.n_times:
                raise ValueError('A trial extends outside its recording.')
            trials.append(raw.get_data(start=start, stop=stop) * 1e6)

        features = make_features(np.array(trials), sampling_rate)
        feature_parts.append(features)
        labels.extend(events[:, 2])
        run_numbers.extend([run] * len(events))

    return np.concatenate(feature_parts), np.array(labels), np.array(run_numbers)


def make_model():
    """Scale the features, then fit a strongly regularized linear classifier."""
    # mean and spread are learned through training trials, c limits overfitting when lack of trials
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.01, max_iter=1000),
    )


def evaluate_subject(features, labels, run_numbers):
    probabilities = np.zeros(len(labels))
    for test_run in RUNS:
        train = run_numbers != test_run
        test = run_numbers == test_run
        model = make_model()
        model.fit(features[train], labels[train])
        probabilities[test] = model.predict_proba(features[test])[:, 1]
    return probabilities


def run_evaluation(subjects=SUBJECTS, download=False, output=None):
    subjects = list(subjects)
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError('Use a nonempty list of different subject numbers.')
    mne.set_log_level('ERROR')
    output = Path(output or PROJECT / 'results' / 'imagery_simple')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text('{"complete": false}', encoding='utf-8')
    rows = []

    with threadpool_limits(limits=1):
        for subject in subjects:
            features, labels, runs = load_subject(subject, download)
            probabilities = evaluate_subject(features, labels, runs)
            predictions = (probabilities >= 0.5).astype(int)
            print(f'Subject {subject:2d}: {accuracy_score(labels, predictions):.2%}', flush=True)

            for trial in range(len(labels)):
                rows.append({'subject': subject, 'run': int(runs[trial]),
                             'trial': trial, 'truth': int(labels[trial]),
                             'prediction': int(predictions[trial]),
                             'probability_right': float(probabilities[trial])})

    results = pd.DataFrame(rows)
    results['correct'] = results['truth'] == results['prediction']
    results.to_csv(output / 'predictions.csv', index=False)
    by_subject = results.groupby('subject')['correct'].agg(['sum', 'count', 'mean'])
    by_subject.columns = ['correct', 'trials', 'accuracy']
    by_subject.to_csv(output / 'subjects.csv')
    accuracy = float(results['correct'].mean())
    summary = {'complete': True, 'task': 'imagined left versus right fist',
               'evaluation': 'leave one whole recording run out per subject',
               'subjects': subjects, 'trials': len(results),
               'correct': int(results['correct'].sum()), 'accuracy': accuracy,
               'target': TARGET, 'target_met': accuracy >= TARGET,
               'status': 'development cross-validation; fresh validation still needed'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\nOverall: {summary["correct"]}/{len(results)} correct = {accuracy:.2%}')
    print(f'75% overall target met: {summary["target_met"]}')
    print('Confusion matrix (rows=true, columns=predicted; left then right):')
    print(confusion_matrix(results['truth'], results['prediction'], labels=[0, 1]))
    return by_subject, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subjects', nargs='+', type=int, default=SUBJECTS)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--output')
    arguments = parser.parse_args()
    run_evaluation(arguments.subjects, arguments.download, arguments.output)
