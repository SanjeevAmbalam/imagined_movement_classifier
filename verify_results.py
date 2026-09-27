"""Negative controls for the final, frozen simple classifier."""
import json
import mne
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from motor_imagery_simple import load_subject, evaluate_subject, find_recording, PROJECT

MOTOR = ['FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1',
         'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4']


def main():
    mne.set_log_level('ERROR')
    output = PROJECT / 'results' / 'imagery_simple'
    original = pd.read_csv(output / 'predictions.csv')
    assert len(original) == 450
    assert not original.duplicated(['subject', 'trial']).any()
    summary = json.loads((output / 'summary.json').read_text())
    assert int((original.truth == original.prediction).sum()) == summary['correct']

    records, datasets = [], []
    with threadpool_limits(limits=1):
        for subject in range(1, 11):
            x, y, runs = load_subject(subject)
            datasets.append((x, y, runs))
            saved = original[original.subject == subject].sort_values('trial')
            reproduced = evaluate_subject(x, y, runs)
            np.testing.assert_array_equal(saved.truth, y)
            np.testing.assert_array_equal(saved.run, runs)
            np.testing.assert_array_equal(saved.prediction, reproduced >= .5)
            np.testing.assert_allclose(saved.probability_right, reproduced, atol=1e-10)
            before, before_y, before_runs = load_subject(subject, start_time=-4.)
            np.testing.assert_array_equal(y, before_y)
            np.testing.assert_array_equal(runs, before_runs)
            raw = mne.io.read_raw_edf(find_recording(subject, 4), verbose=False)
            mne.datasets.eegbci.standardize(raw)
            indices = [raw.ch_names.index(name) for name in MOTOR]
            motor = x.reshape(45, 64, 7)[:, indices].reshape(45, -1)
            for name, features in [('before_cue', before), ('motor_channels_only', motor)]:
                probabilities = evaluate_subject(features, y, runs)
                for trial, probability in enumerate(probabilities):
                    records.append(dict(subject=subject, control=name, trial=trial,
                                        truth=int(y[trial]), prediction=int(probability >= .5)))
        pd.DataFrame(records).to_csv(output / 'control_predictions.csv', index=False)
        random = np.random.default_rng(42)
        permutation_scores = []
        for repeat in range(200):
            correct = 0
            for x, y, runs in datasets:
                shuffled = y.copy()
                for run in [4, 8, 12]:
                    mask = runs == run
                    shuffled[mask] = random.permutation(y[mask])
                predictions = evaluate_subject(x, shuffled, runs) >= .5
                correct += int(np.sum(predictions == shuffled))
            permutation_scores.append(correct / 450)
            if (repeat + 1) % 25 == 0:
                print('Completed label-shuffle control', repeat + 1, 'of 200', flush=True)
    pd.DataFrame({'accuracy': permutation_scores}).to_csv(output / 'permutation_scores.csv', index=False)
    controls = pd.DataFrame(records)
    controls['correct'] = controls.truth == controls.prediction
    scores = controls.groupby('control').correct.mean().to_dict()
    results = dict(controls=scores, permutations=200, seed=42,
                   mean_shuffled_accuracy=float(np.mean(permutation_scores)),
                   maximum_shuffled_accuracy=float(np.max(permutation_scores)),
                   permutation_p=(1 + int(np.sum(np.array(permutation_scores) >= summary['accuracy']))) / 201,
                   caveat='Conditional on frozen features; does not correct for all exploratory model search.')
    subject_scores = original.groupby('subject').apply(
        lambda rows: (rows.truth == rows.prediction).mean(), include_groups=False).to_numpy()
    bootstraps = random.choice(subject_scores, size=(10000, 10), replace=True).mean(axis=1)
    results['subject_bootstrap_95_percent_interval'] = np.quantile(bootstraps, [.025, .975]).tolist()
    (output / 'verification.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
