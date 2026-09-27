"""Small checks for the important data-isolation rules."""
import unittest
import numpy as np
from motor_imagery_simple import make_features, evaluate_subject


class SimpleModelTests(unittest.TestCase):
    def test_features_do_not_mix_trials(self):
        random = np.random.default_rng(42)
        signals = random.normal(size=(2, 64, 560))
        original = make_features(signals, 160)
        signals[1] *= 1000
        changed = make_features(signals, 160)
        self.assertEqual(original.shape, (2, 448))
        np.testing.assert_array_equal(original[0], changed[0])

    def test_shared_voltage_is_removed(self):
        random = np.random.default_rng(42)
        signals = random.normal(size=(2, 64, 560))
        common_voltage = random.normal(size=(2, 1, 560))
        np.testing.assert_allclose(make_features(signals, 160),
                                   make_features(signals + common_voltage, 160), atol=1e-12)

    def test_test_labels_cannot_change_their_predictions(self):
        random = np.random.default_rng(42)
        features = random.normal(size=(45, 12))
        labels = np.tile([0, 1], 23)[:45]
        runs = np.repeat([4, 8, 12], 15)
        original = evaluate_subject(features, labels, runs)
        changed_labels = labels.copy()
        changed_labels[runs == 12] = 1 - changed_labels[runs == 12]
        changed = evaluate_subject(features, changed_labels, runs)
        np.testing.assert_allclose(original[runs == 12], changed[runs == 12])


if __name__ == '__main__':
    unittest.main()
