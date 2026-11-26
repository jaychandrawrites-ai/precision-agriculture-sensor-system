import unittest

import numpy as np

from Temporal.ForecastUncertainty import TrajectoryResiduals, ValidationBlend


class ValidationBlendTests(unittest.TestCase):
    def test_per_lead_feature_weights_and_fallback(self):
        targets = np.ones((9, 2, 2))
        forecasts = np.array([targets + [[0, 2], [3, 0]], targets + [[5, 0], [0, 4]]])
        model = ValidationBlend.fit(forecasts, targets)
        np.testing.assert_array_equal(model.weights.argmax(axis=-1), [[0, 1], [1, 0]])
        self.assertTrue((model.weights >= 0).all())
        np.testing.assert_allclose(model.weights.sum(axis=-1), 1.)
        forecasts[:, 0, 0, 0] = np.nan
        self.assertEqual(model.predict(forecasts, fallback=targets)[0, 0, 0], 1.)
        with self.assertRaisesRegex(ValueError, "fallback"):
            model.predict(forecasts)

    def test_residual_sampling_has_shared_site_feature_and_lead_blocks(self):
        point = np.zeros((2, 2, 2))
        forecast = np.zeros((8, *point.shape))
        observed = forecast + np.arange(8)[:, None, None, None]
        calibration = TrajectoryResiduals.calibrate(forecast, observed, np.arange(8))
        np.testing.assert_array_equal(calibration.retained_origins, [0, 2, 4, 6])
        draws, origins = calibration.sample(point, 30)
        np.testing.assert_array_equal(draws, np.broadcast_to(origins[:, None, None, None], draws.shape))
