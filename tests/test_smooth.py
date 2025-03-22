import numpy as np

import neuroboros as nb


class TestSmooth:
    def test_zero_smooth(self):
        lh = np.random.standard_normal((10, 10242))
        rh = np.random.standard_normal((10, 10242))
        dm = np.concatenate([lh, rh], axis=1)
        for keep_sum in [True, False]:
            lh_smooth = nb.smooth(
                "l", 0, space="onavg-ico32", mask=None, keep_sum=keep_sum
            )
            rh_smooth = nb.smooth(
                "r", 0, space="onavg-ico32", mask=None, keep_sum=keep_sum
            )
            smoothed = np.concatenate([lh @ lh_smooth, rh @ rh_smooth], axis=1)
            np.testing.assert_array_equal(smoothed, dm)
        for keep_sum in [True, False]:
            smooth = nb.smooth(
                "lr", 0, space="onavg-ico32", mask=None, keep_sum=keep_sum
            )
            smoothed = dm @ smooth
            np.testing.assert_array_equal(smoothed, dm)

    def test_smooth_lr(self):
        lh = np.random.standard_normal((10, 10242))
        rh = np.random.standard_normal((10, 10242))
        dm = np.concatenate([lh, rh], axis=1)
        for keep_sum in [True, False]:
            smooth = nb.smooth(
                "lr", 10, space="onavg-ico32", mask=None, keep_sum=keep_sum
            )
            smoothed_lh = lh @ nb.smooth(
                "l", 10, space="onavg-ico32", mask=None, keep_sum=keep_sum
            )
            smoothed_rh = rh @ nb.smooth(
                "r", 10, space="onavg-ico32", mask=None, keep_sum=keep_sum
            )
            smoothed = dm @ smooth
            np.testing.assert_array_equal(
                smoothed, np.concatenate([smoothed_lh, smoothed_rh], axis=1)
            )
            if keep_sum:
                np.testing.assert_allclose(np.sum(smoothed, axis=1), np.sum(dm, axis=1))
