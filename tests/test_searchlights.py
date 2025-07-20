import numpy as np

import neuroboros as nb


class TestSearchlights:
    def test_basic(self):
        for space in [_ + "-ico32" for _ in ["onavg", "fsavg", "fslr", "mkavg"]]:
            for lr in "lr":
                sls = nb.sls(lr, 20, space, mask=False)
                assert len(sls) == 10242
                assert all([isinstance(_, np.ndarray) for _ in sls])
                assert np.concatenate(sls).max() == 10241
                assert np.concatenate(sls).min() == 0

    def test_masked(self):
        for space in [_ + "-ico32" for _ in ["onavg", "fsavg", "fslr", "mkavg"]]:
            for lr in "lr":
                sls = nb.sls(lr, 20, space, mask=True)
                nv = nb.mask(lr, space).sum()
                assert len(sls) == nv
                assert all([isinstance(_, np.ndarray) for _ in sls])
                assert np.concatenate(sls).max() == nv - 1
                assert np.concatenate(sls).min() == 0

    def test_radius(self):
        for space in [_ + "-ico32" for _ in ["onavg", "fsavg", "fslr", "mkavg"]]:
            for lr in "lr":
                radius = 10
                dists = nb.sls(lr, radius, space, mask=False, return_dists=True)[1]
                assert all([np.all(_ <= radius) for _ in dists])
                dists = nb.sls(lr, radius, space, mask=True, return_dists=True)[1]
                assert all([np.all(_ <= radius) for _ in dists])

    def test_center_space(self):
        for space in [_ + "-ico64" for _ in ["onavg", "fsavg", "fslr"]]:
            center_space = space.replace("-ico64", "-ico32")
            for lr in "lr":
                sls = nb.sls(lr, 20, space, center_space=center_space, mask=False)
                assert len(sls) == 10242
                assert all([isinstance(_, np.ndarray) for _ in sls])
                assert np.concatenate(sls).max() == 40961
                assert np.concatenate(sls).min() == 0

    def test_center_mask(self):
        for space in [_ + "-ico64" for _ in ["onavg", "fsavg", "fslr"]]:
            center_space = space.replace("-ico64", "-ico32")
            for lr in "lr":
                sls = nb.sls(
                    lr,
                    20,
                    space,
                    center_space=center_space,
                    center_mask=True,
                    mask=False,
                )
                assert len(sls) == nb.mask(lr, center_space).sum()
                assert all([isinstance(_, np.ndarray) for _ in sls])
                assert np.concatenate(sls).min() == 0
                sls2 = nb.sls(
                    lr,
                    20,
                    space,
                    center_space=center_space,
                    center_mask=True,
                    mask=True,
                )
                assert len(sls2) == nb.mask(lr, center_space).sum()
                assert all([isinstance(_, np.ndarray) for _ in sls2])
                assert np.concatenate(sls2).min() == 0
                assert not all(
                    [np.array_equal(sl1, sl2) for sl1, sl2 in zip(sls, sls2)]
                )

        for space in [_ + "-ico64" for _ in ["onavg", "fsavg", "fslr"]]:
            center_space = space.replace("-ico64", "-ico32")
            for lr in "lr":
                sls = nb.sls(
                    lr,
                    20,
                    space,
                    center_space=center_space,
                    center_mask=False,
                    mask=True,
                )
                assert len(sls) == 10242
                assert all([isinstance(_, np.ndarray) for _ in sls])
                assert np.concatenate(sls).min() == 0
                sls2 = nb.sls(
                    lr,
                    20,
                    space,
                    center_space=center_space,
                    center_mask=False,
                    mask=False,
                )
                assert len(sls2) == 10242
                assert all([isinstance(_, np.ndarray) for _ in sls2])
                assert np.concatenate(sls2).min() == 0
                assert not all(
                    [np.array_equal(sl1, sl2) for sl1, sl2 in zip(sls, sls2)]
                )

    def test_lr(self):
        rng = np.random.default_rng(0)
        for space in [_ + "-ico32" for _ in ["onavg", "fsavg", "fslr", "mkavg"]]:
            lh = rng.standard_normal((10, 10242))
            rh = rng.standard_normal((10, 10242))
            dm = np.concatenate([lh, rh], axis=1)
            sls = nb.sls("lr", 20, space, mask=False)
            sls_l = nb.sls("l", 20, space, mask=False)
            sls_r = nb.sls("r", 20, space, mask=False)
            lh_m = np.stack([lh[:, sl].mean(axis=1) for sl in sls_l], axis=1)
            rh_m = np.stack([rh[:, sl].mean(axis=1) for sl in sls_r], axis=1)
            m = np.stack([dm[:, sl].mean(axis=1) for sl in sls], axis=1)
            np.testing.assert_allclose(
                m, np.concatenate([lh_m, rh_m], axis=1), atol=1e-6
            )

        for space in [_ + "-ico32" for _ in ["onavg", "fsavg", "fslr", "mkavg"]]:
            lh_nv = nb.mask("l", space).sum()
            rh_nv = nb.mask("r", space).sum()
            lh = rng.standard_normal((10, lh_nv))
            rh = rng.standard_normal((10, rh_nv))
            dm = np.concatenate([lh, rh], axis=1)
            sls = nb.sls("lr", 20, space, mask=True)
            sls_l = nb.sls("l", 20, space, mask=True)
            sls_r = nb.sls("r", 20, space, mask=True)
            lh_m = np.stack([lh[:, sl].mean(axis=1) for sl in sls_l], axis=1)
            rh_m = np.stack([rh[:, sl].mean(axis=1) for sl in sls_r], axis=1)
            m = np.stack([dm[:, sl].mean(axis=1) for sl in sls], axis=1)
            np.testing.assert_allclose(
                m, np.concatenate([lh_m, rh_m], axis=1), atol=1e-6
            )
