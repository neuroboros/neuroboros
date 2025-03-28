import numpy as np

import neuroboros as nb

spaces = [
    "onavg-ico32",
    "onavg-ico48",
    "onavg-ico64",
    "fsavg-ico32",
    "fsavg-ico64",
    "fslr-ico32",
    "fslr-ico64",
    "fslr-ico57",
]


class TestMapping:
    def test_no_mask(self):
        for s1 in spaces:
            for s2 in spaces:
                for lr in "lr":
                    xfm = nb.mapping(lr, s1, s2)
                    ico1 = int(s1.split("-ico")[1])
                    ico2 = int(s2.split("-ico")[1])
                    nv1 = ico1**2 * 10 + 2
                    nv2 = ico2**2 * 10 + 2
                    assert xfm.shape == (nv1, nv2)

    def test_mask(self):
        for s1 in spaces:
            for s2 in spaces:
                for lr in "lr":
                    xfm = nb.mapping(lr, s1, s2, mask=True)
                    nv1 = nb.mask(lr, s1).sum()
                    nv2 = nb.mask(lr, s2).sum()
                    assert xfm.shape == (nv1, nv2)

    def test_lr(self):
        rng = np.random.default_rng(0)
        for s1 in spaces:
            for s2 in spaces:
                xfm = nb.mapping("lr", s1, s2)
                xfm_l = nb.mapping("l", s1, s2)
                xfm_r = nb.mapping("r", s1, s2)
                lh = rng.standard_normal((10, xfm_l.shape[0]))
                rh = rng.standard_normal((10, xfm_r.shape[0]))
                dm = np.concatenate([lh, rh], axis=1)
                mapped = dm @ xfm
                mapped_l = lh @ xfm_l
                mapped_r = rh @ xfm_r
                # Not sure why not identical, but diff is often < 1e-10
                np.testing.assert_allclose(
                    mapped,
                    np.concatenate([mapped_l, mapped_r], axis=1),
                    atol=1e-6,
                )
