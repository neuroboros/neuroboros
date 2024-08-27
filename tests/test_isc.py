import numpy as np

import neuroboros as nb


class TestISC:
    def test_compute_isc_pairwise(self):
        ns, nt, nv = 10, 100, 10
        rng = np.random.default_rng(0)
        dms = rng.standard_normal((ns, nt, nv))
        n_pairs = nv * (nv - 1) // 2
        for metric in ["correlation", "cosine"]:
            isc = nb.isc(dms, pairwise=True, metric=metric)
            assert isc.shape == (n_pairs, nv)
            assert np.all(isc >= -1) and np.all(isc <= 1)

    def test_compute_isc_ovr(self):
        ns, nt, nv = 10, 100, 10
        rng = np.random.default_rng(0)
        dms = rng.standard_normal((ns, nt, nv))
        for metric in ["correlation", "cosine"]:
            isc = nb.isc(dms, pairwise=False, metric=metric)
            assert isc.shape == (ns, nv)
            assert np.all(isc >= -1) and np.all(isc <= 1)
