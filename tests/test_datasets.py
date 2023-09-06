import numpy as np

import neuroboros as nb


class TestDatasets:
    def test_concat_lr(self):
        dset = nb.Forrest()
        sid = dset.subjects[0]
        for runs in [1, [1, 2, 3]]:
            lh = dset.get_data(sid, 'forrest', runs, 'l')
            rh = dset.get_data(sid, 'forrest', runs, 'r')
            lr = dset.get_data(sid, 'forrest', runs, 'lr')
            cat = np.concatenate([lh, rh], axis=1)
            np.testing.assert_allclose(lr, cat, atol=1e-10)

    def test_concat_lr_scrub(self):
        dset = nb.Forrest(prep='scrub')
        sid = dset.subjects[0]
        for runs in [1, [1, 2, 3]]:
            lh = dset.get_data(sid, 'forrest', 1, 'l')[0]
            rh = dset.get_data(sid, 'forrest', 1, 'r')[0]
            lr = dset.get_data(sid, 'forrest', 1, 'lr')[0]
            cat = np.concatenate([lh, rh], axis=1)
            np.testing.assert_allclose(lr, cat, atol=1e-10)

    def test_concat_runs(self):
        dset = nb.Forrest()
        sid = dset.subjects[0]
        for lr in ['l', 'r', 'lr']:
            cat = np.concatenate(
                [dset.get_data(sid, 'forrest', run_, lr) for run_ in [1, 2, 3]], axis=0
            )
            dm = dset.get_data(sid, 'forrest', [1, 2, 3], lr)
            np.testing.assert_array_equal(dm, cat)

    def test_concat_runs_scrub(self):
        dset = nb.Forrest(prep='scrub')
        sid = dset.subjects[0]
        for lr in ['l', 'r', 'lr']:
            cat = np.concatenate(
                [dset.get_data(sid, 'forrest', run_, lr)[0] for run_ in [1, 2, 3]],
                axis=0,
            )
            mask = np.concatenate(
                [dset.get_data(sid, 'forrest', run_, lr)[1] for run_ in [1, 2, 3]],
                axis=0,
            )
            dm, mask2 = dset.get_data(sid, 'forrest', [1, 2, 3], lr)
            np.testing.assert_array_equal(dm, cat)
            np.testing.assert_array_equal(mask, mask2)
