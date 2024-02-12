import numpy as np
from scipy.stats import ttest_ind

from neuroboros import glm


def test_glm():
    rng = np.random.default_rng()
    dm = rng.standard_normal((1000, 500))
    nuisance = rng.standard_normal((1000, 10))
    design = rng.standard_normal((1000, 10))
    betas, ts, R2s = glm(dm, design, nuisance, return_r2=True)

    assert betas.shape == (10, 500)
    assert ts.shape == (10, 500)
    assert R2s.shape == (500,)

    assert np.allclose(np.mean(R2s), 0, atol=0.001)
    assert np.allclose(np.mean(betas), 0, atol=0.001)


def test_t_value():
    rng = np.random.default_rng()
    dm = rng.standard_normal((100, 1))
    nuisance = None
    design = rng.choice([0, 1], (100, 1))
    design = np.concatenate([design, np.ones_like(design)], axis=1)
    ts = glm(dm, design, nuisance)[1]
    t = ttest_ind(dm[design[:, 0] == 1], dm[design[:, 0] == 0], equal_var=True)
    print(ts[0], t.statistic)
    assert np.allclose(ts[0], t.statistic, atol=0.001)
