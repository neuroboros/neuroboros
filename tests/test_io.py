from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from neuroboros import utils


@pytest.mark.parametrize("extension", [".npz", ".pkl"])
def test_save_sparse_round_trip(tmp_path, extension):
    data = sparse.csr_matrix([[0, 1], [2, 0]])
    output = tmp_path / ("sparse" + extension)
    utils.save(str(output), data)
    assert output.exists()
    assert list(tmp_path.iterdir()) == [output]
    loaded = utils.load(str(output))
    assert sparse.issparse(loaded)
    np.testing.assert_array_equal(loaded.toarray(), data.toarray())



def test_save_sparse_unsupported_extension(tmp_path):
    with pytest.raises(TypeError):
        utils.save(str(tmp_path / "sparse.unknown"), sparse.eye(2, format="csr"))
    assert not list(tmp_path.iterdir())
