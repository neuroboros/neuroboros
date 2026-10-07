from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from neuroboros import utils


def test_record_local_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    expected = np.arange(3)
    wrapped = utils.save_results(
        "result.npy", lambda: expected, return_results=True, verbose=False
    )
    np.testing.assert_array_equal(wrapped(), expected)
    np.testing.assert_array_equal(wrapped(), expected)
    assert (tmp_path / "result.npy.finish").exists()
    assert not (tmp_path / "result.npy.running").exists()
    assert utils.parse_record("result.npy.finish").shape == (2,)


@pytest.mark.parametrize("result_count", [1, 3])
def test_record_validates_output_count_before_saving(tmp_path, result_count):
    outputs = [str(tmp_path / f"{i}.npy") for i in range(2)]
    wrapped = utils.save_results(
        outputs, lambda: [np.arange(3)] * result_count, verbose=False
    )
    with pytest.raises(ValueError, match=f"Expected 2 results, got {result_count}"):
        wrapped()
    assert not list(tmp_path.glob("*.npy"))


def test_record_multiple_outputs_round_trip(tmp_path):
    outputs = [str(tmp_path / f"{i}.npy") for i in range(2)]
    expected = [np.arange(3), np.arange(4)]
    calls = []

    def compute():
        calls.append(True)
        return expected

    wrapped = utils.save_results(outputs, compute, return_results=True, verbose=False)
    for result in (wrapped(), wrapped()):
        for actual, wanted in zip(result, expected):
            np.testing.assert_array_equal(actual, wanted)
    assert len(calls) == 1


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
