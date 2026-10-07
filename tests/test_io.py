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
    assert not list(tmp_path.glob("*.finish"))


@pytest.mark.parametrize("previous_finish", [False, True])
def test_record_save_failure_has_no_finish_marker(
    tmp_path, monkeypatch, previous_finish
):
    outputs = [str(tmp_path / f"{i}.npy") for i in range(2)]
    finish = tmp_path / "0.npy.finish"
    if previous_finish:
        finish.write_text("Previous successful run")
    original_save = utils.save

    def failing_save(path, data):
        assert not finish.exists()
        if Path(path).name == Path(outputs[1]).name:
            raise OSError("Simulated save failure")
        original_save(path, data)

    monkeypatch.setattr(utils, "save", failing_save)
    with pytest.raises(OSError, match="Simulated save failure"):
        utils.save_results(
            outputs,
            lambda: [np.arange(3), np.arange(4)],
            rerun=previous_finish,
            verbose=False,
        )()
    assert not finish.exists()


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


def test_finish_is_published_after_all_outputs(tmp_path, monkeypatch):
    outputs = [str(tmp_path / f"{i}.npy") for i in range(2)]
    finish = outputs[0] + ".finish"
    original_replace = utils.os.replace
    replaced = []

    def inspect_replace(source, target):
        assert not Path(finish).exists()
        if target == finish:
            assert replaced == outputs
            for index, output in enumerate(outputs):
                np.testing.assert_array_equal(utils.load(output), np.arange(index + 3))
            assert Path(source).read_text().startswith("Computation finished at")
        original_replace(source, target)
        replaced.append(target)

    monkeypatch.setattr(utils.os, "replace", inspect_replace)
    utils.save_results(outputs, lambda: [np.arange(3), np.arange(4)], verbose=False)()
    assert replaced == outputs + [finish]
    assert utils.parse_record(finish).shape == (2,)


def test_failed_finish_publication_leaves_no_marker(tmp_path, monkeypatch):
    output = str(tmp_path / "result.npy")
    original_replace = utils.os.replace

    def fail_finish(source, target):
        if target == output + ".finish":
            raise OSError("Could not publish finish")
        original_replace(source, target)

    monkeypatch.setattr(utils.os, "replace", fail_finish)
    with pytest.raises(OSError, match="Could not publish finish"):
        utils.save_results(output, lambda: np.arange(3), verbose=False)()
    assert not (tmp_path / "result.npy.finish").exists()
    np.testing.assert_array_equal(utils.load(output), np.arange(3))
    assert not list(tmp_path.glob(".nb-*"))
