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
    assert (tmp_path / "result.npy.completed").exists()
    assert not (tmp_path / "result.npy.started").exists()
    assert utils.parse_record("result.npy.completed").shape == (2,)


def test_record_waits_for_removed_running_file(tmp_path, monkeypatch):
    output = tmp_path / "result.npy"
    running = tmp_path / "result.npy.started"
    running.write_text(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
    expected = np.arange(3)

    def finish_other_job(seconds):
        assert seconds == 600
        utils.save(str(output), expected)
        running.unlink()

    def unexpected_compute():
        pytest.fail("Completed output should be loaded without recomputing")

    monkeypatch.setattr(utils.time, "sleep", finish_other_job)
    result = utils.save_results(
        str(output), unexpected_compute, return_results=True, verbose=False
    )()
    np.testing.assert_array_equal(result, expected)


def test_record_running_file_disappears_before_remove(tmp_path, monkeypatch):
    output = str(tmp_path / "result.npy")
    original_remove = utils.os.remove

    def remove_after_other_job(path):
        if path == output + ".started":
            original_remove(path)
        original_remove(path)

    monkeypatch.setattr(utils.os, "remove", remove_after_other_job)
    utils.save_results(output, lambda: np.arange(3), verbose=False)()
    assert (tmp_path / "result.npy.completed").exists()


@pytest.mark.parametrize("result_count", [1, 3])
def test_record_validates_output_count_before_saving(tmp_path, result_count):
    outputs = [str(tmp_path / f"{i}.npy") for i in range(2)]
    wrapped = utils.save_results(
        outputs, lambda: [np.arange(3)] * result_count, verbose=False
    )
    with pytest.raises(ValueError, match=f"Expected 2 results, got {result_count}"):
        wrapped()
    assert not list(tmp_path.glob("*.npy"))
    assert not list(tmp_path.glob("*.completed"))


@pytest.mark.parametrize("previous_finish", [False, True])
def test_record_save_failure_has_no_finish_marker(
    tmp_path, monkeypatch, previous_finish
):
    outputs = [str(tmp_path / f"{i}.npy") for i in range(2)]
    finish = tmp_path / "0.npy.completed"
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


def _record_contender(output, barrier, release, messages):
    def compute():
        messages.put("computing")
        if not release.wait(15):
            raise RuntimeError("Test did not release the computation")
        return np.arange(3)

    barrier.wait(timeout=15)
    utils.save_results(output, compute, verbose=False)()
    messages.put("returned")


def test_record_simultaneous_processes(tmp_path):
    import multiprocessing

    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(4)
    release = ctx.Event()
    messages = ctx.Queue()
    output = str(tmp_path / "result.npy")
    processes = [
        ctx.Process(target=_record_contender, args=(output, barrier, release, messages))
        for _ in range(4)
    ]
    for process in processes:
        process.start()
    try:
        # The winner stays inside compute until every other process skips.
        first_messages = [messages.get(timeout=15) for _ in processes]
        assert first_messages.count("computing") == 1
        assert first_messages.count("returned") == 3
    finally:
        release.set()
        for process in processes:
            process.join(timeout=15)
            if process.is_alive():
                process.terminate()
                process.join()
    assert all(process.exitcode == 0 for process in processes)
    np.testing.assert_array_equal(utils.load(output), np.arange(3))
    assert not list(tmp_path.glob("*.started*"))


def test_claim_publishes_complete_timestamp(tmp_path, monkeypatch):
    running = str(tmp_path / "result.started")
    original_link = utils.os.link
    fmt = "%Y-%m-%d %H:%M:%S.%f"

    def inspect_link(source, target):
        with open(source) as f:
            datetime.strptime(f.read(), fmt)
        original_link(source, target)

    monkeypatch.setattr(utils.os, "link", inspect_link)
    assert utils._claim_started(running, fmt) is not None
    assert utils._claim_started(running, fmt) is None
    assert len(list(tmp_path.iterdir())) == 1


def test_claim_handles_lost_nfs_reply(tmp_path, monkeypatch):
    original_link = utils.os.link

    def lost_reply(source, target):
        original_link(source, target)
        raise OSError("Simulated lost NFS reply")

    monkeypatch.setattr(utils.os, "link", lost_reply)
    running = str(tmp_path / "result.started")
    assert utils._claim_started(running, "%Y-%m-%d %H:%M:%S.%f") is not None
    assert len(list(tmp_path.iterdir())) == 1


def test_claim_propagates_filesystem_errors(tmp_path, monkeypatch):
    def denied(source, target):
        raise PermissionError("Denied")

    monkeypatch.setattr(utils.os, "link", denied)
    with pytest.raises(PermissionError):
        utils._claim_started(str(tmp_path / "result.started"), "%Y-%m-%d %H:%M:%S.%f")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("timestamp", ["", "incomplete"])
def test_record_skips_incomplete_running_file(tmp_path, timestamp):
    output = str(tmp_path / "result.npy")
    (tmp_path / "result.npy.started").write_text(timestamp)

    def unexpected_compute():
        pytest.fail("An incomplete timestamp must not permit a duplicate job")

    assert utils.save_results(output, unexpected_compute, verbose=False)() is None


def test_record_loses_claim_after_missing_file_read(tmp_path, monkeypatch):
    output = str(tmp_path / "result.npy")
    original_claim = utils._claim_started

    def another_node_claims_first(running, fmt):
        assert original_claim(running, fmt) is not None
        return original_claim(running, fmt)

    def unexpected_compute():
        pytest.fail("Losing the claim must skip computation")

    monkeypatch.setattr(utils, "_claim_started", another_node_claims_first)
    assert utils.save_results(output, unexpected_compute, verbose=False)() is None
    assert (tmp_path / "result.npy.started").exists()


def test_record_rechecks_completion_after_claim(tmp_path, monkeypatch):
    output = str(tmp_path / "result.npy")
    original_claim = utils._claim_started

    def another_node_finishes_first(running, fmt):
        utils.save(output, np.arange(3))
        return original_claim(running, fmt)

    def unexpected_compute():
        pytest.fail("Results completed before acquisition must be reused")

    monkeypatch.setattr(utils, "_claim_started", another_node_finishes_first)
    actual = utils.save_results(
        output, unexpected_compute, return_results=True, verbose=False
    )()
    np.testing.assert_array_equal(actual, np.arange(3))
    assert not (tmp_path / "result.npy.started").exists()


def test_cleanup_preserves_replacement_lock(tmp_path):
    running = str(tmp_path / "result.started")
    fmt = "%Y-%m-%d %H:%M:%S.%f"
    identity = utils._claim_started(running, fmt)
    # Keep the old inode alive so its number cannot be reused.
    utils.os.rename(running, running + ".old")
    replacement = utils._claim_started(running, fmt)
    utils._remove_started(running, identity)
    assert utils._file_identity(utils.os.stat(running)) == replacement


@pytest.mark.parametrize("rerun", [False, True])
def test_record_replaces_expired_or_overridden_lock(tmp_path, rerun):
    output = str(tmp_path / "result.npy")
    running = tmp_path / "result.npy.started"
    timestamp = datetime.now() if rerun else datetime(2000, 1, 1)
    running.write_text(timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"))
    result = utils.save_results(
        output, lambda: np.arange(3), rerun=rerun, return_results=True, verbose=False
    )()
    np.testing.assert_array_equal(result, np.arange(3))
    assert not running.exists()


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


@pytest.mark.parametrize("return_results", [False, True])
def test_finish_takes_precedence_over_running(tmp_path, monkeypatch, return_results):
    output = str(tmp_path / "result.npy")
    expected = np.arange(3)
    utils.save_results(output, lambda: expected, verbose=False)()
    running = tmp_path / "result.npy.started"
    running.write_text(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))

    def unexpected(*args):
        pytest.fail("Completed work must neither wait nor recompute")

    monkeypatch.setattr(utils.time, "sleep", unexpected)
    actual = utils.save_results(
        output, unexpected, return_results=return_results, verbose=False
    )()
    if return_results:
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual is None
    assert not running.exists()
    assert (tmp_path / "result.npy.completed").exists()


def test_waiter_uses_finish_without_running_removal(tmp_path, monkeypatch):
    output = str(tmp_path / "result.npy")
    running = tmp_path / "result.npy.started"
    running.write_text(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
    sleeps = []

    def other_job_finishes(seconds):
        sleeps.append(seconds)
        assert len(sleeps) == 1
        utils.save(output, np.arange(3))
        utils._write_completed(output + ".completed", "Completed by another job")
        # A crashed competitor's running marker remains.

    def unexpected():
        pytest.fail("The completed job must not be recomputed")

    monkeypatch.setattr(utils.time, "sleep", other_job_finishes)
    actual = utils.save_results(
        output, unexpected, return_results=True, verbose=False
    )()
    np.testing.assert_array_equal(actual, np.arange(3))
    assert not running.exists()


def test_failed_competitor_preserves_successful_output(tmp_path, monkeypatch):
    output = str(tmp_path / "result.npy")
    winner = np.arange(3)
    original_save = utils.save

    def partial_save_then_fail(path, data):
        # The other job completes while this job is saving its staged output.
        original_save(output, winner)
        utils._write_completed(output + ".completed", "Successful competitor")
        Path(path).write_bytes(b"partial failed output")
        raise OSError("Failed competitor")

    monkeypatch.setattr(utils, "save", partial_save_then_fail)
    with pytest.raises(OSError, match="Failed competitor"):
        utils.save_results(output, lambda: np.arange(5), verbose=False)()
    np.testing.assert_array_equal(utils.load(output), winner)
    assert (tmp_path / "result.npy.completed").read_text() == "Successful competitor"
    assert (tmp_path / "result.npy.started").exists()
    actual = utils.save_results(
        output,
        lambda: pytest.fail("Must reuse winner"),
        return_results=True,
        verbose=False,
    )()
    np.testing.assert_array_equal(actual, winner)
    assert not (tmp_path / "result.npy.started").exists()
    assert not list(tmp_path.glob(".nb-*"))


def test_finish_is_published_after_all_outputs(tmp_path, monkeypatch):
    outputs = [str(tmp_path / f"{i}.npy") for i in range(2)]
    finish = outputs[0] + ".completed"
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


def test_rerun_overrides_both_markers(tmp_path):
    output = str(tmp_path / "result.npy")
    utils.save_results(output, lambda: np.arange(3), verbose=False)()
    running = tmp_path / "result.npy.started"
    running.write_text(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
    actual = utils.save_results(
        output, lambda: np.arange(5), rerun=True, return_results=True, verbose=False
    )()
    np.testing.assert_array_equal(actual, np.arange(5))
    np.testing.assert_array_equal(utils.load(output), actual)
    assert not running.exists()


def test_success_cleans_crashed_competitors_running_file(tmp_path):
    output = str(tmp_path / "result.npy")
    running = tmp_path / "result.npy.started"

    def compute():
        # A timeout/override replaced this job's lock; that competitor crashed.
        running.rename(tmp_path / "old-running")
        utils._claim_started(str(running), "%Y-%m-%d %H:%M:%S.%f")
        return np.arange(3)

    utils.save_results(output, compute, verbose=False)()
    assert (tmp_path / "result.npy.completed").exists()
    assert not running.exists()
    np.testing.assert_array_equal(utils.load(output), np.arange(3))


def test_failed_finish_publication_leaves_no_marker(tmp_path, monkeypatch):
    output = str(tmp_path / "result.npy")
    original_replace = utils.os.replace

    def fail_finish(source, target):
        if target == output + ".completed":
            raise OSError("Could not publish finish")
        original_replace(source, target)

    monkeypatch.setattr(utils.os, "replace", fail_finish)
    with pytest.raises(OSError, match="Could not publish finish"):
        utils.save_results(output, lambda: np.arange(3), verbose=False)()
    assert not (tmp_path / "result.npy.completed").exists()
    np.testing.assert_array_equal(utils.load(output), np.arange(3))
    assert not list(tmp_path.glob(".nb-*"))
