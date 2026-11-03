"""
===================================
Utilities (:mod:`neuroboros.utils`)
===================================

.. currentmodule:: neuroboros.utils

Function execution
==================

.. autosummary::
    :toctree:

    monitor - Monitors function execution and records running time and system information.
    save_results - Run the function and save the returned output to file.
    parse_record - Parse the timing information based on record.

Input / Output
==============

.. autosummary::
    :toctree:

    save - Save the data using the automatically determined format.
    load - Load the file using the automatically determined function.

High-performance computing
==========================

.. autosummary::
    :toctree:

    assert_sufficient_time - Check if remaining SLURM walltime is sufficient.

"""

import functools
import gzip
import json
import os
import pickle
import subprocess
import tempfile
import time
import warnings
from datetime import datetime, timedelta

import joblib
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.sparse as sparse

try:
    from PIL import Image

    PIL_ok = True
except ImportError as e:
    PIL_ok = False


def percentile(data, ignore_nan=False, **kwargs):
    count = np.isnan(data).sum()
    if count:
        if not ignore_nan:
            if not isinstance(data, np.ndarray):
                data = np.asarray(data)
            warnings.warn(f"{count} values out of {data.size} are NaNs.")
        print(np.nanpercentile(data, np.linspace(0, 100, 11), **kwargs))
    else:
        print(np.percentile(data, np.linspace(0, 100, 11), **kwargs))


def save(fn, data):
    """Save the data using the automatically determined format.

    If ``fn`` ends with ".npy", save as npy file.
    If ``fn`` ends with ".npz" and ``data`` is sparse, save as npz file.
    If dict and ``fn`` ends with ".npz", save as npz file.
    If ``fn`` ends with ".pkl", save using ``pickle.dump``.
    If ``fn`` ends with ".png", save as PNG file (requires the Pillow
    package).
    If ``fn`` ends with ".json", save as JSON file.

    The function also automatically creates the directory ``fn`` is in if it
    does not exist.

    Parameters
    ----------
    fn : str
        The filename (including path) to save the data to.
    data : ndarray or spmatrix or dict or object
        The data to be saved.

    Returns
    -------
    ret
        Returned value of ``numpy.save``, ``numpy.savez``,
        ``scipy.sparse.save_npz``, or ``pickle.dump``.

    Raises
    ------
    TypeError
        Raised when ``type(data)`` is not supported.


    """
    dirname = os.path.dirname(fn)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    if fn.endswith(".npy"):
        if not isinstance(data, np.ndarray):
            warnings.warn("`data` is not an ndarray, trying to convert.")
        return np.save(fn, data)

    if fn.endswith(".npz") and sparse.issparse(data):
        return sparse.save_npz(fn, data)
    if fn.endswith(".npz") and isinstance(data, dict):
        return np.savez(fn, **data)

    if fn.endswith(".pkl"):
        with open(fn, "wb") as f:
            return pickle.dump(data, f)

    if fn.endswith(".png"):
        assert PIL_ok, "Needs the Pillow package to save images."
        if isinstance(data, Image.Image):
            im = data
        elif isinstance(data, np.ndarray):
            im = Image.fromarray(data)
        else:
            raise TypeError(f"Cannot save '{type(data)}' to image.")
        return im.save(fn)

    if fn.endswith(".shape.gii"):
        darray = nib.gifti.GiftiDataArray(
            data.astype(np.float32),
            intent=nib.nifti1.intent_codes["NIFTI_INTENT_SHAPE"],
            datatype=nib.nifti1.data_type_codes["NIFTI_TYPE_FLOAT32"],
        )
        gii = nib.gifti.GiftiImage(darrays=[darray])
        return nib.save(gii, fn)

    if fn.endswith(".func.gii"):
        if data.ndim == 1:
            data = data[:, None]
        darrays = [
            nib.gifti.GiftiDataArray(
                d.astype(np.float32),
                intent=nib.nifti1.intent_codes["NIFTI_INTENT_TIME_SERIES"],
                datatype=nib.nifti1.data_type_codes["NIFTI_TYPE_FLOAT32"],
            )
            for d in data.T
        ]
        gii = nib.gifti.GiftiImage(darrays=darrays)
        return nib.save(gii, fn)

    if fn.endswith(".json"):
        with open(fn, "w") as f:
            return json.dump(data, f)

    raise TypeError(f"`data` type {type(data)} not supported.")


def load(fn, **kwargs):
    """Load the file using the automatically determined function.

    Parameters
    ----------
    fn : str
        File name of the file to be loaded.

    Returns
    -------
    ret
        Returned value of ``numpy.load``, ``scipy.sparse.load_npz``,
        ``pickle.load``, ``pandas.read_csv``, or ``json.load``.

    Raises
    ------
    TypeError
        Raised when the type of ``fn`` is not supported.
    """
    if fn.endswith(".npy"):
        return np.load(fn, **kwargs)
    if fn.endswith(".npz"):
        try:
            return sparse.load_npz(fn, **kwargs)
        except (OSError, ValueError):
            return np.load(fn, **kwargs)
    if fn.endswith(".tsv"):
        d = dict(delimiter="\t", na_values="n/a")
        d.update(kwargs)
        return pd.read_csv(fn, **d)
    if fn.endswith(".csv"):
        d = dict(na_values="n/a")
        d.update(kwargs)
        return pd.read_csv(fn, **d)
    if fn.endswith(".pkl"):
        with open(fn, "rb") as f:
            return pickle.load(f, **kwargs)
    if fn.endswith(".json.gz"):
        with gzip.open(fn, "rb") as f:
            return json.load(f, **kwargs)
    if fn.endswith(".json"):
        with open(fn, "rb") as f:
            return json.load(f, **kwargs)
    raise TypeError(f"file type of `fn` is not supported.")


def parse_record(record_fn, assert_node=None):
    """Parse the timing information based on record.

    This function parses the record file generated by ``monitor`` or
    ``save_results`` and returns the elapsed CPU time and wall time as a NumPy
    array.

    Parameters
    ----------
    record_fn : str
        File name of the record file.

    Returns
    -------
    t : ndarray
        A numpy array with 2 elements, where the 1st is the CPU time, and the
        2nd is the wall time. Both are in nanoseconds.
    """
    with open(record_fn) as f:
        lines = f.read().splitlines()
    cpu_time = int(lines[2]) - int(lines[1])
    wall_time = int(lines[4]) - int(lines[3])
    if assert_node is not None:
        if assert_node not in [lines[5], lines[5].split(".")[0]]:
            raise ValueError(
                f"Expecting Node `{assert_node}`, got {lines[5]} in record."
            )
    t = np.array([cpu_time, wall_time])
    return t


def monitor(func, record_fn=None):
    """
    Monitors function execution and records running time and system information.

    This function takes a function ``func`` as input and outputs a wrapped
    function ``wrapped_func``. ``wrapped_func`` is similar to ``func``, except
    that it also records the running time of function execution and system
    information. When ``record_fn`` is None, ``wrapped_func`` will return a
    tuple, where the 1st element is the record and the 2nd element is the
    original returned value of ``func``. When it's not None, the record will
    be saved into the file, and only the original returned value of ``func``
    will be returned.

    Parameters
    ----------
    func : function
        The original function to be wrapped.
    record_fn : {str, None}, default=None
        When it is None, the monitoring record will be the first returned
        value of ``wrapped_func``. Otherwise, the information will be saved
        into a text file with the file name ``record_fn`` and only the
        original returned value of ``func`` will be returned.

    Returns
    -------
    wrapped_func : function
        The wrapped function that will record running information.
    """
    fmt = "%Y-%m-%d %H:%M:%S.%f"

    def monitored_func(*args, **kwargs):
        cpu_time_start = time.process_time_ns()
        wall_time_start = time.perf_counter_ns()
        results = func(*args, **kwargs)
        cpu_time_end = time.process_time_ns()
        wall_time_end = time.perf_counter_ns()

        hostname = os.uname()[1]
        total_cpus = os.cpu_count()
        avail_cpus = joblib.cpu_count()

        info = (
            f"Computation finished at {datetime.now().strftime(fmt)}\n"
            f"{cpu_time_start}\n{cpu_time_end}\n"
            f"{wall_time_start}\n{wall_time_end}\n"
            f"{hostname}\n{avail_cpus}\n{total_cpus}\n"
        )
        if record_fn is None:
            return info, results
        else:
            with open(record_fn, "w") as f:
                f.write(info)
            return results

    wrapped_func = functools.wraps(func)(monitored_func)
    return wrapped_func


def _file_identity(stat):
    return stat.st_dev, stat.st_ino


def _claim_started(started_fn, fmt):
    """Publish a complete timestamp using the NFS hard-link lock protocol."""
    dirname = os.path.dirname(started_fn) or "."
    with tempfile.NamedTemporaryFile(
        mode="w", dir=dirname, prefix=os.path.basename(started_fn) + "."
    ) as f:
        f.write(datetime.now().strftime(fmt))
        f.flush()
        os.fsync(f.fileno())
        try:
            os.link(f.name, started_fn)
        except OSError as exc:
            # An NFS server can complete LINK even if its reply is lost.
            # The second link then establishes that this client owns the lock.
            if os.stat(f.name).st_nlink != 2:
                if isinstance(exc, FileExistsError):
                    return None
                raise
        return _file_identity(os.fstat(f.fileno()))


def _remove_started(started_fn, identity):
    """Best-effort cleanup that checks for a replacement lock first."""
    try:
        if _file_identity(os.stat(started_fn)) == identity:
            os.remove(started_fn)
    except FileNotFoundError:
        pass


def _save_record_output(fn, data):
    """Keep partial writes away from the final output path."""
    dirname = os.path.dirname(fn) or "."
    os.makedirs(dirname, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dirname, prefix=".nb-save-") as tmp:
        staged = os.path.join(tmp, os.path.basename(fn))
        save(staged, data)
        with open(staged, "rb") as f:
            os.fsync(f.fileno())
        os.replace(staged, fn)


def _write_completed(completed_fn, info):
    """Publish the complete timing record after every output has been saved."""
    dirname = os.path.dirname(completed_fn) or "."
    with tempfile.TemporaryDirectory(dir=dirname, prefix=".nb-completed-") as tmp:
        staged = os.path.join(tmp, "completed")
        with open(staged, "w") as f:
            f.write(info)
            f.flush()
            os.fsync(f.fileno())
        os.replace(staged, completed_fn)


def save_results(
    out_fn,
    func,
    return_results=False,
    log_fn=None,
    rerun_hours=48,
    verbose=True,
    rerun=False,
):
    """Run the function and save the returned output to file.

    This function takes a function ``func`` as input and outputs a wrapped
    function ``wrapped_func``. ``wrapped_func`` is similar to ``func``, except
    that it will also save the returned output of ``func`` to files (file
    names indicated by ``out_fn``), so that it can be accessed later.

    Runs use ``.started`` and ``.completed`` markers.

    Parameters
    ----------
    out_fn : str or list/tuple of str
        The names of the output file(s).
    func : function
        The function to be wrapped.
    return_results : bool, default=False
        Whether ``wrapped_func`` will return the output of the original
        function ``func``. If it's False, None will be returned.
    log_fn : {str, None}, default=None
        The root file name of the text files for logging.
    rerun_hours : int, default=48
        Rerun the function if the start of the last run was more than
        ``rerun_hours`` ago.
    verbose : bool, default=True
        Whether to output additional logging information.
    rerun : bool, default=False
        Whether to rerun the function even if the output files exist,
        overriding existing start markers as well.

    Returns
    -------
    wrapped_func : function
        The wrapped function that will run ``func`` and save the results to
        ``out_fn``.

    Raises
    ------
    ValueError
        Raised when ``out_fn`` is neither a string or a tuple/list of strings.
    """
    if isinstance(out_fn, (list, tuple)):
        out_fns = out_fn
    elif isinstance(out_fn, str):
        out_fns = [out_fn]
    else:
        raise ValueError("`out_fn` must be a string or a list/tuple of strings.")

    if log_fn is None:
        log_fn = out_fns[0]

    started_fn = log_fn + ".started"
    completed_fn = log_fn + ".completed"
    fmt = "%Y-%m-%d %H:%M:%S.%f"

    monitored_func = monitor(func)

    def cached_results():
        if verbose:
            print(datetime.now(), f"Using completed results: {out_fns}")
        if return_results:
            results = [load(_) for _ in out_fns]
            return results[0] if len(results) == 1 else results
        return None

    def finished():
        if not os.path.exists(completed_fn):
            return False
        # A successful job takes precedence over another job's leftover lock.
        try:
            identity = _file_identity(os.stat(started_fn))
            _remove_started(started_fn, identity)
        except FileNotFoundError:
            pass
        return True

    def complete():
        return finished() or all(os.path.exists(fn) for fn in out_fns)

    def func_w_cache(*args, **kwargs):
        dirname = os.path.dirname(log_fn)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        while True:
            if not rerun and finished():
                return cached_results()
            try:
                with open(started_fn) as f:
                    stat = os.fstat(f.fileno())
                    try:
                        started = datetime.strptime(f.read(), fmt)
                    except ValueError:
                        # Incomplete markers still count as started.
                        started = datetime.fromtimestamp(stat.st_mtime)
                expired = datetime.now() - started >= timedelta(hours=rerun_hours)
                if rerun or expired:
                    _remove_started(started_fn, _file_identity(stat))
                elif not return_results:
                    return None
                else:
                    time.sleep(600)
                    continue
            except FileNotFoundError:
                pass

            if not rerun and complete():
                return cached_results()

            # Always let the server arbitrate acquisition, even if this
            # client's directory cache says the running file does not exist.
            identity = _claim_started(started_fn, fmt)
            if identity is not None:
                break
            if not rerun and finished():
                return cached_results()
            if not return_results:
                return None
            time.sleep(600)

        release = False
        try:
            # A competing job may have finished between our initial check
            # and acquisition. Check its outputs while holding the claim.
            if not rerun and complete():
                release = True
                return cached_results()

            if verbose:
                print(datetime.now(), f"Starting to compute for: {out_fns}")

            # Only an explicit rerun invalidates a completed result.
            # A normal competitor must not delete a newly published marker.
            if rerun:
                try:
                    os.remove(completed_fn)
                except FileNotFoundError:
                    pass

            info, results = monitored_func(*args, **kwargs)

            if len(out_fns) > 1:
                output_results = list(results)
                if len(output_results) != len(out_fns):
                    raise ValueError(
                        f"Expected {len(out_fns)} results, got {len(output_results)}."
                    )
                for res, fn in zip(output_results, out_fns):
                    _save_record_output(fn, res)
            else:
                _save_record_output(out_fns[0], results)

            _write_completed(completed_fn, info)
            release = True

            if return_results:
                return results
            return None
        finally:
            if release and not finished():
                _remove_started(started_fn, identity)

    wrapped_func = functools.wraps(func)(func_w_cache)
    return wrapped_func


def assert_sufficient_time(minimum="1:00:00"):
    """Check if remaining SLURM walltime is sufficient.

    This function checks if the remaining walltime of SLURM is larger than the
    ``minimum`` when the script is run through the SLURM job manager. If the
    remaining walltime is insufficient, it will stop the remaining parts of
    the script from running by executing ``exit(0)``.

    Parameters
    ----------
    minimum : str
        The minimal walltime in SLURM format. E.g., '1-02:03:04' means 1 day,
        2 hours, 3 minutes, and 4 seconds.
    """
    if "SLURM_JOBID" not in os.environ:
        return

    cmd = ["squeue", "-h", "-j", os.environ["SLURM_JOBID"], "-o", '"%L"']
    sp = subprocess.run(cmd, capture_output=True)
    remaining = sp.stdout.decode("ascii").split('"')[1]

    def parse_time(s):
        days = int(s.split("-")[0]) if "-" in s else 0
        parts = s.split("-")[-1].split(":")
        hours = int(parts[-3]) if len(parts) >= 3 else 0
        minutes = int(parts[-2]) if len(parts) >= 2 else 0
        seconds = int(parts[-1]) if len(parts) >= 1 else 0
        return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

    r = parse_time(remaining)
    m = parse_time(minimum)
    print(datetime.now(), "Remaining walltime:", r)
    print(datetime.now(), "Minimal walltime:  ", m)
    if r < m:
        print(datetime.now(), "Insufficient remaining walltime time. Exiting.")
        exit(0)


def optimize_dtype(data):
    """Optimize the data type of a NumPy array for memory efficiency.

    Parameters
    ----------
    data : ndarray
        The input NumPy array to be optimized.

    Returns
    -------
    optimized_data : ndarray
        The NumPy array with the optimized data type.

    Raises
    ------
    ValueError
        If the input data is not a NumPy array.
    TypeError
        If the input data is a scalar value.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array.")
    if data.ndim == 0:
        raise TypeError("Input data must be a non-scalar numpy array.")
    optimized_data = data
    for dtype in [np.int32, np.uint32, np.int16, np.uint16, np.int8, np.uint8]:
        if np.all(data.astype(dtype) == data):
            optimized_data = data.astype(dtype)
    return optimized_data
