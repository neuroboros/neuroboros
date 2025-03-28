import io
import lzma
import os
import tarfile
from datetime import datetime


def _prepare_file_list(inputs):
    """
    Parameters
    ----------
    inputs : str or list of str
        A file or directory or a list of files and/or directories to be added
        to the tar file.

    Returns
    -------
    todo : list of tuples
        A list of tuples, each containing the size and the file name.
        The size is the size of the file in bytes.
    """
    if isinstance(inputs, str):
        inputs = [inputs]
    todo = []
    for input_ in inputs:
        if os.path.isfile(input_):
            size = os.path.getsize(input_)
            todo.append((size, input_))
        elif os.path.isdir(input_):
            for root, dirs, files in os.walk(input_):
                for name in files:
                    size = os.path.getsize(os.path.join(root, name))
                    todo.append((size, os.path.join(root, name)))
    return todo


def archive(lzma_fn, inputs, rename_func=None, check=True, overwrite=False):
    """
    Create a lzma compressed tar file from a list of files and/or directories.

    Parameters
    ----------
    lzma_fn : str
        The name of the lzma file to be created.
    inputs : str or list of str
        A file or directory or a list of files and/or directories to be added
        to the tar file.
    rename_func : function or None, default=None
        A function to rename the files in the tar file.
    check : bool, default=True
        If True, check if the lzma file is correct. The default is True.
    overwrite : bool, default=False
        If True, overwrite the lzma file if it exists. If False, do not
        overwrite the lzma file if it exists.
    """
    if not overwrite and os.path.exists(lzma_fn):
        return
    os.makedirs(os.path.dirname(os.path.realpath(lzma_fn)), exist_ok=True)
    t0 = datetime.now()
    tar_io = io.BytesIO()
    todo = _prepare_file_list(inputs)
    s1 = sum([_[0] for _ in todo])

    with tarfile.open(fileobj=tar_io, mode="w") as tf:
        for size, fn in todo:
            new_fn = fn if rename_func is None else rename_func(fn)
            tf.add(fn, new_fn)
    tar_bytes = tar_io.getvalue()
    s2 = len(tar_bytes)

    lzma_bytes = lzma.compress(tar_bytes, preset=9 | lzma.PRESET_EXTREME)
    s3 = len(lzma_bytes)
    with open(lzma_fn, "wb") as f:
        f.write(lzma_bytes)

    t1 = datetime.now()
    print(t1, lzma_fn, t1 - t0, s1, s2, s3, s3 / s2, s2 / s1, s3 / s1)

    if check:
        assert archive_check(lzma_fn, inputs, rename_func=rename_func)


def archive_check(lzma_fn, inputs, rename_func=None):
    """
    Check if the lzma file is correct by comparing it with the original files.

    Parameters
    ----------
    lzma_fn : str
        The name of the lzma file to be compared.
    inputs : str or list of str
        A file or directory or a list of files and/or directories to be
        compared with the tar file.
    rename_func : function or None, default=None
        A function to rename the files in the tar file.

    Returns
    -------
    success : bool
        True if the lzma file is correct, False otherwise.
    """
    todo = _prepare_file_list(inputs)
    with tarfile.open(lzma_fn, "r:xz") as tf:
        tf_fns = [_.name for _ in tf]
        if len(todo) != len(tf_fns):
            print(f"WARNING: Number of files mismatch.")
        for size, fn in todo:
            new_fn = fn if rename_func is None else rename_func(fn)
            if new_fn not in tf_fns:
                print(f"ERROR: {new_fn} not in tar file.")
                return False
            out_file = tf.extractfile(new_fn).read()
            with open(fn, "rb") as f:
                in_file = f.read()
            if (len(in_file) != len(out_file)) or (in_file != out_file):
                print(f"ERROR: {new_fn} file content mismatch.")
                return False
    return True
