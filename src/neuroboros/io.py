import os
import datalad.api as dl

from .utils import load

DATA_ROOT = os.environ.get(
    'NEUROBOROS_DATA_DIR',
    os.path.join(os.path.expanduser('~'), '.neuroboros-data'))


def get_core_dataset():
    core_dir = os.path.join(DATA_ROOT, 'core')
    if os.path.exists(core_dir):
        dset = dl.Dataset(core_dir)
    else:
        dset = dl.clone(
            source='https://gin.g-node.org/neuroboros/core',
            path=core_dir)
    return dset, core_dir


def _follow_symlink(fn, root):
    # os.path.islink(path)
    fn_ = os.path.join(root, fn)
    fn = os.path.join(os.path.dirname(fn_), os.readlink(fn_))
    fn = os.path.normpath(fn)
    fn = os.path.relpath(fn, root)
    return fn


def download_datalad_file(fn, dl_dset):
    result = dl_dset.get(fn)[0]
    if result['status'] not in ['ok', 'notneeded']:
        raise RuntimeError(
            f"datalad `get` status is {result['status']}, likely due to "
            "problems downloading the file.")
    return result['path']


def load_file(fn, dset=None, root=None, load_func=None):
    if root is None:
        if dset is None:
            dset, root = get_core_dataset()
        else:
            root = dset.path
    path = os.path.join(root, fn)

    if not os.path.lexists(path):
        return None
    if not os.path.exists(path):
        if dset is None:
            dset = dl.Dataset(root)
        if dset.repo is None:
            return None

        res = download_datalad_file(fn, dset)
        assert os.path.normpath(path) == os.path.normpath(res)

    if load_func is None:
        return load(path)
    return load_func(path)
