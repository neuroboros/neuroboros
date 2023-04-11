import os
import datalad.api as dl

from .utils import load

DATA_ROOT = os.environ.get(
    'NEUROBOROS_DATA_DIR',
    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nb_data'))


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


def load_file(fn, root=None):
    if root is None:
        root = DATA_ROOT
    dset = dl.Dataset(root)
    path = os.path.join(root, fn)
    if not (dset.repo is None):
        res = download_datalad_file(fn, dset)
        assert os.path.normpath(path) == os.path.normpath(res['path'])
    if not os.path.exists(path):
        return None
    return load(path)
