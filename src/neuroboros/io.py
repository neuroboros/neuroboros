import os
import warnings

import requests

try:
    import datalad.api as dl
except ImportError:
    dl = None

from .utils import load

DATA_ROOT = os.environ.get(
    'NEUROBOROS_DATA_DIR', os.path.join(os.path.expanduser('~'), '.neuroboros-data')
)


class AlternativeDataset:
    def __init__(self, name, source, root=None):
        self.name = name
        self.root = os.path.join(DATA_ROOT, name) if root is None else root
        os.makedirs(self.root, exist_ok=True)
        self.source = source
        if self.source.startswith('https://gin.g-node.org/'):
            self.url_base = self.source + '/raw/master/'
        else:
            raise NotImplementedError(
                "Only supports downloading data from GIN without DataLad."
            )

    def get(self, fn, load_func=None, on_missing='warn'):
        local_fn = os.path.join(self.root, fn)
        if not os.path.exists(local_fn):
            url = self.url_base + fn
            try:
                r = requests.get(url)
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Error downloading {url}: {e}")
            else:
                if r.status_code == 404:
                    if on_missing == 'raise':
                        raise RuntimeError(f"File {url} not found.")
                    elif on_missing == 'warn':
                        warnings.warn(f"File {url} not found.")
                        return None
                    elif on_missing == 'ignore':
                        return None
                    else:
                        raise ValueError(
                            f"Invalid value for `on_missing`: {on_missing}"
                        )
                if r.status_code != 200:
                    raise RuntimeError(f"Error downloading {url}: {r.status_code}")
            os.makedirs(os.path.dirname(local_fn), exist_ok=True)
            with open(local_fn, 'wb') as f:
                f.write(r.content)

        if load_func is None:
            return load(local_fn)
        return load_func(local_fn)


class DataLadDataset:
    def __init__(self, name, source, root=None):
        self.name = name
        self.root = os.path.join(DATA_ROOT, name) if root is None else root
        if os.path.exists(self.root):
            self.dset = dl.Dataset(self.root)
            assert self.dset.repo is not None
            urls = [
                self.dset.repo.get_remote_url(remote)
                for remote in self.dset.repo.get_remotes()
            ]
            if source not in urls:
                warnings.warn(
                    f"DataLad dataset {self.name} exists at {self.root}, but "
                    f"does not have source {source} as a sibling."
                )
        else:
            self.dset = dl.clone(source, self.root)
        self.source = source

    def get(self, fn, load_func=None, on_missing='warn'):
        local_fn = os.path.join(self.root, fn)
        if not os.path.lexists(local_fn):
            if on_missing == 'raise':
                raise RuntimeError(f"File {local_fn} not found in DataLad repo.")
            elif on_missing == 'warn':
                warnings.warn(f"File {local_fn} not found in DataLad repo.")
                return None
            elif on_missing == 'ignore':
                return None
            else:
                raise ValueError(f"Invalid value for `on_missing`: {on_missing}")

        if not os.path.exists(local_fn):
            result = self.dset.get(fn)[0]
            if result['status'] not in ['ok', 'notneeded']:
                raise RuntimeError(
                    f"datalad `get` status is {result['status']}, likely due "
                    "to problems downloading the file."
                )
            assert os.path.normpath(result['path']) == os.path.normpath(local_fn)

        if load_func is None:
            return load(local_fn)
        return load_func(local_fn)


class LocalDataset:
    def __init__(self, name, root=None):
        self.name = name
        self.root = os.path.join(DATA_ROOT, name) if root is None else root
        assert os.path.exists(self.root)

    def get(self, fn, load_func=None, on_missing='warn'):
        local_fn = os.path.join(self.root, fn)
        if not os.path.exists(local_fn):
            if on_missing == 'raise':
                raise RuntimeError(f"File {local_fn} not found.")
            elif on_missing == 'warn':
                warnings.warn(f"File {local_fn} not found.")
                return None
            elif on_missing == 'ignore':
                return None
            else:
                raise ValueError(f"Invalid value for `on_missing`: {on_missing}")

        if load_func is None:
            return load(local_fn)
        return load_func(local_fn)


DefaultDataset = AlternativeDataset if dl is None else DataLadDataset

if os.path.exists(os.path.join(DATA_ROOT, 'core')) and not os.path.exists(
    os.path.join(DATA_ROOT, 'core', '.git')
):
    core_dataset = LocalDataset('core')
else:
    core_dataset = DefaultDataset('core', 'https://gin.g-node.org/neuroboros/core')
