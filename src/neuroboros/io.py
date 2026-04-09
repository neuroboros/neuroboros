import os
import warnings
from glob import glob

import requests

try:
    import datalad.api as dl
except ImportError:
    dl = None

from .utils import load

DATA_ROOT = os.environ.get(
    "NEUROBOROS_DATA_DIR", os.path.join(os.path.expanduser("~"), ".neuroboros-data")
)


class DatasetManager:
    def __init__(self, name, root=None, source=None, kind=None):
        self.name = name
        self.root = os.path.join(DATA_ROOT, name) if root is None else root
        self.source = source
        self.kind = kind

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, kind):
        if kind is None:
            if self.source is None:
                kind = "local"
            elif os.path.exists(os.path.join(self.root, ".git")) and dl is not None:
                kind = "datalad"
            else:
                kind = "alternative"
        else:
            assert kind in ["local", "datalad", "alternative"]
        self._kind = kind

        if self._kind == "local":
            assert os.path.exists(
                self.root
            ), f"Local dataset {self.name} does not exist at {self.root}."
        elif self._kind == "datalad":
            if os.path.exists(self.root):
                self.dset = dl.Dataset(self.root)
                assert self.dset.repo is not None
                urls = [
                    self.dset.repo.get_remote_url(remote)
                    for remote in self.dset.repo.get_remotes()
                ]
                if self.source not in urls:
                    warnings.warn(
                        f"DataLad dataset {self.name} exists at {self.root}, but "
                        f"does not have source {self.source} as a sibling."
                    )
            else:
                self.dset = dl.clone(self.source, self.root)
        elif self._kind == "alternative":
            if self.source.startswith("https://gin.g-node.org/"):
                self.url_base = self.source + "/raw/master/"
            else:
                raise NotImplementedError(
                    "Only supports downloading data from GIN without DataLad."
                )

        self.download = getattr(self, f"_download_{kind}")

    def get(self, fn, load_func=None, on_missing="warn"):
        if isinstance(fn, (tuple, list)):
            local_fn = os.path.join(self.root, *fn)
        else:
            local_fn = os.path.join(self.root, fn)

        if "*" in local_fn:
            fns = glob(local_fn)
            assert (
                len(fns) == 1
            ), f"Expecting exactly 1 file, found {len(fns)} files matching: {local_fn}."
            local_fn = fns[0]

        if not os.path.exists(local_fn):
            self.download(fn, local_fn, on_missing=on_missing)
        if not os.path.exists(local_fn):
            return None

        if load_func is None:
            return load(local_fn)
        return load_func(local_fn)

    def _download_local(self, fn, local_fn, on_missing="warn"):
        if on_missing == "raise":
            raise RuntimeError(f"File {local_fn} not found.")
        elif on_missing == "warn":
            warnings.warn(f"File {local_fn} not found.")
            return None
        elif on_missing == "ignore":
            return None
        else:
            raise ValueError(f"Invalid value for `on_missing`: {on_missing}")

    def _download_datalad(self, fn, local_fn, on_missing="warn"):
        if not os.path.lexists(local_fn):
            if on_missing == "raise":
                raise RuntimeError(f"File {local_fn} not found in DataLad repo.")
            elif on_missing == "warn":
                warnings.warn(f"File {local_fn} not found in DataLad repo.")
                return None
            elif on_missing == "ignore":
                return None
            else:
                raise ValueError(f"Invalid value for `on_missing`: {on_missing}")

        result = self.dset.get("/".join(fn))[0]
        if result["status"] not in ["ok", "notneeded"]:
            raise RuntimeError(
                f"datalad `get` status is {result['status']}, likely due "
                "to problems downloading the file."
            )
        assert os.path.normpath(result["path"]) == os.path.normpath(local_fn)

    def _download_alternative(self, fn, local_fn, on_missing="warn"):
        if isinstance(fn, (tuple, list)):
            url = self.url_base + "/".join(fn)
        else:
            url = self.url_base + fn.replace("\\", "/")

        try:
            r = requests.get(url)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error downloading {url}: {e}")
        else:
            if r.status_code == 404:
                if on_missing == "raise":
                    raise RuntimeError(f"File {url} not found.")
                elif on_missing == "warn":
                    warnings.warn(f"File {url} not found.")
                    return None
                elif on_missing == "ignore":
                    return None
                else:
                    raise ValueError(f"Invalid value for `on_missing`: {on_missing}")
            if r.status_code != 200:
                raise RuntimeError(f"Error downloading {url}: {r.status_code}")
        os.makedirs(os.path.dirname(local_fn), exist_ok=True)
        with open(local_fn, "wb") as f:
            f.write(r.content)


# DefaultDataset = DatasetManager
core_dataset = DatasetManager("core", source="https://gin.g-node.org/neuroboros/core")
