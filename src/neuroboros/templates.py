"""Hyperalignment templates."""

import json
import os
import warnings

import numpy as np

from .io import hyperbrain_dataset
from .spaces import get_mapping, get_mask

DEFAULT_TEMPLATE_VERSION = "sparta-0.9"
MANIFEST_FN = "manifest.json"
TEMPLATE_SPACE = "onavg-ico64"
TARGET_SPACE = "onavg-ico8"

_manifest = None


def _parse_version(version):
    """Split ``"sparta-0.9"`` into ``("sparta", "0.9")``."""
    name, sep, number = version.rpartition("-")
    if not sep or not name or not number:
        raise ValueError(
            f"Invalid template version {version!r}. Expected "
            "'<name>-<number>', e.g., 'sparta-0.9'."
        )
    return name, number


def _available_versions(manifest):
    versions = set()
    for fn in manifest:
        parts = fn.split("/")
        if len(parts) >= 3 and parts[1].startswith("v"):
            versions.add(f"{parts[0]}-{parts[1][1:]}")
    return sorted(versions)


def _parse_manifest(manifest):
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise ValueError(
            "Invalid hyperbrain manifest: expected a JSON object with a "
            "'files' list."
        )
    return set(manifest["files"])


def load_manifest():
    """Set of file paths available in the ``hyperbrain`` dataset.

    The manifest is downloaded once and cached locally. Use
    :func:`update_manifest` to refresh it.
    """
    global _manifest
    if _manifest is None:
        manifest = hyperbrain_dataset.get(MANIFEST_FN, on_missing="raise")
        _manifest = _parse_manifest(manifest)
    return _manifest


def update_manifest():
    """Refresh the local manifest of the ``hyperbrain`` dataset.

    Call this after a new template version has been published so that it
    can be requested with :func:`get_template`.

    Returns
    -------
    manifest : dict
        The parsed ``manifest.json``.
    """
    global _manifest
    if hyperbrain_dataset.kind == "datalad":
        hyperbrain_dataset.dset.update(how="merge")
    elif hyperbrain_dataset.kind == "https":
        fn = os.path.join(hyperbrain_dataset.root, MANIFEST_FN)
        if os.path.exists(fn):
            # Keep the old manifest until the new one has been downloaded
            # and validated.
            new_fn = fn + ".new"
            try:
                hyperbrain_dataset.download(MANIFEST_FN, new_fn, on_missing="raise")
                with open(new_fn) as f:
                    _parse_manifest(json.load(f))
            except BaseException:
                if os.path.exists(new_fn):
                    os.remove(new_fn)
                raise
            os.replace(new_fn, fn)
    manifest = hyperbrain_dataset.get(MANIFEST_FN, on_missing="raise")
    _manifest = _parse_manifest(manifest)
    return manifest


def _check_manifest(path):
    rel = "/".join(path)
    manifest = load_manifest()
    if rel not in manifest:
        raise ValueError(
            f"{rel!r} is not in the hyperbrain manifest. Available template "
            f"versions: {_available_versions(manifest)}. If a new version was "
            "published recently, run `nb.tpl.update()` to refresh the manifest."
        )


def get_template(
    which,
    mask=True,
    version=DEFAULT_TEMPLATE_VERSION,
    space=TEMPLATE_SPACE,
    dtype=np.float64,
):
    """Connectivity hyperalignment template.

    Templates are stored in the ``hyperbrain`` dataset
    (https://gin.g-node.org/neuroboros/hyperbrain) and downloaded on first
    use. Each hemisphere's template has shape ``(n_targets, n_vertices)``:
    rows are connectivity targets (connectivity patterns) and columns are
    the vertices of that hemisphere (connectivity profiles). The targets are
    the ``onavg-ico8`` vertices of the left hemisphere followed by those of
    the right hemisphere, the same order as ``nb.mask("lr", "onavg-ico8")``.

    Parameters
    ----------
    which : {'l', 'r', 'lr', 'meta'}
        Hemisphere of the template ('lh' and 'rh' are also accepted). 'lr'
        concatenates the two hemispheres along the vertex axis. 'meta'
        returns the metadata of the template version as a dict.
    mask : bool, default=True
        Whether to apply the standard cortical masks: the ``onavg-ico8`` mask
        of both hemispheres to the targets (rows) and the mask of ``space``
        for the requested hemisphere to the vertices (columns). Ignored
        when ``which`` is 'meta'.
    version : str, default='sparta-0.9'
        Template name and version, ``'<name>-<number>'``. Versions never
        change once published.
    space : str, default='onavg-ico64'
        Surface space of the vertices of the returned array. The templates
        are stored in ``onavg-ico64``; requesting another space resamples
        them with :func:`neuroboros.mapping` and emits a warning, because
        resampling may reduce data quality. Ignored when ``which`` is 'meta'.
    dtype : dtype or None, default=numpy.float64
        Data type of the returned array. If None, the data type stored in
        the file (float32) is kept. Ignored when ``which`` is 'meta'.

    Returns
    -------
    template : ndarray or dict
        Array of shape ``(n_targets, n_vertices)`` or, for 'meta', the
        parsed ``meta.json``.

    See Also
    --------
    update_manifest : Refresh the list of available template files.
    """
    if which == "lr":
        kwargs = dict(mask=mask, version=version, space=space, dtype=dtype)
        lh = get_template("l", **kwargs)
        rh = get_template("r", **kwargs)
        return np.concatenate([lh, rh], axis=1)

    name, number = _parse_version(version)
    if which == "meta":
        basename = "meta.json"
    elif which in ("l", "lh", "r", "rh"):
        lr = which[0]
        basename = f"{lr}h_{TEMPLATE_SPACE}.npy"
    else:
        raise ValueError(
            f"Invalid value for `which`: {which!r}. "
            "Expected 'l', 'r', 'lr', or 'meta'."
        )
    path = (name, f"v{number}", basename)
    _check_manifest(path)

    data = hyperbrain_dataset.get(path, on_missing="raise")
    if which == "meta":
        return data

    if data.ndim != 2:
        raise ValueError(
            f"Expected a 2-D template array, got shape {data.shape} for {path}."
        )
    file_dtype = data.dtype
    if mask:
        target_mask = get_mask("lr", TARGET_SPACE)
        vertex_mask = get_mask(lr, TEMPLATE_SPACE)
        data = data[np.ix_(target_mask, vertex_mask)]
    if space != TEMPLATE_SPACE:
        warnings.warn(
            f"Templates are stored in {TEMPLATE_SPACE!r}; requesting {space!r} "
            "forces resampling, which may reduce data quality.",
            stacklevel=2,
        )
        # `get_mapping` does not accept `mask=False`; None means no mask.
        M = get_mapping(lr, TEMPLATE_SPACE, space, mask=True if mask else None)
        data = np.asarray(data @ M)
    if dtype is None:
        dtype = file_dtype
    return data.astype(dtype, copy=False)


get_template.update = update_manifest
