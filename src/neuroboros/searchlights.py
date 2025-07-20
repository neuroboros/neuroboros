import os

import numpy as np

from .io import core_dataset
from .spaces import get_mask


def load_npz(npz_fn):
    npz = np.load(npz_fn)
    sls = np.array_split(npz["concatenated_searchlights"], npz["boundaries"])
    dists = np.array_split(npz["concatenated_distances"], npz["boundaries"])
    return sls, dists


def load_searchlights(lr, radius, space, center_space=None, **kwargs):
    species = "macaque" if space.startswith("mkavg-") else "human"
    if species == "human":
        group, dist_type, avg_type = "on1031", "dijkstra", "trimmed"
    elif species == "macaque":
        group, dist_type, avg_type = "mk12", "geodesic", "average"
    else:
        raise ValueError(f"Unknown species: {species}")
    group = kwargs.get("group", group)
    dist_type = kwargs.get("dist_type", dist_type)
    avg_type = kwargs.get("avg_type", avg_type)
    if center_space is None:
        center_space = space
    npz_fn = os.path.join(
        space,
        "searchlights",
        f"{center_space}_center",
        f"{lr}h",
        f"{group}_{avg_type}",
        f"{dist_type}_{radius}mm.npz",
    )
    sls, dists = core_dataset.get(npz_fn, load_func=load_npz, on_missing="raise")
    return sls, dists


def convert_searchlights(sls, dists, radius, mask, center_mask):
    """
    Convert larger-radius/unmasked searchlights to smaller-radius/masked.

    Parameters
    ----------
    sls : list of ndarray
        Each entry is an ndarray of integers, which are the indices of the
        vertices in a searchlight.
    dists : list of ndarray
        Each entry is an ndarray of float numbers, which are the distances
        between vertices in a searchlight and the center of the
        searchlight. The order of vertices are the same as ``sls``.
    radius : int or float
        The searchlight radius in mm.
    mask : ndarray
        A boolean ndarray indicating whether a vertex belongs to the cortex
        (True) or not (False).
    center_mask : ndarray
        A boolean ndarray indicating whether a searchlight center belongs to
        the cortex (True) or not (False). Mainly used in sparse searchlights
        where the number of searchlights is smaller than the number of
        vertices.

    Returns
    -------
    sls_new : list of ndarray
        The new list of searchlight indices after conversion.
    dists_new : list of ndarray
        The new list of distances to searchlight center after conversion.
    """
    if mask is not False:
        cortical_indices = np.where(mask)[0]
        mapping = np.cumsum(mask) - 1

    sls_new, dists_new = [], []
    for i, (sl, d) in enumerate(zip(sls, dists)):
        if center_mask is False or center_mask[i]:
            m = d <= radius
            if mask is not False:
                m = np.logical_and(np.isin(sl, cortical_indices), m)
                sls_new.append(mapping[sl[m]])
            else:
                sls_new.append(sl[m])
            dists_new.append(d[m])

    return sls_new, dists_new


def get_searchlights(
    lr,
    radius,
    space="onavg-ico32",
    center_space=None,
    mask=True,
    center_mask=None,
    return_dists=False,
    **kwargs,
):
    """
    Get a list of vertex indices for searchlight analysis.

    Parameters
    ----------
    lr : {'l', 'r', 'lr'}
        'l' for left hemisphere, 'r' for right hemisphere, 'lr' for both.
    radius : int or float
        The searchlight radius in mm.
    space : str, default='onavg-ico32'
        The space in which the searchlights are defined. It should match the
        space of the data to be analyzed.
    center_space : str, default=None
        The space in which the searchlight centers are defined. If None, it
        will be the same as `space`.
    mask : bool or ndarray, default=True
        If True, use the default mask for the specified space. If False, no
        mask is applied. If an ndarray, it should be a boolean array indicating
        which vertices are included in the searchlight. The mask should match
        the mask applied to the data to be analyzed.
    center_mask : bool or ndarray, default=None
        If True, use the default mask for the center space. If False, no mask
        is applied. If an ndarray, it should be a boolean array indicating
        which vertices are included in the searchlight center.
    return_dists : bool, default=False
        If True, return the distances of the vertices in the searchlight to
        the center of the searchlight.
    **kwargs : dict
        Additional keyword arguments passed to the searchlight loading function.

    Returns
    -------
    sls : list of ndarray
        Each entry is an ndarray of integers, which are the indices of the
        vertices in a searchlight.
    dists : list of ndarray, optional
        Each entry is an ndarray of float numbers, which are the distances
        between vertices in a searchlight and the center of the searchlight.
        Only returned if `return_dists` is True.
    """
    if lr == "lr":
        nv = 0
        if isinstance(mask, np.ndarray):
            raise NotImplementedError
        m = get_mask("l", space, **kwargs)
        nv = m.sum() if mask else m.size
        kwargs.update(
            {
                "radius": radius,
                "space": space,
                "center_space": center_space,
                "mask": mask,
                "center_mask": center_mask,
                "return_dists": True,
            }
        )
        sls_l, dists_l = get_searchlights("l", **kwargs)
        sls_r, dists_r = get_searchlights("r", **kwargs)

        # Ensure dtype is sufficient using int64
        sls_r = [s.astype(np.int64) + nv for s in sls_r]
        sls = sls_l + sls_r

        if return_dists:
            dists = dists_l + dists_r
            return sls, dists

        return sls

    radius_ = 20
    if radius > radius_:
        raise ValueError(
            f"Searchlight radius {radius}mm is larger than the maximum "
            f"supported radius {radius_}mm. Please use a smaller radius."
        )

    if center_mask is None:
        center_mask = mask

    if not isinstance(mask, np.ndarray) and mask:
        mask = get_mask(lr, space, **kwargs)

    if center_space is None:
        center_space = space
        center_mask = mask
    else:
        if center_mask is True:
            center_mask = get_mask(lr, center_space, **kwargs)

    sls, dists = load_searchlights(
        lr, radius_, space, center_space=center_space, **kwargs
    )
    sls, dists = convert_searchlights(sls, dists, radius, mask, center_mask)

    if return_dists:
        return sls, dists
    return sls
