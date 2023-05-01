import os
import numpy as np
import neuroboros as nb

from .io import load_file


def load_npz(npz_fn):
    npz = np.load(npz_fn)
    sls = np.array_split(npz['concatenated_searchlights'], npz['boundaries'])
    dists = np.array_split(npz['concatenated_distances'], npz['boundaries'])
    return sls, dists


def load_searchlights(lr, radius, space, center_space=None, **kwargs):
    group = kwargs.get('group', 'on1031')
    dist_type = kwargs.get('dist_type', 'dijkstra')
    avg_type = kwargs.get('avg_type', 'trimmed')
    if center_space is None:
        center_space = space
    npz_fn = os.path.join(
        space, 'searchlights', f'{center_space}_center', f'{lr}h',
        f'{group}_{avg_type}', f'{dist_type}_{radius}mm.npz')
    sls, dists = load_file(npz_fn, load_func=load_npz)
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
        The new list of distnaces to searchlight center after conversion.
    """
    if mask is not False:
        cortical_indices = np.where(mask)[0]
        mapping = np.cumsum(mask) - 1

    sls_new, dists_new = [], []
    for i, (sl, d) in enumerate(zip(sls, dists)):
        if center_mask is False or center_mask[i]:
            m = (d <= radius)
            if mask is not False:
                m = np.logical_and(np.isin(sl, cortical_indices), m)
                sls_new.append(mapping[sl[m]])
            else:
                sls_new.append(sl[m])
            dists_new.append(d[m])

    return sls_new, dists_new


def get_searchlights(lr, radius, space='onavg-ico32', center_space=None, mask=True,
                     center_mask=None, return_dists=False, **kwargs):
    radius_ = 20

    if center_mask is None:
        center_mask = mask

    if mask and not isinstance(mask, np.ndarray):
        mask = nb.mask(lr, space, **kwargs)

    if center_space is None:
        center_space = space
        center_mask = mask
    else:
        if center_mask is True:
            center_mask = nb.mask(lr, center_space, **kwargs)

    sls, dists = load_searchlights(lr, radius_, space,
        center_space=center_space, **kwargs)
    sls, dists = convert_searchlights(sls, dists, radius, mask, center_mask)

    if return_dists:
        return sls, dists
    return sls
