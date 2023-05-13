import os
import numpy as np
import scipy.sparse as sparse

from .io import load_file


MEASURES = [
    'area', 'area.mid', 'area.pial',
    'curv', 'curv.pial',
    'jacobian_white', 'sulc', 'thickness', 'volume']
PARCELLATIONS = ['aparc', 'aparc.DKTatlas', 'aparc.a2009s']


def get_morphometry(which, lr, space='onavg-ico32', **kwargs):
    """Group-based morphometry measure.

    Parameters
    ----------
    which : str
        Which morphometric measure to get. One of the following:
        'area', 'area.mid', 'area.pial', 'curv', 'curv.pial',
        'jacobian_white', 'sulc', 'thickness', 'volume'.
    lr : str
        Hemisphere, either 'l' or 'r'.
    space : str, default='onavg-ico32'
        Surface space.

    Returns
    -------
    measure : ndarray
        Morphometric measure. Shape (n_vertices,).
    """
    assert which in MEASURES
    group = kwargs.get('group', 'on1031')
    resample = kwargs.get('resample', 'overlap-8div')
    avg_type = kwargs.get('avg_type', 'trimmed')
    assert lr in 'lr'
    fn = os.path.join(space, 'morphometry', which, f'{lr}h',
                      f'{group}_{avg_type}', f'{resample}.npy')
    measure = load_file(fn)
    return measure


def get_parcellation(which, lr, space='onavg-ico32', prob=False, **kwargs):
    """Group-based parcellation.

    Parameters
    ----------
    which : str
        Which parcellation to get. One of the following:
        'aparc', 'aparc.DKTatlas', 'aparc.a2009s'.
    lr : str
        Hemisphere, either 'l' or 'r'.
    space : str, default='onavg-ico32'
        Surface space.
    prob : bool, default=False
        Whether to load the probabilistic version of the parcellation.

    Returns
    -------
    parc : ndarray
        The cortical parcellation. It is a discrete parcellation if ``prob``
        is False, and a probabilistic parcellation if ``prob`` is True.
    """
    assert which in PARCELLATIONS
    group = kwargs.get('group', 'on1031')
    resample = kwargs.get('resample', 'overlap-8div')
    avg_type = kwargs.get('avg_type', 'trimmed')
    assert lr in 'lr'
    if prob:
        basename = f'{resample}_prob.npy'
    else:
        basename = f'{resample}_parc.npy'
    fn = os.path.join(space, 'parcellations', which, f'{lr}h',
                      f'{group}_{avg_type}', basename)
    parc = load_file(fn)
    return parc


def get_mask(lr, space='onavg-ico32', legacy=False, **kwargs):
    """Standard cortical mask.

    Parameters
    ----------
    lr : str
        Hemisphere, either 'l' or 'r'.
    space : str, default='onavg-ico32'
        Surface space.
    legacy : bool, default=False
        Whether to load the legacy version of the mask, e.g., "fsaverage" or
        "32k_fs_LR", instead of the group-based cortical mask.

    Returns
    -------
    mask : ndarray
        Mask of the cortical surface. Shape (n_vertices,).
    """
    resample = kwargs.get('resample', 'overlap-8div')
    which = kwargs.get('which', 'aparc.a2009s')
    assert lr in 'lr'
    if legacy:
        if space.startswith('fsavg-ico'):
            flavor = kwargs.get('flavor', 'fsaverage')
        elif space.startswith('fslr-ico'):
            flavor = kwargs.get('flavor', '32k_fs_LR')
        fn = os.path.join(space, 'masks', which, f'{lr}h',
                          f'{flavor}', f'{resample}.npy')
    else:
        group = kwargs.get('group', 'on1031')
        avg_type = kwargs.get('avg_type', 'trimmed')
        fn = os.path.join(space, 'masks', which, f'{lr}h',
                          f'{group}_{avg_type}', f'{resample}.npy')
    mask = load_file(fn)
    return mask


def get_geometry(which, lr, space='onavg-ico32', vertices_only=False, **kwargs):
    """Surface geometry.

    Parameters
    ----------
    which : str
        Which geometry to get. One of the following:
        'sphere', 'sphere.reg', 'white', 'pial', 'inflated', 'midthickness',
        'faces'.
    lr : str
        Hemisphere, either 'l' or 'r'.
    space : str, default='onavg-ico32'
        Surface space.
    vertices_only : bool, default=False
        Whether to return only the coordinates of the vertices.

    Returns
    -------
    coords : ndarray
        Coordinates of the vertices. Shape (n_vertices, 3).
        When ``which == 'faces'``, ``coords`` will not be returned.
    faces : ndarray
        Faces of the triangle surface mesh. Shape (n_faces, 3).
        Only returned when ``vertices_only == False``.
    """
    group = kwargs.get('group', 'on1031')
    avg_type = kwargs.get('avg_type', 'trimmed')
    assert lr in 'lr'
    if not vertices_only:
        ffn = os.path.join(space, 'geometry', 'faces', f'{lr}h.npy')
        faces = load_file(ffn)
        if which == 'faces':
            return faces

    if which in ['sphere', 'sphere.reg']:
        fn = os.path.join(space, 'geometry', 'sphere.reg', f'{lr}h.npy')
    else:
        fn = os.path.join(space, 'geometry', which, f'{lr}h',
                f'{group}_{avg_type}.npy')
    coords = load_file(fn)
    if vertices_only:
        return coords

    return coords, faces


def get_distances(lr, source, target=None, mask=None,
                  source_mask=None, target_mask=None, **kwargs):
    """Distances between vertices.

    Parameters
    ----------
    lr : str
        Hemisphere, either 'l' or 'r'.
    source : str
        Source space.
    target : str, default=None
        Target space. If None, assuming it's the same as ``source``.
    mask : str, default=None
        Mask to apply to the distance matrix. If None or False, no mask is
        applied. If True, the group-based cortical mask is used.
        If a boolean array, it is used as the mask.
        ``source_mask`` and ``target_mask`` can be set separately, and they
        take precedence over ``mask``.
    source_mask : str, default=None
        Mask to apply to source space (rows) of the distance matrix.
    target_mask : str, default=None
        Mask to apply to target space (columns) of the distance matrix.

    Returns
    -------
    M : ndarray
        Distance matrix. The shape is (n_source_vertices, n_target_vertices).
    """
    group = kwargs.get('group', 'on1031')
    avg_type = kwargs.get('avg_type', 'trimmed')
    dist_type = kwargs.get('dist_type', 'dijkstra')

    if target is None:
        target = source
    if source_mask is None:
        source_mask = mask
    if target_mask is None:
        target_mask = mask

    fn = os.path.join(source, 'distances', f'to_{target}', f'{lr}h',
                      f'{group}_{avg_type}', f'{dist_type}.npy')

    assert target == source
    d = load_file(fn)
    ico = int(source.split('-ico')[1])
    nv = ico**2 * 10 + 2
    mat = np.zeros((nv, nv), dtype=d.dtype)
    idx1, idx2 = np.triu_indices(nv, 1)
    mat[idx1, idx2] = d
    mat = np.maximum(mat, mat.T)

    if source_mask is not None:
        if isinstance(source_mask, np.ndarray):
            mask1 = source_mask
        else:
            mask1 = get_mask(lr, source, **kwargs)
    if target_mask is not None:
        if isinstance(target_mask, np.ndarray):
            mask2 = target_mask
        else:
            mask2 = get_mask(lr, target, **kwargs)

    if source_mask is not None and target_mask is not None:
        M = mat[np.ix_(mask1, mask2)]
    elif source_mask is not None:
        M = mat[mask1, :]
    elif target_mask is not None:
        M = mat[:, mask2]
    else:
        M = mat

    return M


def smooth(lr, fwhm, space='onavg-ico32', mask=None, keep_sum=False):
    """Get a smoothing matrix.

    Parameters
    ----------
    lr : str
        Hemisphere, either 'l' or 'r'.
    fwhm : float
        Full-width at half-maximum of the Gaussian kernel.
    space : str, default='onavg-ico32'
        Surface space where the data is in.
    mask : ndarray or bool or None, default=None
        Mask to apply to the smoothing matrix. If None or False, no mask is
        applied. If True, the standard cortical mask is used. If an ndarray,
        it is used as the mask.
    keep_sum : bool, default=False
        If True, keep the sum of the data. Useful for transforming area,
        volume, etc., where the total area/volume is preserved when
        ``keep_sum=True``.

    Returns
    -------
    M : sparse matrix
        Smoothing matrix. Can be applied to data matrix ``X`` as ``X @ M``.
    """
    d = get_distances(lr, space, space, mask=mask)
    s2 = fwhm / (4. * np.log(2))
    weights = np.exp(-d**2/ s2)
    mat = sparse.csr_matrix(weights)

    if keep_sum:
        with np.errstate(divide='ignore'):
            d = np.nan_to_num(np.reciprocal(mat.sum(axis=1).A.ravel()))
        M = sparse.diags(d) @ mat
    else:
        with np.errstate(divide='ignore'):
            d = np.nan_to_num(np.reciprocal(mat.sum(axis=0).A.ravel()))
        M = mat @ sparse.diags(d)

    return M


def get_mapping(lr, source, target, mask=None, nn=False, keep_sum=False,
                source_mask=None, target_mask=None, **kwargs):
    """Get mapping (transform) from one space to another.

    Parameters
    ----------
    lr : str
        Hemisphere, either 'l' or 'r'.
    source : str
        Source space, the space where the data is currently in.
    target : str
        Target space, the space where the data will be transformed into.
    mask : bool or boolean array or None
        Mask to apply to the mapping. If None or False, no mask is applied.
        If True, the group-based cortical mask is used.
        If a boolean array, it is used as the mask.
        ``source_mask`` and ``target_mask`` can be set separately, and they
        take precedence over ``mask``.
    nn : bool
        If True, use nearest neighbor interpolation.
        If False, use overlap-area-based interpolation.
    keep_sum : bool
        If True, keep the sum of the data. Useful for transforming area,
        volume, etc., where the total area/volume is preserved when
        ``keep_sum=True``.
    source_mask : bool or boolean array or None
        Mask to apply to the source space. Similar to ``mask``.
    target_mask : bool or boolean array or None
        Mask to apply to the target space. Similar to ``mask``.

    Returns
    -------
    M : sparse matrix
        Mapping matrix. Can be applied to data in the source space to
        transform it into the target space. For example, if ``X`` is a
        data matrix in the source space, then ``X @ M`` is the data matrix
        in the target space.
    """
    group = kwargs.get('group', 'on1031')
    resample = kwargs.get('resample', 'overlap-8div')
    avg_type = kwargs.get('avg_type', 'trimmed')

    if source_mask is None:
        source_mask = mask
    if target_mask is None:
        target_mask = mask

    if source == target:
        ico = int(source.split('-ico')[1])
        nv = ico**2 * 10 + 2
        mat = sparse.diags(np.ones((nv, ))).tocsr()
        # print(mat.shape, mat.data.shape, type(mat))
    else:
        fn1 = os.path.join(source, 'mapping', f'to_{target}', f'{lr}h',
                        f'{group}_{avg_type}', f'{resample}.npz')
        fn2 = os.path.join(target, 'mapping', f'to_{source}', f'{lr}h',
                        f'{group}_{avg_type}', f'{resample}.npz')
        mat1 = load_file(fn1)
        mat2 = load_file(fn2)
        assert mat1 is not None or mat2 is not None, \
            f'Neither {fn1} nor {fn2} exists.'
        mat = mat1 if mat1 is not None else mat2.T

    if source_mask is not None and source_mask is not False:
        if isinstance(source_mask, np.ndarray):
            mask1 = source_mask
        else:
            mask1 = get_mask(lr, source, **kwargs)
    if target_mask is not None and target_mask is not False:
        if isinstance(target_mask, np.ndarray):
            mask2 = target_mask
        else:
            mask2 = get_mask(lr, target, **kwargs)

    if source_mask is not None and target_mask is not None:
        mat = mat[np.ix_(mask1, mask2)]
    elif source_mask is not None:
        mat = mat[mask1, :]
    elif target_mask is not None:
        mat = mat[:, mask2]
    else:
        pass

    if nn:
        idx = mat.argmax(axis=0).A.ravel()
        m, n = mat.shape
        M = sparse.csr_matrix((np.ones((n, )), (idx, np.arange(n))), (m, n))
        return M

    if keep_sum:
        with np.errstate(divide='ignore'):
            d = np.nan_to_num(np.reciprocal(mat.sum(axis=1).A.ravel()))
        M = sparse.diags(d) @ mat
    else:
        with np.errstate(divide='ignore'):
            d = np.nan_to_num(np.reciprocal(mat.sum(axis=0).A.ravel()))
        M = mat @ sparse.diags(d)

    return M
