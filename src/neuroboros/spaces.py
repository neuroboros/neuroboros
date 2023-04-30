import os
import numpy as np
import scipy.sparse as sparse

from .io import load_file


MEASURES = [
    'area', 'area.mid', 'area.pial',
    'curv', 'curv.pial',
    'jacobian_white', 'sulc', 'thickness', 'volume']
PARCELLATIONS = ['aparc', 'aparc.DKTatlas', 'aparc.a2009s']


def get_morphometry(which, lr, space, **kwargs):
    assert which in MEASURES
    group = kwargs.get('group', 'on1031')
    resample = kwargs.get('resample', 'overlap-8div')
    avg_type = kwargs.get('avg_type', 'trimmed')
    assert lr in 'lr'
    fn = os.path.join(space, 'morphometry', which, f'{lr}h',
                      f'{group}_{avg_type}', f'{resample}.npy')
    measure = load_file(fn)
    return measure


def get_parcellation(which, lr, space, prob=False, **kwargs):
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


def get_mask(lr, space, **kwargs):
    group = kwargs.get('group', 'on1031')
    resample = kwargs.get('resample', 'overlap-8div')
    avg_type = kwargs.get('avg_type', 'trimmed')
    which = kwargs.get('which', 'aparc.a2009s')
    assert lr in 'lr'
    fn = os.path.join(space, 'masks', which, f'{lr}h',
                      f'{group}_{avg_type}', f'{resample}.npy')
    mask = load_file(fn)
    return mask


def get_geometry(which, lr, space, **kwargs):
    group = kwargs.get('group', 'on1031')
    avg_type = kwargs.get('avg_type', 'trimmed')
    assert lr in 'lr'
    if which in ['sphere', 'sphere.reg']:
        fn = os.path.join(space, 'geometry', 'sphere.reg', f'{lr}h.npy')
    else:
        fn = os.path.join(space, 'geometry', which, f'{lr}h',
                f'{group}_{avg_type}.npy')
    coords = load_file(fn)
    ffn = os.path.join(space, 'geometry', 'faces', f'{lr}h.npy')
    faces = load_file(ffn)
    return coords, faces


def get_mapping(lr, source, target, mask=None, nn=False, keep_sum=False,
                source_mask=None, target_mask=None, **kwargs):
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
