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
