"""
================================================
Neuroboros datasets (:mod:`neuroboros.datasets`)
================================================

.. currentmodule:: neuroboros.datasets

.. autosummary::
    :toctree:

    Forrest - The StudyForrest 3 T dataset.


"""

import os
from collections.abc import Iterable
from functools import partial
import numpy as np
import pandas as pd
from scipy.stats import zscore
import datalad.api as dl

from neuroboros.spaces import get_mask

from neuroboros.path import data_root, dl_root

SURFACE_SPACES = ['fsavg-ico32', 'onavg-ico32', 'onavg-ico48', 'onavg-ico64']
SURFACE_RESAMPLES = ['1step_pial_overlap', '1step_pial_area', '2step_normals-equal_nnfr', '2step_normals-sine_nnfr']
VOLUME_SPACES = ['mni-2mm', 'mni-3mm', 'mni-4mm']
VOLUME_RESAMPLES = ['1step_linear_overlap', '1step_fmriprep_overlap']


def guess_surface_volume(space, resample, lr):
    if space in SURFACE_SPACES or resample in SURFACE_RESAMPLES:
        return 'surface'
    if space in VOLUME_SPACES or resample in VOLUME_RESAMPLES:
        return 'volume'
    if lr in ['l', 'r', 'l-cerebrum', 'r-cerebrum']:
        return 'surface'
    return 'volume'


def default_prep(ds, confounds, cortical_mask, z=True, mask=True):
    if mask and cortical_mask is not None:
        ds = ds[:, cortical_mask]
    conf = confounds[0]
    beta = np.linalg.lstsq(conf, ds, rcond=None)[0]
    ds = ds - conf @ beta
    if z:
        ds = np.nan_to_num(zscore(ds, axis=0))
    return ds


def scrub_prep(ds, confounds, cortical_mask, z=True, mask=True):
    if mask and cortical_mask is not None:
        ds = ds[:, cortical_mask]
    conf, _, keep = confounds
    beta = np.linalg.lstsq(conf[keep], ds[keep], rcond=None)[0]
    ds = ds[keep] - conf[keep] @ beta
    if z:
        ds = np.nan_to_num(zscore(ds, axis=0))
    return ds, keep


def get_prep(name, **kwargs):
    prep = {
        'default': default_prep,
        'scrub': scrub_prep,
    }[name]
    if kwargs:
        prep = partial(prep, **kwargs)
    return prep


def _follow_symlink(fn, root):
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


class Dataset:
    def __init__(
            self, name, dl_source, space, resample, prep='default',
            fp_version='20.2.7'):
        self.name = name
        self.dl_source = dl_source
        self.dl_dset = dl.install(
            path=os.path.join(dl_root, self.name),
            source=self.dl_source)
        self.fp_version = fp_version
        self.surface_space = None
        self.volume_space = None
        self.surface_resample = None
        self.volume_resample = None
        if not isinstance(space, (tuple, list)):
            space = [space]
        for sp in space:
            if sp in SURFACE_SPACES:
                self.surface_space = sp
            elif sp in VOLUME_SPACES:
                self.volume_space = sp
            else:
                raise ValueError(f"space {sp} not recognized.")
        if not isinstance(resample, (tuple, list)):
            resample = [resample]
        for resamp in resample:
            if resamp in SURFACE_RESAMPLES:
                self.surface_resample = resamp
            elif resamp in VOLUME_RESAMPLES:
                self.volume_resample = resamp
            else:
                raise ValueError(
                    f"Resampling method {resamp} not recognized.")
        self.prep = prep

    def load_data(self, sid, task, run, lr, space, resample, fp_version=None):
        if fp_version is None:
            fp_version = self.fp_version

        if lr in ['l', 'r']:
            lr = f'{lr}-cerebrum'

        fn = os.path.join(
            fp_version, 'renamed', space, lr, resample,
            f'sub-{sid}_task-{task}_run-{run:02d}.npy')
        fn = _follow_symlink(fn, self.dl_dset.path)

        fn = download_datalad_file(fn, self.dl_dset)
        # result = self.dl_dset.get(fn)[0]
        # if result['status'] not in ['ok', 'notneeded']:
        #     raise RuntimeError(
        #         f"datalad `get` status is {result['status']}, likely due to "
        #         "problems downloading the file.")
        ds = np.load(fn).astype(np.float64)

        return ds

    def load_confounds(self, sid, task, run, fp_version=None):
        if fp_version is None:
            fp_version = self.fp_version
        suffix_li = [
            'desc-confounds_timeseries.npy',
            'desc-confounds_timeseries.tsv',
            'desc-mask_timeseries.npy']
        output = []
        for suffix in suffix_li:
            fn = os.path.join(
                fp_version, 'renamed_confounds',
                f'sub-{sid}_task-{task}_run-{run:02d}_{suffix}')
            fn = _follow_symlink(fn, self.dl_dset.path)
            fn = download_datalad_file(fn, self.dl_dset)

            if fn.endswith('.npy'):
                output.append(np.load(fn))
            else:
                output.append(pd.read_csv(fn, delimiter='\t', na_values='n/a'))
        return output

    def get_data(self, sid, task, run, lr, space=None, resample=None,
                 prep=None, fp_version=None, force_volume=False,
                 prep_kwargs=None):
        if force_volume:
            space_kind = 'volume'
        else:
            space_kind = guess_surface_volume(space, resample, lr)
        if space is None:
            space = {
                'surface': self.surface_space,
                'volume': self.volume_space,
            }[space_kind]
        if resample is None:
            resample = {
                'surface': self.surface_resample,
                'volume': self.volume_resample,
            }[space_kind]
        if prep is None:
            prep = self.prep
        if fp_version is None:
            fp_version = self.fp_version

        ds = self.load_data(sid, task, run, lr, space, resample, fp_version)
        confounds = self.load_confounds(sid, task, run, fp_version)
        if space_kind == 'surface':
            cortical_mask = get_mask(lr, space)
        else:
            cortical_mask = None
        if isinstance(prep, str):
            prep = get_prep(prep, **(prep_kwargs if prep_kwargs is not None else {}))
        ds = prep(ds, confounds, cortical_mask)
        return ds


class Bologna(Dataset):
    def __init__(
            self, space=['onavg-ico32', 'mni-4mm'],
            resample=['1step_pial_overlap', '1step_linear_overlap'],
            prep='default', fp_version='20.2.7'):
        name = 'bologna'
        dl_source = 'git@github.com:feilong/bologna.git'
        super().__init__(name, dl_source, space, resample, prep, fp_version)
        self.subjects = [f'{_+1:02d}' for _ in range(30)]
        self.tasks = ['rest']


class Forrest(Dataset):
    def __init__(
            self, space=['onavg-ico32', 'mni-4mm'],
            resample=['1step_pial_overlap', '1step_linear_overlap'],
            prep='default', fp_version='20.2.7'):
        name = 'forrest'
        dl_source = 'https://gin.g-node.org/neuroboros/forrest'
        super().__init__(name, dl_source, space, resample, prep, fp_version)
        self.subjects = ['01', '02', '03', '04', '05', '06', '09', '10', '14', '15', '16', '17', '18', '19', '20']
        self.tasks = ["forrest", "movielocalizer", "objectcategories", "retmapccw", "retmapclw", "retmapcon", "retmapexp"]
