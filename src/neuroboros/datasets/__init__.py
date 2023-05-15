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
import warnings
from collections.abc import Iterable
from functools import partial
import numpy as np
import pandas as pd
from scipy.stats import zscore
import datalad.api as dl

from ..spaces import get_mask
from ..io import load_file, DATA_ROOT


SURFACE_SPACES = ['fsavg-ico32', 'onavg-ico32', 'onavg-ico48', 'onavg-ico64']
SURFACE_RESAMPLES = ['1step_pial_overlap', '1step_pial_area', '2step_normals-equal_nnfr', '2step_normals-sine_nnfr']
VOLUME_SPACES = ['mni-2mm', 'mni-3mm', 'mni-4mm']
VOLUME_RESAMPLES = ['1step_linear_overlap', '1step_fmriprep_overlap']


def guess_surface_volume(space, resample, lr):
    if space in SURFACE_SPACES or resample in SURFACE_RESAMPLES:
        return 'surface'
    if space in VOLUME_SPACES or resample in VOLUME_RESAMPLES:
        return 'volume'
    if lr in ['l', 'r', 'l-cerebrum', 'r-cerebrum', 'lr']:
        return 'surface'
    return 'volume'


def default_prep(ds, confounds, cortical_mask, z=True, mask=True, gsr=False):
    if mask and cortical_mask is not None:
        ds = ds[:, cortical_mask]
    conf = confounds[0]
    if gsr:
        gs = np.array(confounds[1]['global_signal'])
        conf = np.concatenate([conf, gs[:, np.newaxis]], axis=1)
    beta = np.linalg.lstsq(conf, ds, rcond=None)[0]
    ds = ds - conf @ beta
    if z:
        ds = np.nan_to_num(zscore(ds, axis=0))
    return ds


def scrub_prep(ds, confounds, cortical_mask, z=True, mask=True, gsr=False):
    if mask and cortical_mask is not None:
        ds = ds[:, cortical_mask]
    conf, _, keep = confounds
    if gsr:
        gs = np.array(confounds[1]['global_signal'])
        conf = np.concatenate([conf, gs[:, np.newaxis]], axis=1)
    beta = np.linalg.lstsq(conf[keep], ds[keep], rcond=None)[0]
    ds = ds[keep] - conf[keep] @ beta
    if z:
        ds = np.nan_to_num(zscore(ds, axis=0))
    return ds, keep


def get_prep(name, **kwargs):
    if name.endswith('-gsr'):
        gsr = True
        name = name[:-4]
    else:
        gsr = False

    prep = {
        'default': default_prep,
        'scrub': scrub_prep,
    }[name]
    if gsr:
        prep = partial(prep, gsr=True)
    if kwargs:
        prep = partial(prep, **kwargs)
    return prep


def _follow_symlink(fn, root):
    fn_ = os.path.join(root, fn)
    fn = os.path.join(os.path.dirname(fn_), os.readlink(fn_))
    fn = os.path.normpath(fn)
    fn = os.path.relpath(fn, root)
    return fn


# def download_datalad_file(fn, dl_dset):
#     result = dl_dset.get(fn)[0]
#     if result['status'] not in ['ok', 'notneeded']:
#         raise RuntimeError(
#             f"datalad `get` status is {result['status']}, likely due to "
#             "problems downloading the file.")
#     return result['path']


class Dataset:
    def __init__(
            self, name, dl_source, root_dir, space, resample,
            surface_space=None, surface_resample=None, volume_space=None,
            volume_resample=None, prep='default', fp_version='20.2.7'):
        self.name = name

        self.dl_source = dl_source
        self.root_dir = root_dir
        s = (dl_source is None) + (root_dir is None)
        if s == 0:
            raise ValueError('At least one of `dl_source` and `root_dir` '
                             'needs to be set.')
        if s > 1:
            warnings.warn('Both `dl_source` and `root_dir` are set. Will use '
                          '`root_dir`.')
        if root_dir is None:
            self.use_datalad = True
            path = os.path.join(DATA_ROOT, self.name)
            if os.path.exists(path):
                self.dl_dset = dl.Dataset(path)
            else:
                self.dl_dset = dl.install(
                    path=path,
                    source=self.dl_source)
        else:
            self.use_datalad = False

        self.fp_version = fp_version
        self.surface_space = surface_space
        self.volume_space = volume_space
        self.surface_resample = surface_resample
        self.volume_resample = volume_resample

        if space is not None:
            if not isinstance(space, (tuple, list)):
                space = [space]
            for sp in space:
                if sp in SURFACE_SPACES and self.surface_space is None:
                    self.surface_space = sp
                elif sp in VOLUME_SPACES and self.volume_space is None:
                    self.volume_space = sp
                else:
                    raise ValueError(f"space {sp} not recognized.")

        if resample is not None:
            if not isinstance(resample, (tuple, list)):
                resample = [resample]
            for resamp in resample:
                if resamp in SURFACE_RESAMPLES and self.surface_resample is None:
                    self.surface_resample = resamp
                elif resamp in VOLUME_RESAMPLES and self.volume_resample is None:
                    self.volume_resample = resamp
                else:
                    raise ValueError(
                        f"Resampling method {resamp} not recognized.")

        self.prep = prep

        self.renaming = load_file('rename.json.gz', dset=self.dl_dset, root=self.root_dir)

    def load_data(self, sid, task, run, lr, space, resample, fp_version=None):
        if lr == 'lr':
            ds = np.concatenate(
                [self.load_data(
                        sid, task, run, lr_, space, resample, fp_version)
                    for lr_ in 'lr'],
                axis=1)
            return ds

        if isinstance(run, (tuple, list)):
            ds = np.concatenate(
                [self.load_data(
                        sid, task, run_, lr, space, resample, fp_version)
                    for run_ in run],
                axis=0)
            return ds

        if fp_version is None:
            fp_version = self.fp_version

        if lr in ['l', 'r']:
            lr = f'{lr}-cerebrum'

        if self.renaming is None:
            fn = os.path.join(
                fp_version, 'resampled', space, lr, resample,
                f'sub-{sid}_task-{task}_run-{run:02d}.npy')
        else:
            fn = os.path.join(
                fp_version, 'renamed', space, lr, resample,
                f'sub-{sid}_task-{task}_run-{run:02d}.npy')
            fn = self.renaming[fn]

        # if self.use_datalad:
        #     fn = _follow_symlink(fn, self.dl_dset.path)
        ds = load_file(fn, dset=self.dl_dset, root=self.root_dir).astype(np.float64)
            # fn = download_datalad_file(fn, self.dl_dset)
        # ds = np.load(fn).astype(np.float64)

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
            if self.renaming is None:
                fn = os.path.join(
                    fp_version, 'confounds',
                    f'sub-{sid}_task-{task}_run-{run}_{suffix}')
            else:
                fn = os.path.join(
                    fp_version, 'renamed_confounds',
                    f'sub-{sid}_task-{task}_run-{run:02d}_{suffix}')
                fn = self.renaming[fn]
            # fn = _follow_symlink(fn, self.dl_dset.path)
            # fn = download_datalad_file(fn, self.dl_dset)
            o = load_file(fn, dset=self.dl_dset, root=self.root_dir)
            output.append(o)
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
            if lr == 'lr':
                cortical_mask = np.concatenate(
                    [get_mask(lr_, space) for lr_ in 'lr'], axis=0)
            else:
                cortical_mask = get_mask(lr, space)
        else:
            cortical_mask = None
        if isinstance(prep, str):
            prep = get_prep(prep, **(prep_kwargs if prep_kwargs is not None else {}))
        ds = prep(ds, confounds, cortical_mask)
        return ds

    def _get_anatomical_data(self, sid, which, lr, mask, space, fp_version):
        if fp_version is None:
            fp_version = self.fp_version
        if space is None:
            space = self.surface_space
        fn = os.path.join(fp_version, 'anatomy', space, 'overlap', which,
                          f'{sid}_{lr}h.npy')
        d = load_file(fn, dset=self.dl_dset, root=self.root_dir)
        d = d.astype(np.float64)
        if mask is not False and mask is not None:
            if isinstance(mask, np.ndarray):
                cortical_mask = mask
            else:
                cortical_mask = get_mask(lr, space)
            d = d[cortical_mask]
        return d

    def morphometry(self, sid, which, lr, mask=True, space=None,
                    fp_version=None):
        return self._get_anatomical_data(
            sid=sid, which=which, lr=lr, mask=mask, space=space,
            fp_version=fp_version)

    def parcellation(self, sid, which, lr, mask=True, space=None,
                    fp_version=None):
        return self._get_anatomical_data(
            sid=sid, which=which + '.annot', lr=lr, mask=mask, space=space,
            fp_version=fp_version)

class Bologna(Dataset):
    def __init__(
            self, space=['onavg-ico32', 'mni-4mm'],
            resample=['1step_pial_overlap', '1step_linear_overlap'],
            prep='default', fp_version='20.2.7'):
        name = 'bologna'
        dl_source = 'git@github.com:feilong/bologna.git'
        super().__init__(
            name, dl_source=dl_source, root_dir=None, space=space,
            resample=resample, prep=prep, fp_version=fp_version)
        self.subjects = [f'{_+1:02d}' for _ in range(30)]
        self.tasks = ['rest']


class Forrest(Dataset):
    def __init__(
            self, space=['onavg-ico32', 'mni-4mm'],
            resample=['1step_pial_overlap', '1step_linear_overlap'],
            prep='default', fp_version='20.2.7'):
        name = 'forrest'
        dl_source = 'https://gin.g-node.org/neuroboros/forrest'
        super().__init__(
            name, dl_source=dl_source, root_dir=None, space=space,
            resample=resample, prep=prep, fp_version=fp_version)
        self.subjects = ['01', '02', '03', '04', '05', '06', '09', '10', '14', '15', '16', '17', '18', '19', '20']
        self.tasks = ["forrest", "movielocalizer", "objectcategories", "retmapccw", "retmapclw", "retmapcon", "retmapexp"]


class Dalmatians(Dataset):
    def __init__(
            self, space=['onavg-ico32', 'mni-4mm'],
            resample=['1step_pial_overlap', '1step_linear_overlap'],
            prep='default', fp_version='20.2.7'):
        name = 'dalmatians'
        dl_source = 'git@github.com:feilong/dalmatians.git'
        super().__init__(
            name, dl_source=dl_source, root_dir=None, space=space,
            resample=resample, prep=prep, fp_version=fp_version)
        self.subjects = [
            'AB033', 'AB034', 'AB035', 'AB036', 'AB037', 'AB038', 'AB039',
            'AB041', 'AB042', 'AB043', 'AB053', 'AO003', 'AO004', 'AO005',
            'AO006', 'AO007', 'AO008', 'AO009', 'AO010', 'AO011', 'AO027',
            'AV012', 'AV013', 'AV014', 'AV015', 'AV016', 'AV017', 'AV018',
            'AV019', 'AV022', 'AV032', 'VD044', 'VD045', 'VD046', 'VD047',
            'VD048', 'VD049', 'VD050', 'VD051', 'VD052', 'VO020', 'VO021',
            'VO023', 'VO024', 'VO025', 'VO026', 'VO028', 'VO029', 'VO030',
            'VO031']
        self.tasks = ['dalmatians', 'scrambled']


class SpaceTop(Dataset):
    def __init__(self, space=['onavg-ico32', 'mni-4mm'],
            resample=['1step_pial_overlap', '1step_linear_overlap'],
            prep='default', fp_version='20.2.7'):
        name = 'spacetop'
        dl_source = 'git@github.com:feilong/spacetop.git'
        super().__init__(
            name, dl_source=dl_source, root_dir=None, space=space,
            resample=resample, prep=prep, fp_version=fp_version)
        self.tasks = ['faces']


class CamCAN(Dataset):
    def __init__(self, space=['onavg-ico32', 'mni-4mm'],
            resample=['1step_pial_overlap', '1step_linear_overlap'],
            prep='default', fp_version='20.2.7'):
        name = 'camcan'
        dl_source = 'git@github.com:feilong/camcan.git'
        super().__init__(
            name, dl_source=dl_source, root_dir=None, space=space,
            resample=resample, prep=prep, fp_version=fp_version)
        self.tasks = ['bang', 'rest', 'smt']

        self.subject_sets = {}
        mod_dir = os.path.dirname(os.path.realpath(__file__))
        for task in ['bang', 'rest', 'smt']:
            with open(os.path.join(mod_dir, f'camcan_{task}.txt'), 'r') as f:
                self.subject_sets[task] = f.read().splitlines()


class ID1000(Dataset):
    def __init__(self, space=['onavg-ico32', 'mni-4mm'],
            resample=['1step_pial_overlap', '1step_linear_overlap'],
            prep='default', fp_version='20.2.7'):
        name = 'id1000'
        dl_source = 'git@github.com:feilong/id1000.git'
        super().__init__(
            name, dl_source=dl_source, root_dir=None, space=space,
            resample=resample, prep=prep, fp_version=fp_version)
        self.tasks = ['moviewatching']

        mod_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(mod_dir, f'id1000.txt'), 'r') as f:
            self.subjects = f.read().splitlines()
