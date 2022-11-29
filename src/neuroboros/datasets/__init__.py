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
import numpy as np
from scipy.stats import zscore
import datalad.api as dl

from surface_spaces import get_cortical_mask

from neuroboros.path import data_root, dl_root


class Dataset:
    def __init__(self, name, space, interp, denoise, dl_source,
            fp_version='21.0.2'):
        self.name = name
        self.fp_version = fp_version
        self.dl_source = dl_source
        if dl_source is not None:
            self.dl_dset = dl.install(
                path=os.path.join(dl_root, self.name),
                source=self.dl_source)
        self.space = space
        self.interp = interp
        self.denoise = denoise

    def load_data(self, sid, task, run, lr, space, interp, denoise):
        fn = os.path.join(f'{self.name}_{self.fp_version}', space, interp,
            denoise, f'{sid}_{task}_{run:02d}_{lr}h.npy')

        local_fn = os.path.join(data_root, 'datasets', fn)
        if os.path.exists(local_fn):
            ds = np.load(local_fn)
        elif self.dl_source is not None:
            result = self.dl_dset.get(fn)[0]
            if result['status'] not in ['ok', 'notneeded']:
                raise RuntimeError(
                    f"datalad `get` status is {result['status']}, likely due to "
                    "problems downloading the file.")
            ds = np.load(result['path'])
        else:
            raise RuntimeError(f"Cannot locate required data file {fn}.")

        return ds

    def get_data(
            self, sid, task, run, lr, z=True, mask=False,
            space=None, interp=None, denoise=None):

        if isinstance(run, Iterable):
            ds = [self.get_data(
                      sid, task, run_, lr, z, mask, space, interp, denoise)
                  for run_ in run]
            ds = np.concatenate(ds, axis=0)
            return ds

        if space is None:
            space = self.space
        if interp is None:
            interp = self.interp

        ds = self.load_data(
            sid, task, run, lr, space=space, interp=interp, denoise=denoise)

        if mask:
            m = get_cortical_mask(lr, space, mask)
            ds = ds[:, m]
        if z:
            ds = np.nan_to_num(zscore(ds, axis=0))

        return ds


class Forrest(Dataset):
    def __init__(
            self, space='onavg-ico32', interp='2step_normals_equal',
            denoise='no-gsr', fp_version='21.0.2'):

        super().__init__(
            name='forrest',
            space=space,
            interp=interp,
            denoise=denoise,
            dl_source='https://gin.g-node.org/feilong/nidata-forrest',
            fp_version=fp_version,
        )
        self.subjects = ['01', '02', '03', '04', '05', '06', '09', '10', '14', '15', '16', '17', '18', '19', '20']
        self.tasks = ["forrest", "movielocalizer", "objectcategories", "retmapccw", "retmapclw", "retmapcon", "retmapexp"]
