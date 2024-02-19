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
from functools import partial

import numpy as np
from scipy.stats import zscore

from ..io import DatasetManager
from ..spaces import get_mask

SURFACE_SPACES = ['fsavg-ico32', 'onavg-ico32', 'onavg-ico48', 'onavg-ico64']
SURFACE_RESAMPLES = [
    '1step_pial_overlap',
    '1step_pial_area',
    '2step_normals-equal_nnfr',
    '2step_normals-sine_nnfr',
]
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


class Dataset:
    def __init__(
        self,
        name,
        dl_source,
        root_dir,
        space,
        resample,
        surface_space=None,
        surface_resample=None,
        volume_space=None,
        volume_resample=None,
        prep='default',
        fp_version='20.2.7',
    ):
        self.name = name

        self.dl_source = dl_source
        self.root_dir = root_dir

        self.dl_dset = DatasetManager(
            self.name, root=self.root_dir, source=self.dl_source
        )

        # if self.dl_source is None:
        #     if self.root_dir is not None:
        #         self.dl_dset = LocalDataset(self.name, self.root_dir)
        #     else:
        #         try:
        #             self.dl_dset = LocalDataset(self.name, self.root_dir)
        #         except AssertionError as e:
        #             raise RuntimeError(
        #                 "Dataset not found locally and `dl_source` not " "specified."
        #             ) from e
        # else:
        #     self.dl_dset = DefaultDataset(self.name, self.dl_source, self.root_dir)

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
                    raise ValueError(f"Resampling method {resamp} not recognized.")

        self.prep = prep

        self.renaming = self.dl_dset.get('rename.json.gz')

    def load_data(self, sid, task, run, lr, space, resample, fp_version=None):
        if lr == 'lr':
            dm = np.concatenate(
                [
                    self.load_data(sid, task, run, lr_, space, resample, fp_version)
                    for lr_ in 'lr'
                ],
                axis=1,
            )
            return dm

        if fp_version is None:
            fp_version = self.fp_version

        if lr in ['l', 'r']:
            lr = f'{lr}-cerebrum'

        if self.renaming is None:
            fn = [
                fp_version,
                'resampled',
                space,
                lr,
                resample,
                f'sub-{sid}_task-{task}_run-{run:02d}.npy',
            ]
        else:
            fn = [
                fp_version,
                'renamed',
                space,
                lr,
                resample,
                f'sub-{sid}_task-{task}_run-{run:02d}.npy',
            ]
            fn = self.renaming['/'.join(fn)].split('/')

        dm = self.dl_dset.get(fn, on_missing='raise').astype(np.float64)

        return dm

    def load_confounds(self, sid, task, run, fp_version=None):
        if fp_version is None:
            fp_version = self.fp_version
        suffix_li = [
            'desc-confounds_timeseries.npy',
            'desc-confounds_timeseries.tsv',
            'desc-mask_timeseries.npy',
        ]
        output = []
        for suffix in suffix_li:
            if self.renaming is None:
                fn = [
                    fp_version,
                    'confounds',
                    f'sub-{sid}_task-{task}_run-{run}_{suffix}',
                ]
            else:
                fn = [
                    fp_version,
                    'renamed_confounds',
                    f'sub-{sid}_task-{task}_run-{run:02d}_{suffix}',
                ]
                fn = self.renaming['/'.join(fn)].split('/')
            o = self.dl_dset.get(fn, on_missing='raise')
            output.append(o)
        return output

    def load_design(self, sid, task, run, fp_version=None):
        if fp_version is None:
            fp_version = self.fp_version
        suffix = 'design.json'
        if self.renaming is None:
            fn = [
                fp_version,
                'design',
                f'sub-{sid}_task-{task}_run-{run:02d}_{suffix}',
            ]
        else:
            fn = [
                fp_version,
                'renamed_design',
                f'sub-{sid}_task-{task}_run-{run:02d}_{suffix}',
            ]
            fn = self.renaming['/'.join(fn)].split('/')
        output = self.dl_dset.get(fn, on_missing='raise')
        return output

    def load_contrasts(
        self,
        sid,
        task,
        run,
        lr,
        kind='t',
        space=None,
        resample=None,
        prep=None,
        force_volume=False,
        fp_version=None,
    ):
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
        suffix = f'{kind}.npy'
        fn = [
            fp_version,
            'contrasts',
            space,
            f'{lr}-cerebrum',
            resample,
            prep,
            f'sub-{sid}_task-{task}_run-{run:02d}_{suffix}',
        ]
        output = self.dl_dset.get(fn, on_missing='raise')
        return output

    def get_data(
        self,
        sid,
        task,
        run,
        lr,
        space=None,
        resample=None,
        prep=None,
        fp_version=None,
        force_volume=False,
        prep_kwargs=None,
        slicer=None,
    ):
        if isinstance(run, (tuple, list)):
            ret = [
                self.get_data(
                    sid,
                    task,
                    run_,
                    lr,
                    space,
                    resample,
                    prep,
                    fp_version,
                    force_volume,
                    prep_kwargs,
                )
                for run_ in run
            ]
            if isinstance(ret[0], tuple):
                n = len(ret[0])
                ret = tuple(
                    [
                        np.concatenate([ret_[i] for ret_ in ret], axis=0)
                        for i in range(n)
                    ]
                )
            elif isinstance(ret[0], np.ndarray):
                ret = np.concatenate(ret, axis=0)
            return ret

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
        if slicer is None:
            slicer = getattr(self, 'slicer', None)

        dm = self.load_data(sid, task, run, lr, space, resample, fp_version)
        confounds = self.load_confounds(sid, task, run, fp_version)
        if space_kind == 'surface':
            if lr == 'lr':
                cortical_mask = np.concatenate(
                    [get_mask(lr_, space) for lr_ in 'lr'], axis=0
                )
            else:
                cortical_mask = get_mask(lr, space)
        else:
            cortical_mask = None
        if isinstance(prep, str):
            prep = get_prep(prep, **(prep_kwargs if prep_kwargs is not None else {}))
        if slicer is not None:
            dm = slicer(dm, task, run)
            confounds = [slicer(c, task, run) for c in confounds]
        dm = prep(dm, confounds, cortical_mask)
        return dm

    def _get_anatomical_data(self, sid, which, lr, mask, space, fp_version):
        if fp_version is None:
            fp_version = self.fp_version
        if space is None:
            space = self.surface_space
        fn = os.path.join(
            fp_version, 'anatomy', space, 'overlap', which, f'{sid}_{lr}h.npy'
        )
        d = self.dl_dset.get(fn, on_missing='raise')
        d = d.astype(np.float64)
        if mask is not False and mask is not None:
            if isinstance(mask, np.ndarray):
                cortical_mask = mask
            else:
                cortical_mask = get_mask(lr, space)
            d = d[cortical_mask]
        return d

    def morphometry(self, sid, which, lr, mask=True, space=None, fp_version=None):
        return self._get_anatomical_data(
            sid=sid, which=which, lr=lr, mask=mask, space=space, fp_version=fp_version
        )

    def parcellation(self, sid, which, lr, mask=True, space=None, fp_version=None):
        return self._get_anatomical_data(
            sid=sid,
            which=which + '.annot',
            lr=lr,
            mask=mask,
            space=space,
            fp_version=fp_version,
        )


class Bologna(Dataset):
    def __init__(
        self,
        space=['onavg-ico32', 'mni-4mm'],
        resample=['1step_pial_overlap', '1step_linear_overlap'],
        prep='default',
        fp_version='20.2.7',
        name='bologna',
        root_dir=None,
        dl_source=None,
    ):
        super().__init__(
            name,
            dl_source=dl_source,
            root_dir=root_dir,
            space=space,
            resample=resample,
            prep=prep,
            fp_version=fp_version,
        )
        self.subjects = [f'{_+1:02d}' for _ in range(69)]
        self.tasks = ['rest']


class Forrest(Dataset):
    """The Forrest dataset.

    This dataset contains fMRI responses while participants passively
    viewed the audiovisual movie *Forrest Gump* (1994).

    See Sengupta et al. (2016) in the References for more details.

    References
    ----------
    https://doi.org/10.1038/sdata.2016.93
    https://datasets.datalad.org/?dir=/studyforrest
    https://www.studyforrest.org/
    """

    def __init__(
        self,
        space=['onavg-ico32', 'mni-4mm'],
        resample=['1step_pial_overlap', '1step_linear_overlap'],
        prep='default',
        fp_version='20.2.7',
        name='forrest',
        root_dir=None,
        dl_source='https://gin.g-node.org/neuroboros/forrest',
    ):
        super().__init__(
            name,
            dl_source=dl_source,
            root_dir=root_dir,
            space=space,
            resample=resample,
            prep=prep,
            fp_version=fp_version,
        )
        self.subjects = [
            '01',
            '02',
            '03',
            '04',
            '05',
            '06',
            '09',
            '10',
            '14',
            '15',
            '16',
            '17',
            '18',
            '19',
            '20',
        ]
        self.tasks = [
            "forrest",
            "movielocalizer",
            "objectcategories",
            "retmapccw",
            "retmapclw",
            "retmapcon",
            "retmapexp",
        ]


class Dalmatians(Dataset):
    def __init__(
        self,
        space=['onavg-ico32', 'mni-4mm'],
        resample=['1step_pial_overlap', '1step_linear_overlap'],
        prep='default',
        fp_version='20.2.7',
        name='dalmatians',
        root_dir=None,
        dl_source=None,
    ):
        super().__init__(
            name,
            dl_source=dl_source,
            root_dir=root_dir,
            space=space,
            resample=resample,
            prep=prep,
            fp_version=fp_version,
        )
        self.subjects = [
            'AB033',
            'AB034',
            'AB035',
            'AB036',
            'AB037',
            'AB038',
            'AB039',
            'AB041',
            'AB042',
            'AB043',
            'AB053',
            'AO003',
            'AO004',
            'AO005',
            'AO006',
            'AO007',
            'AO008',
            'AO009',
            'AO010',
            'AO011',
            'AO027',
            'AV012',
            'AV013',
            'AV014',
            'AV015',
            'AV016',
            'AV017',
            'AV018',
            'AV019',
            'AV022',
            'AV032',
            'VD044',
            'VD045',
            'VD046',
            'VD047',
            'VD048',
            'VD049',
            'VD050',
            'VD051',
            'VD052',
            'VO020',
            'VO021',
            'VO023',
            'VO024',
            'VO025',
            'VO026',
            'VO028',
            'VO029',
            'VO030',
            'VO031',
        ]
        self.tasks = ['dalmatians', 'scrambled']


class SpaceTop(Dataset):
    def __init__(
        self,
        space=['onavg-ico32', 'mni-4mm'],
        resample=['1step_pial_overlap', '1step_linear_overlap'],
        prep='default',
        fp_version='20.2.7',
        name='spacetop',
        root_dir=None,
        dl_source=None,
    ):
        super().__init__(
            name,
            dl_source=dl_source,
            root_dir=root_dir,
            space=space,
            resample=resample,
            prep=prep,
            fp_version=fp_version,
        )
        self.tasks = ['faces']


class CamCAN(Dataset):
    def __init__(
        self,
        space=['onavg-ico32', 'mni-4mm'],
        resample=['1step_pial_overlap', '1step_linear_overlap'],
        prep='default',
        fp_version='20.2.7',
        name='camcan',
        root_dir=None,
        dl_source=None,
    ):
        super().__init__(
            name,
            dl_source=dl_source,
            root_dir=root_dir,
            space=space,
            resample=resample,
            prep=prep,
            fp_version=fp_version,
        )
        self.tasks = ['bang', 'rest', 'smt']

        self.subject_sets = {}
        mod_dir = os.path.dirname(os.path.realpath(__file__))
        for task in ['bang', 'rest', 'smt']:
            with open(os.path.join(mod_dir, f'camcan_{task}.txt')) as f:
                self.subject_sets[task] = f.read().splitlines()


class ID1000(Dataset):
    def __init__(
        self,
        space=['onavg-ico32', 'mni-4mm'],
        resample=['1step_pial_overlap', '1step_linear_overlap'],
        prep='default',
        fp_version='20.2.7',
        name='id1000',
        root_dir=None,
        dl_source=None,
    ):
        super().__init__(
            name,
            dl_source=dl_source,
            root_dir=root_dir,
            space=space,
            resample=resample,
            prep=prep,
            fp_version=fp_version,
        )
        self.tasks = ['moviewatching']

        mod_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(mod_dir, f'id1000.txt')) as f:
            self.subjects = f.read().splitlines()


class Raiders(Dataset):
    """The Raiders dataset collected with the Siemens scanner.

    This dataset contains fMRI responses while participants passively
    viewed the audiovisual movie *Raiders of the Lost Ark* (1981).

    See Nastase (2018) in the References for more details.

    References
    ----------
    https://www.proquest.com/docview/2018905893/abstract/F07E13FBBC234A93PQ/1
    """

    def __init__(
        self,
        space=['onavg-ico32', 'mni-4mm'],
        resample=['1step_pial_overlap', '1step_linear_overlap'],
        prep='default',
        fp_version='20.2.7',
        name='raiders',
        root_dir=None,
        dl_source=None,
    ):
        super().__init__(
            name,
            dl_source=dl_source,
            root_dir=root_dir,
            space=space,
            resample=resample,
            prep=prep,
            fp_version=fp_version,
        )
        self.tasks = ['raiders', 'actions']
        self.subjects = [
            'sid000005',
            'sid000007',
            'sid000009',
            'sid000010',
            'sid000012',
            'sid000013',
            'sid000020',
            'sid000021',
            'sid000024',
            'sid000029',
            'sid000034',
            'sid000052',
            'sid000102',
            'sid000114',
            'sid000120',
            'sid000134',
            'sid000142',
            'sid000278',
            'sid000416',
            'sid000433',
            'sid000499',
            'sid000522',
            'sid000535',
        ]

    def slicer(self, data, task, run):
        if task == 'raiders':  # stimulus overlap between runs
            if run == 1:
                data = data[:-10]
            elif run == 4:
                data = data[10:]
            elif run in [2, 3]:
                data = data[10:-10]
            else:
                raise ValueError(f"Run {run} not recognized.")
        return data


class PhilipsRaiders(Dataset):
    """The Raiders dataset collected using the Philips scanner.

    This dataset contains fMRI responses while participants passively
    viewed the audiovisual movie *Raiders of the Lost Ark* (1981).

    See Haxby et al. (2011) and Guntupalli et al. (2016) in the References
    for more details.

    References
    ----------
    https://doi.org/10.1016/j.neuron.2011.08.026
    https://doi.org/10.1093/cercor/bhw068
    https://datasets.datalad.org/?dir=/labs/haxby/raiders
    """

    def __init__(
        self,
        space=['onavg-ico32', 'mni-4mm'],
        resample=['1step_pial_overlap', '1step_linear_overlap'],
        prep='default',
        fp_version='20.2.7',
        name='raiders',
        root_dir=None,
        dl_source=None,
    ):
        super().__init__(
            name,
            dl_source=dl_source,
            root_dir=root_dir,
            space=space,
            resample=resample,
            prep=prep,
            fp_version=fp_version,
        )


class Budapest(Dataset):
    """The Budapest dataset.

    This dataset contains fMRI responses while participants passively
    viewed the audiovisual movie *The Grand Budapest Hotel* (2014).

    See Visconti di Oleggio Castello et al. (2020) in the References
    for more details.

    References
    ----------
    https://doi.org/10.1038/s41597-020-00735-4
    https://datasets.datalad.org/?dir=/labs/gobbini/budapest
    https://openneuro.org/datasets/ds003017
    https://doi.org/10.18112/openneuro.ds003017.v1.0.3
    """

    def __init__(
        self,
        space=['onavg-ico32', 'mni-4mm'],
        resample=['1step_pial_overlap', '1step_linear_overlap'],
        prep='default',
        fp_version='20.2.7',
        name='budapest',
        root_dir=None,
        dl_source=None,
    ):
        super().__init__(
            name,
            dl_source=dl_source,
            root_dir=root_dir,
            space=space,
            resample=resample,
            prep=prep,
            fp_version=fp_version,
        )
        self.tasks = ['budapest', 'hyperface', 'localizer']
        self.subjects = [
            'sid000005',
            'sid000007',
            'sid000009',
            'sid000010',
            'sid000013',
            'sid000020',
            'sid000021',
            'sid000024',
            'sid000029',
            'sid000034',
            'sid000052',
            'sid000114',
            'sid000120',
            'sid000134',
            'sid000142',
            'sid000278',
            'sid000416',
            'sid000499',
            'sid000522',
            'sid000535',
            'sid000560',
        ]


class MonkeyKingdom(Dataset):
    def __init__(
        self,
        space=['onavg-ico32', 'mni-4mm'],
        resample=['1step_pial_overlap', '1step_linear_overlap'],
        prep='default',
        fp_version='20.2.7',
        name='monkey-kingdom',
        root_dir=None,
        dl_source=None,
    ):
        super().__init__(
            name,
            dl_source=dl_source,
            root_dir=root_dir,
            space=space,
            resample=resample,
            prep=prep,
            fp_version=fp_version,
        )
        self.tasks = ['monkey', 'rest', 'localizer', 'language']
        self.subjects = [
            'sid001123',
            'sid001293',
            'sid001294',
            'sid001678',
            'sid001784',
            'sid001830',
            'sid001835',
            'sid001986',
            'sid002015',
            'sid002161',
            'sid002180',
            'sid002317',
            'sid002325',
            'sid002406',
            'sid002414',
            'sid002435',
            'sid002446',
            'sid002449',
            'sid002454',
            'sid002471',
            'sid002499',
            'sid002509',
            'sid002519',
            'sid002570',
        ]


class Life(Dataset):
    """The Life dataset.

    This dataset contains fMRI responses while participants passively
    viewed four audiovisual segments of the *Life* (2009) nature documentary
    ("life"), as well as fMRI responses while performing an attention task
    while viewing video clips of various animals performing different
    actions ("attention").

    During the "attention" session, participants were asked to attend to
    either the behavior ("beh") or taxonomy ("tax") of the animal in the
    video. Participants performed a 1-back repetition detection task for
    either the same behavior or same taxonomy category in two consecutive
    videos. Participants performed 5 runs for each task. In each run, there
    were 20 conditions (5 taxonomic categories x 4 behavioral categories).
    The 5 taxonomic categories were: primates, ungulates, birds, reptiles,
    and insects. The 4 behavioral categories were: eating, fighting,
    running, and swimming.

    The "life" data was collected with TR = 2.5 s and 3 mm isotropic
    voxels. The "attention" data was collected with TR = 2 s and 3 mm
    isotropic voxels.

    See Nastase et al. (2017, 2018) in the References for more details.

    References
    ----------
    https://doi.org/10.1093/cercor/bhx138
    https://doi.org/10.3389/fnins.2018.00316
    https://datasets.datalad.org/?dir=/labs/haxby/life
    https://datasets.datalad.org/?dir=/labs/haxby/attention
    https://openneuro.org/datasets/ds000233
    https://doi.org/10.18112/openneuro.ds000233.v1.0.1
    https://snastase.github.io/data/
    """

    def __init__(
        self,
        space=['onavg-ico32', 'mni-4mm'],
        resample=['1step_pial_overlap', '1step_linear_overlap'],
        prep='default',
        fp_version='20.2.7',
        name='life',
        root_dir=None,
        dl_source='https://gin.g-node.org/neuroboros/life',
    ):
        super().__init__(
            name,
            dl_source=dl_source,
            root_dir=root_dir,
            space=space,
            resample=resample,
            prep=prep,
            fp_version=fp_version,
        )
        self.tasks = ['life', 'tax', 'beh']
        self.subject_sets = {
            'attention': [
                'rid000001',
                'rid000012',
                'rid000017',
                'rid000024',
                'rid000027',
                'rid000031',
                'rid000032',
                'rid000033',
                'rid000034',
                'rid000036',
                'rid000037',
                'rid000041',
            ],
            'all': [
                'rid000001',
                'rid000005',
                'rid000006',
                'rid000009',
                'rid000012',
                'rid000014',
                'rid000017',
                'rid000019',
                'rid000020',
                'rid000024',
                'rid000027',
                'rid000031',
                'rid000032',
                'rid000033',
                'rid000034',
                'rid000036',
                'rid000037',
                'rid000038',
                'rid000041',
            ],
        }
        self.subjects = self.subject_sets['all']
        self.contrasts = [
            'primate_eating',
            'primate_fighting',
            'primate_running',
            'primate_swimming',
            'ungulate_eating',
            'ungulate_fighting',
            'ungulate_running',
            'ungulate_swimming',
            'bird_eating',
            'bird_fighting',
            'bird_running',
            'bird_swimming',
            'reptile_eating',
            'reptile_fighting',
            'reptile_running',
            'reptile_swimming',
            'insect_eating',
            'insect_fighting',
            'insect_running',
            'insect_swimming',
            'primate',
            'ungulate',
            'bird',
            'reptile',
            'insect',
            'eating',
            'fighting',
            'running',
            'swimming',
        ]

    def slicer(self, data, task, run):
        """
        The duration of the 4 video stimuli for the "life" data are: 15:21.32,
        14:12.32, 15:28.33, 16:55.31. That is, 921.32, 852.32, 928.33, and
        1015.31 seconds, respectively. For runs 1, 2, and 4, the first 8 s of
        the video were skipped. For run 3, the first 8.6 seconds of the video
        were skipped.
        After accounting for the skipped video frames, the length of the
        videos (in TRs) are 365.328, 337.728, 367.892, 402.924, respectively.

        The 4 runs of the "life" data contain 374, 346, 377, and 412 volumes,
        respectively. That is, 935, 865, 942, and 1030 seconds, respectively.

        We remove the last few volumes, which hardly contain any responses
        even after accounting for the hemodynamic delay. One participant only
        has 366 and 338 volumes for the first two runs, respectively.
        We keep the same number of volumes for all participants for the first
        two runs, so that we can use the participant's data.
        """
        if task == 'life':
            end = {1: 366, 2: 338, 3: 370, 4: 405}[run]
            data = data[:end]
        return data


class HBNSSI(Dataset):
    def __init__(
        self,
        space=['onavg-ico32', 'mni-4mm'],
        resample=['1step_pial_overlap', '1step_linear_overlap'],
        prep='default',
        fp_version='23.2.0',
        name='hbn-ssi',
        root_dir=None,
        dl_source=None,
    ):
        super().__init__(
            name,
            dl_source=dl_source,
            root_dir=root_dir,
            space=space,
            resample=resample,
            prep=prep,
            fp_version=fp_version,
        )
        self.tasks = ['raiders', 'matrix', 'walle', 'afewgoodmen']
        self.subjects = [
            '0031121',
            '0031122',
            '0031123',
            '0031126',
            '0031127',
            '0031130',
            '0031131',
            '0031132',
            '0031133',
            '0031124',
            '0031125',
            '0031128',
            '0031129',
        ]


class Whiplash(Dataset):
    def __init__(
        self,
        space=['onavg-ico32', 'mni-4mm'],
        resample=['1step_pial_overlap', '1step_linear_overlap'],
        prep='default',
        fp_version='23.2.0',
        name='whiplash',
        root_dir=None,
        dl_source=None,
    ):
        super().__init__(
            name,
            dl_source=dl_source,
            root_dir=root_dir,
            space=space,
            resample=resample,
            prep=prep,
            fp_version=fp_version,
        )
        self.tasks = ['whiplash']


datasets = {
    'forrest': Forrest,
    'bologna': Bologna,
    'dalmatians': Dalmatians,
    'spacetop': SpaceTop,
    'camcan': CamCAN,
    'id1000': ID1000,
    'raiders': Raiders,
    'budapest': Budapest,
    'monkeykingdom': MonkeyKingdom,
    'life': Life,
    'hbn-ssi': HBNSSI,
    'whiplash': Whiplash,
}


def get_dataset(name):
    if name not in datasets:
        raise ValueError(
            f"Dataset {name} not recognized. Valid datasets are: {datasets.keys()}"
        )
    return datasets[name]()
