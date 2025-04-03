import os
from glob import glob

from .__init__ import Surface, Sphere
from ..spaces import get_geometry
from hyperalignment.procrustes import procrustes


class Aligner:
    def __init__(self, sphere, anat, lr, kind):
        self.sphere = sphere
        self.anat = anat
        self.lr = lr
        self.kind = kind

    @classmethod
    def from_fmriprep(cls, folder, lr, kind="pial"):
        aid = os.path.basename(os.path.dirname(folder)).split("_")[0].split("-")[1]
        sid = os.path.basename(folder).split("-")[1]
        anat_dirs = sorted(glob(f"{folder}/ses-*/anat"))
        if len(anat_dirs) != 1:
            print(folder)
            print(anat_dirs)
            raise RuntimeError
        anat_dir = anat_dirs[0]
        ses = os.path.basename(os.path.dirname(anat_dir)).split("-")[1]
        print(sid, ses, aid)
        sphere = Sphere.from_file(
            f"{anat_dir}/sub-{sid}_ses-{ses}_acq-MPRAGE_hemi-{lr.upper()}_space-fsaverage_desc-reg_sphere.surf.gii"
        )
        anat = Surface.from_file(
            f"{anat_dir}/sub-{sid}_ses-{ses}_acq-MPRAGE_hemi-{lr.upper()}_{kind}.surf.gii"
        )
        return cls(sphere, anat, lr, kind)

    @classmethod
    def from_fs(cls, folder, lr, kind="pial"):
        if not folder.endswith("/surf"):
            folder = f"{folder}/surf"
        sphere = Sphere.from_file(
            f"{folder}/{lr}h.sphere.reg"
        )
        anat = Surface.from_file(
            f"{folder}/{lr}h.{kind}"
        )
        return cls(sphere, anat, lr, kind)

    def run(self):
        sphere = self.sphere
        anat = self.anat
        coords = get_geometry("sphere", self.lr, vertices_only=True)
        xfm = sphere.barycentric(coords)
        resampled = xfm.T @ anat.coords
        tpl = get_geometry(self.kind, self.lr, vertices_only=True)
        c1 = resampled.mean(axis=0, keepdims=True)
        c2 = tpl.mean(axis=0, keepdims=True)
        resampled = resampled - c1
        tpl = tpl - c2
        rot = procrustes(resampled, tpl, reflection=False, scaling=False)
        aligned = resampled @ rot
        return aligned