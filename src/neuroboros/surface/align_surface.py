import os
from glob import glob

from hyperalignment.procrustes import procrustes

from ..spaces import get_geometry
from .__init__ import Sphere, Surface


class Aligner:
    """
    Aligns a surface to a template using Procrustes analysis.
    The surface is resampled to the template space, and the coordinates are
    centered and rotated to minimize the distance between the surface and the
    template.
    """

    def __init__(self, sphere, anat, lr, kind):
        """
        Parameters
        ----------
        sphere : Sphere
            The sphere to be aligned.
        anat : Surface
            The anatomical surface to be aligned.
        lr : str
            The hemisphere to be aligned ('l' or 'r').
        kind : str
            The kind of surface to be aligned ('pial', 'white', etc.).
        """
        self.sphere = sphere
        self.anat = anat
        self.lr = lr
        self.kind = kind

    @classmethod
    def from_fmriprep(cls, folder, lr, kind="pial"):
        aid = os.path.basename(os.path.dirname(folder)).split("_")[0].split("-")[1]
        sid = os.path.basename(folder).split("-")[1]
        anat_dirs = sorted(glob(os.path.join(folder, "ses-*", "anat")))
        if len(anat_dirs) != 1:
            print(folder)
            print(anat_dirs)
            raise RuntimeError
        anat_dir = anat_dirs[0]
        ses = os.path.basename(os.path.dirname(anat_dir)).split("-")[1]
        print(sid, ses, aid)
        sphere_fns = glob(
            os.path.join(
                anat_dir,
                f"sub-{sid}_ses-{ses}*_hemi-{lr.upper()}_"
                "space-fsaverage_desc-reg_sphere.surf.gii",
            )
        )
        anat_fns = glob(
            os.path.join(
                anat_dir, f"sub-{sid}_ses-{ses}*_hemi-{lr.upper()}_{kind}.surf.gii"
            )
        )
        if len(sphere_fns) != 1:
            print(sphere_fns)
            raise RuntimeError
        if len(anat_fns) != 1:
            print(anat_fns)
            raise RuntimeError
        sphere = Sphere.from_file(sphere_fns[0])
        anat = Surface.from_file(anat_fns[0])
        return cls(sphere, anat, lr, kind)

    @classmethod
    def from_fs(cls, folder, lr, kind="pial"):
        if not folder.endswith("surf"):
            folder = os.path.join(folder, "surf")
        sphere = Sphere.from_file(os.path.join(folder, f"{lr}h.sphere.reg"))
        anat = Surface.from_file(os.path.join(folder, f"{lr}h.{kind}"))
        return cls(sphere, anat, lr, kind)

    def run(self):
        """
        Returns
        -------
        aligned : np.ndarray
            The coordinates of the aligned surface.
        """
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
