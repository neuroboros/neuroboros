import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree

from .properties import compute_neighbors, compute_neighbor_distances, \
    compute_vertex_normals_equal_weight, compute_vertex_normals_sine_weight
from .barycentric import barycentric, barycentric_vectors, barycentric_weights
from .nnfr import nnfr
from .areal import areal
from .union import compute_union_sphere


class Surface(object):
    def __init__(self, coords, faces):
        self.coords = coords.astype('float')
        self.faces = faces
        self.nv = self.coords.shape[0]
        self.nf = self.faces.shape[0]

    @property
    def neighbors(self):
        if not hasattr(self, '_neighbors'):
            self._neighbors = compute_neighbors(self.faces, self.nv)
        return self._neighbors

    @property
    def neighbor_distances(self):
        if not hasattr(self, '_neighbor_distances'):
            self._neighbor_distances = compute_neighbor_distances(
                self.coords, self.neighbors)
        return self._neighbor_distances

    @property
    def tree(self):
        if not hasattr(self, '_tree'):
            self._tree = cKDTree(self.coords)
        return self._tree

    @property
    def v2f(self):
        if not hasattr(self, '_v2f'):
            self._v2f = [[] for _ in range(self.nv)]
            for i, f in enumerate(self.faces):
                for v in f:
                    self._v2f[v].append(i)
        return self._v2f

    @classmethod
    def from_gifti(cls, fn):
        gii = nib.load(fn)
        coords, faces = [_.data for _ in gii.darrays]
        instance = cls(coords, faces)
        return instance


class Sphere(Surface):
    def __init__(self, coords, faces):
        super().__init__(coords, faces)
        self.sphericalize()

    def sphericalize(self):
        norm = np.sqrt(np.sum(self.coords**2, axis=1))[:, np.newaxis]
        if not np.allclose(norm, 1):
            self.coords /= norm

    @classmethod
    def from_gifti(cls, fn):
        gii = nib.load(fn)
        coords, faces = [_.data for _ in gii.darrays]
        instance = cls(coords, faces)
        return instance

    @property
    def normals(self):
        if not hasattr(self, '_normals'):
            self._normals = compute_vertex_normals_equal_weight(
                self.coords, self.faces)
        return self._normals

    @property
    def normals_sine(self):
        if not hasattr(self, '_normals_sine'):
            self._normals_sine = compute_vertex_normals_sine_weight(
                self.coords, self.faces)
        return self._normals_sine

    def prepare_barycentric(self):
        if not hasattr(self, 'vecs'):
            f_coords = self.coords[self.faces, :]
            a = f_coords[:, 0, :]
            e1 = f_coords[:, 1, :] - f_coords[:, 0, :]
            e2 = f_coords[:, 2, :] - f_coords[:, 0, :]
            self.vecs = np.stack([
                a, e1, e2,
                np.cross(e1, e2),
                np.cross(e2, a),
                np.cross(a, e1)],
                axis=1)

    def barycentric(self, coords, **kwargs):
        self.prepare_barycentric()
        return barycentric(self.vecs, coords, self.v2f, self.tree, self.faces, self.nv, **kwargs)

    def nnfr(self, coords, reverse=True):
        return nnfr(self.coords, coords, reverse=reverse)

    def areal(self, coords, anat_coords):
        return areal(self, coords, anat_coords)

    def union(self, to_unite, eps=1e-10):
        if isinstance(to_unite, Sphere):
            coords = to_unite.coords
        elif isinstance(to_unite, np.ndarray):
            coords = to_unite
        else:
            raise TypeError("`to_unite` must be a Surface or ndarray.")
        new_coords, new_faces, indices1, indices2 = compute_union_sphere(self, coords, eps=eps)
        union_sphere = Sphere(new_coords, new_faces)
        return union_sphere, indices1, indices2
