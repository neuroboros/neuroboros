import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree

from .barycentric import barycentric, barycentric_vectors, barycentric_weights


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


def compute_neighbors(faces, nv=None):
    if nv is None:
        nv = faces.max() + 1

    neighbors = [[] for _ in range(nv)]
    for vs in faces:
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                if vs[j] in neighbors[vs[i]]:
                    continue
                neighbors[vs[i]].append(vs[j])
    for i in range(nv):
        neighbors[i] = np.array(neighbors[i])

    return neighbors
