import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree

from .barycentric import barycentric, barycentric_vectors, barycentric_weights
from .nnfr import nnfr
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


def compute_neighbor_distances(coords, neighbors):
    nv = coords.shape[0]
    dists = []
    for i in range(nv):
        d = np.linalg.norm(coords[neighbors[i]] - coords[[i]], axis=1)
        dists.append(d)
    return dists


def compute_vertex_normals_sine_weight(coords, faces):
    normals = np.zeros(coords.shape)

    f_coords = coords[faces]
    edges = np.roll(f_coords, 1, axis=1) - f_coords
    del f_coords
    edges /= np.linalg.norm(edges, axis=2, keepdims=True)

    for f, ee in zip(faces, edges):
        normals[f[0]] += np.cross(ee[0], ee[1])
        normals[f[1]] += np.cross(ee[1], ee[2])
        normals[f[2]] += np.cross(ee[2], ee[0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    return normals


def compute_vertex_normals_equal_weight(coords, faces):
    normals = np.zeros(coords.shape)

    f_coords = coords[faces]
    e01 = f_coords[:, 1, :] - f_coords[:, 0, :]
    e12 = f_coords[:, 2, :] - f_coords[:, 1, :]
    del f_coords

    face_normals = np.cross(e01, e12)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)
    for f, n in zip(faces, face_normals):
        normals[f] += n
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    return normals
