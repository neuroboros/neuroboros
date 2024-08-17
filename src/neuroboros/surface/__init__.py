import os

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

from .areal import areal
from .barycentric import barycentric, barycentric_vectors, barycentric_weights
from .nnfr import nnfr
from .properties import (
    compute_face_areas,
    compute_neighbor_distances,
    compute_neighbors,
    compute_vertex_areas,
    compute_vertex_normals_equal_weight,
    compute_vertex_normals_sine_weight,
)
from .subdivision import surface_subdivision
from .union import compute_union_sphere
from .voronoi import (
    compute_occupation,
    native_voronoi,
    overlap_transform,
    split_triangle,
    subdivide_edges,
)


class Surface:
    def __init__(self, coords, faces):
        self.coords = coords.astype("float")
        self.faces = faces
        self.nv = self.coords.shape[0]
        self.nf = self.faces.shape[0]

    @property
    def neighbors(self):
        if not hasattr(self, "_neighbors"):
            self._neighbors = compute_neighbors(self.faces, self.nv)
        return self._neighbors

    @property
    def neighbor_distances(self):
        if not hasattr(self, "_neighbor_distances"):
            self._neighbor_distances = compute_neighbor_distances(
                self.coords, self.neighbors
            )
        return self._neighbor_distances

    @property
    def tree(self):
        if not hasattr(self, "_tree"):
            self._tree = cKDTree(self.coords)
        return self._tree

    @property
    def vertex_areas(self):
        if not hasattr(self, "_vertex_areas"):
            self._vertex_areas = compute_vertex_areas(
                self.coords, self.faces, self.face_areas
            )
        return self._vertex_areas

    def vertex_areas_nn(self, n_div=8, t_div=32):
        new_coords, e_mapping, neighbors = subdivide_edges(
            self.coords, self.faces, n_div
        )
        coords = np.concatenate([self.coords, new_coords])
        nn, nnd = native_voronoi(coords, self.faces, e_mapping, neighbors)

        areas = np.zeros((self.nv,))
        face_areas = self.face_areas
        ww = split_triangle(t_div)
        for f_idx, f in enumerate(self.faces):
            a, b, c = sorted(f)
            indices = np.concatenate(
                [e_mapping[(a, b)], e_mapping[(a, c)], e_mapping[(b, c)], [a, b, c]]
            )
            uu = compute_occupation(f_idx, f, coords, indices, nn, nnd, {}, ww)
            for u, m1 in uu.items():
                areas[u] += m1.mean() * face_areas[f_idx]
        return areas

    @property
    def face_areas(self):
        if not hasattr(self, "_face_areas"):
            self._face_areas = compute_face_areas(self.coords, self.faces)
        return self._face_areas

    @property
    def v2f(self):
        if not hasattr(self, "_v2f"):
            self._v2f = [[] for _ in range(self.nv)]
            for i, f in enumerate(self.faces):
                for v in f:
                    self._v2f[v].append(i)
        return self._v2f

    def subdivide(self, n_div):
        coords, faces = surface_subdivision(self.coords, self.faces, n_div)
        if isinstance(self, Sphere):
            norm = np.linalg.norm(coords[self.coords.shape[0] :], axis=1, keepdims=True)
            coords[self.coords.shape[0] :] /= norm
            subdivided = Sphere(coords, faces)
        else:
            subdivided = Surface(coords, faces)
        return subdivided

    @classmethod
    def from_gifti(cls, fn):
        gii = nib.load(fn)
        coords, faces = (_.data for _ in gii.darrays)
        instance = cls(coords, faces)
        return instance

    @classmethod
    def from_fs(cls, fn):
        coords, faces = nib.freesurfer.read_geometry(fn)
        instance = cls(coords, faces)
        return instance

    @classmethod
    def from_file(cls, fn):
        try:
            instance = cls.from_gifti(fn)
            return instance
        except Exception:
            instance = cls.from_fs(fn)
            return instance

    def to_gifti(self, fn):
        dirname = os.path.dirname(fn)
        if dirname != "":
            os.makedirs(dirname, exist_ok=True)

        if isinstance(self, Sphere):
            coords = self.coords * 100
        else:
            coords = self.coords

        darrays = [
            nib.gifti.GiftiDataArray(
                coords.astype(np.float32),
                intent=nib.nifti1.intent_codes["NIFTI_INTENT_POINTSET"],
                datatype=nib.nifti1.data_type_codes["NIFTI_TYPE_FLOAT32"],
            ),
            nib.gifti.GiftiDataArray(
                self.faces.astype(np.int32),
                intent=nib.nifti1.intent_codes["NIFTI_INTENT_TRIANGLE"],
                datatype=nib.nifti1.data_type_codes["NIFTI_TYPE_INT32"],
            ),
        ]
        gii = nib.gifti.GiftiImage(darrays=darrays)
        nib.save(gii, fn)

    def to_fs(self, fn):
        dirname = os.path.dirname(fn)
        if dirname != "":
            os.makedirs(dirname, exist_ok=True)

        if isinstance(self, Sphere):
            coords = self.coords * 100
        else:
            coords = self.coords

        nib.freesurfer.write_geometry(fn, coords, self.faces)

    def __eq__(self, other):
        if not np.array_equal(self.coords, other.coords):
            return False
        if not np.array_equal(self.faces, other.faces):
            return False
        return True


class Sphere(Surface):
    def __init__(self, coords, faces):
        super().__init__(coords, faces)
        self.sphericalize()

    def sphericalize(self):
        norm = np.sqrt(np.sum(self.coords**2, axis=1))[:, np.newaxis]
        if not np.allclose(norm, 1):
            self.coords /= norm

    @property
    def normals(self):
        if not hasattr(self, "_normals"):
            self._normals = compute_vertex_normals_equal_weight(self.coords, self.faces)
        return self._normals

    @property
    def normals_sine(self):
        if not hasattr(self, "_normals_sine"):
            self._normals_sine = compute_vertex_normals_sine_weight(
                self.coords, self.faces
            )
        return self._normals_sine

    def prepare_barycentric(self):
        if not hasattr(self, "vecs"):
            f_coords = self.coords[self.faces, :]
            a = f_coords[:, 0, :]
            e1 = f_coords[:, 1, :] - f_coords[:, 0, :]
            e2 = f_coords[:, 2, :] - f_coords[:, 0, :]
            self.vecs = np.stack(
                [a, e1, e2, np.cross(e1, e2), np.cross(e2, a), np.cross(a, e1)], axis=1
            )

    def barycentric(self, coords, **kwargs):
        self.prepare_barycentric()
        return barycentric(
            self.vecs, coords, self.v2f, self.tree, self.faces, self.nv, **kwargs
        )

    def nnfr(self, coords, reverse=True):
        return nnfr(self.coords, coords, reverse=reverse)

    def areal(self, coords1, anat_coords, coords2=None):
        return areal(self, coords1, anat_coords, coords2)

    def areal_highres(self, coords1, anat_coords, coords2=None, n_div=4):
        highres_sphere = self.subdivide(n_div)
        highres_anat = surface_subdivision(anat_coords, self.faces, n_div)[0]
        if coords2 is None:
            mat = highres_sphere.areal(self.coords, highres_anat, coords1)
        else:
            mat = highres_sphere.areal(coords1, highres_anat, coords2)
        return mat

    # def dijkstra_subdivision(self, coords1, anat_coords, n_div=4):
    #     f_indices, weights = barycentric(
    #         self.vecs,
    #         coords1,
    #         self.v2f,
    #         self.tree,
    #         self.faces,
    #         self.nv,
    #         eps=1e-7,
    #         return_sparse=False,
    #     )

    def union(self, to_unite, eps=1e-10):
        if isinstance(to_unite, Sphere):
            coords = to_unite.coords
        elif isinstance(to_unite, np.ndarray):
            coords = to_unite
        else:
            raise TypeError("`to_unite` must be a Surface or ndarray.")
        new_coords, new_faces, indices1, indices2 = compute_union_sphere(
            self, coords, eps=eps
        )
        union_sphere = Sphere(new_coords, new_faces)
        return union_sphere, indices1, indices2
