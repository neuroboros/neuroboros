import numpy as np
from scipy.spatial import ConvexHull, cKDTree


def compute_union_coords(coords1, coords2, eps=1e-10):
    nv1 = coords1.shape[0]
    nv2 = coords2.shape[0]
    mask = np.ones((nv2, ), dtype=bool)
    indices1 = np.arange(nv1)
    indices2 = np.zeros((nv2, ), dtype=int)
    tree = cKDTree(coords1)
    count = nv1
    for i, c in enumerate(coords2):
        d, idx = tree.query(c)
        if d < eps:
            mask[i] = False
            indices2[i] = idx
        else:
            indices2[i] = count
            count += 1

    if not np.any(mask):
        coords = coords1
    else:
        coords = np.concatenate([coords1, coords2[mask]])

    return coords, indices1, indices2


def compute_union_sphere(sphere, coords, eps=1e-10):
    union_coords, indices1, indices2 = compute_union_coords(
        sphere.coords, coords, eps=eps)
    if union_coords.shape == sphere.coords.shape:
        return sphere.coords, sphere.faces, indices1, indices2

    nv1 = sphere.coords.shape[0]
    new_coords = union_coords[nv1:]
    t_indices = sphere.barycentric(new_coords, return_sparse=False)[0]

    new_faces = []
    for t_idx in np.unique(t_indices):
        indices = np.where(t_indices == t_idx)[0]
        f = sphere.faces[t_idx]
        cc = np.concatenate([sphere.coords[f], new_coords[indices]], axis=0)
        mp = np.concatenate([f, nv1 + indices])
        hull = ConvexHull(cc)
        for nf in hull.simplices:
            if not np.all(np.isin(nf, np.arange(3))):
                new_faces.append(mp[nf])
    new_faces = np.stack(new_faces)

    mask = np.logical_not(np.isin(np.arange(sphere.faces.shape[0]), t_indices))
    union_faces = np.concatenate([sphere.faces[mask], new_faces], axis=0)

    return union_coords, union_faces, indices1, indices2
