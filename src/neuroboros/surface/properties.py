import numpy as np


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


def compute_face_areas(coords, faces):
    e1 = coords[faces[:, 1]] - coords[faces[:, 0]]
    e2 = coords[faces[:, 2]] - coords[faces[:, 0]]
    face_areas = np.linalg.norm(np.cross(e1, e2), axis=1) / 2
    return face_areas


def compute_vertex_areas(coords, faces, face_areas=None):
    if face_areas is None:
        face_areas_per_vertex = compute_face_areas(coords, faces) / 3
    else:
        face_areas_per_vertex = face_areas / 3
    vertex_areas = np.zeros((coords.shape[0], ))
    for f, a in zip(faces, face_areas_per_vertex):
        vertex_areas[f] += a
    return vertex_areas
