import heapq
import numpy as np


def dijkstra_distances(src, nv, neighbors, neighbor_distances, max_dist=None):
    """Compute the Dijkstra distances from one vertex to all others.

    Parameters
    ----------
    src : int
        The index of the source vertex. The computed distances are distances
        to this vertex.
    nv : int
        The total number of vertices.
    neighbors : list of ndarray
        Each element in the list is an ndarray of integers, which are the
        indices of the neighbors of a vertex.
    neighbor_distances : list of ndarray
        Each element in the list is an ndarray of floats, which are the
        distances between the vertex and its neighbors. The order is the same
        as ``neighbors``.
    max_dist : {float, None}, default=None
        The maximal distance. The algorithm will stop when ``max_dist`` is
        reached, and the distances more than ``max_dist`` will be filled with
        ``np.inf``. If it's None, there will be no limit, and the algorithm
        will only stop when all Dijkstra distances have been computed.

    Returns
    -------
    dists : ndarray of shape (nv, )
        The Dijkstra distances between the source vertex and all vertices.
    """
    dists = np.full((nv, ), np.inf)
    finished = np.zeros((nv, ), dtype=bool)
    candidates = [(0.0, src)]
    dists[src] = 0.0

    while candidates:
        d, idx = heapq.heappop(candidates)
        if finished[idx]:
            continue

        for nbr, nbr_d in zip(neighbors[idx], neighbor_distances[idx]):
            new_d = d + nbr_d
            if max_dist is not None and new_d > max_dist:
                continue
            if new_d < dists[nbr]:
                dists[nbr] = new_d
                heapq.heappush(candidates, (new_d, nbr))
        finished[idx] = True

    return dists


def dijkstra(src, nv, neighbors, neighbor_distances, max_dist, sort=True):
    """Compute the Dijkstra neighbors and distances to these neighbors.

    Parameters
    ----------
    src : int
        The index of the source vertex. The computed distances are distances
        to this vertex.
    nv : int
        The total number of vertices.
    neighbors : list of ndarray
        Each element in the list is an ndarray of integers, which are the
        indices of the neighbors of a vertex.
    neighbor_distances : list of ndarray
        Each element in the list is an ndarray of floats, which are the
        distances between the vertex and the neighbors. The order is the same
        as ``neighbors``.
    max_dist : float
        The maximal distance. The algorithm will stop when ``max_dist`` is
        reached.
    sort : bool, default=True
        Whether to sort the Dijkstra neighbors by the distance (nearest
        first).

    Returns
    -------
    dijkstra_nbrs : ndarray
        The indices of vertices that are within ``max_dist`` of the source
        vertex based on Dijkstra distance.
    dijkstra_dists : ndarray
        The Dijkstra distances.
    """
    dijkstra_dists = dijkstra_distances(
        src, nv, neighbors, neighbor_distances, max_dist)
    mask = dijkstra_dists < max_dist
    dijkstra_nbrs = np.where(mask)[0]
    dijkstra_dists = dijkstra_dists[mask]
    if sort:
        sort_idx = np.argsort(dijkstra_dists)
        dijkstra_nbrs = dijkstra_nbrs[sort_idx]
        dijkstra_dists = dijkstra_dists[sort_idx]
    return dijkstra_nbrs, dijkstra_dists
