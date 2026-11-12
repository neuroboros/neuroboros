import os

import numpy as np
import scipy.sparse as sparse

from .io import core_dataset

MEASURES = [
    "area",
    "area.mid",
    "area.pial",
    "curv",
    "curv.pial",
    "jacobian_white",
    "sulc",
    "thickness",
    "volume",
]
PARCELLATIONS = ["aparc", "aparc.DKTatlas", "aparc.a2009s", "HCP_MMP"]
# HCP-MMP1.0 (Glasser et al., 2016) groups, made by scripts/resample_HCP_MMP.py. The two 210s are
# the averaged per-subject areal-classifier parcellations of the parcellation (210P) and validation
# (210V) subjects; Related420 is the mean of the two. The default is 210V, which was not used to
# draw the parcellation or train the classifier.
HCP_MMP_DEFAULT_GROUP = "Q1-Q6_RelatedValidation210"
HCP_MMP_GROUPS = (
    "Q1-Q6_RelatedParcellation210",
    "Q1-Q6_RelatedValidation210",
    "Q1-Q6_Related420",
)


# Per-area groupings of HCP_MMP (tables in support_files/, see README_HCPMMP1.md):
#   "Glasser22": the 22 cortices = primary text sections of Glasser et al. 2016 Supplementary Neuroanatomical Results
#   "ColeAnticevic": Cole-Anticevic Brain-wide Network Partition v1.1.6 (Ji et al., 2019), per hemisphere
HCP_MMP_NETWORKS = ("Glasser22", "ColeAnticevic")
_SUPPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "support_files")


def _hcp_mmp_tables():
    import pandas as pd

    regions = pd.read_csv(
        os.path.join(_SUPPORT, "atlas-HCPMMP1_keylabel.tsv"), sep="\t"
    )
    ca = pd.read_csv(
        os.path.join(_SUPPORT, "atlas-HCPMMP1_ColeAnticevic_keylabel.tsv"), sep="\t"
    )
    ca_nets = pd.read_csv(
        os.path.join(_SUPPORT, "atlas-ColeAnticevic_networks.tsv"), sep="\t"
    )
    return regions, ca, ca_nets


def _hcp_mmp_network_lookup(networks, lr):
    """(181,) int array: area key (1-180, index 0 unused) -> network key for hemisphere ``lr``."""
    regions, ca, _ = _hcp_mmp_tables()
    lut = np.full(181, -1, dtype=int)
    if networks == "Glasser22":
        lut[regions["key"].values] = regions["glasser22_key"].values
    elif networks == "ColeAnticevic":
        rows = ca[ca["hemi"] == lr.upper()]
        lut[rows["key"].values] = rows["network_key"].values
    else:
        raise ValueError(
            f"networks must be one of {HCP_MMP_NETWORKS}; got {networks!r}"
        )
    return lut


def get_parcellation_names(which, lr=None, networks=None):
    """Names of the labels returned by get_parcellation, as {key: name}.

    Only 'HCP_MMP' has names in neuroboros. Without ``networks`` the keys are the area keys 1-180 (names like
    'V1'; with ``lr`` given they are prefixed 'L_' / 'R_'). With ``networks`` ('Glasser22' or 'ColeAnticevic') the
    keys are the network keys returned by ``get_parcellation(..., networks=...)``.
    """
    if which != "HCP_MMP":
        raise NotImplementedError(f"neuroboros has no region names for {which!r}")
    regions, _, ca_nets = _hcp_mmp_tables()
    if networks is None:
        prefix = f"{lr.upper()}_" if lr else ""
        return {int(k): prefix + n for k, n in zip(regions["key"], regions["label"])}
    if networks == "Glasser22":
        g = regions.drop_duplicates("glasser22_key").sort_values("glasser22_key")
        return {int(k): n for k, n in zip(g["glasser22_key"], g["glasser22_label"])}
    if networks == "ColeAnticevic":
        return {
            int(k): n for k, n in zip(ca_nets["network_key"], ca_nets["network_label"])
        }
    raise ValueError(f"networks must be one of {HCP_MMP_NETWORKS}; got {networks!r}")


def get_parcellation_colors(which, lr, networks=None):
    """Published colours of the labels returned by get_parcellation, as {key: (r, g, b)} in 0-1.

    'HCP_MMP' areas: the label-table colours of the HCP-MMP1.0 dlabel (BALSA RVVG
    Q1-Q6_RelatedValidation210 ..._Group_Colors), which differ between hemispheres. networks='ColeAnticevic':
    the ColeLab network colours. networks='Glasser22': the region colours of Glasser et al. 2016, Supplementary
    Neuroanatomical Results, Figure 1 (same for both hemispheres; ``lr`` is ignored).
    """
    import pandas as pd

    if which != "HCP_MMP":
        raise NotImplementedError(f"neuroboros has no colours for {which!r}")
    if networks is None:
        t = pd.read_csv(os.path.join(_SUPPORT, "atlas-HCPMMP1_colors.tsv"), sep="\t")
        t = t[t["hemi"] == lr.upper()]
        key = "key"
    elif networks == "ColeAnticevic":
        t = pd.read_csv(
            os.path.join(_SUPPORT, "atlas-ColeAnticevic_networks.tsv"), sep="\t"
        )
        key = "network_key"
    elif networks == "Glasser22":
        t = pd.read_csv(
            os.path.join(_SUPPORT, "atlas-HCPMMP1_Glasser22_colors.tsv"), sep="\t"
        )
        key = "glasser22_key"
    else:
        raise NotImplementedError(f"no published colours for networks={networks!r}")
    return {
        int(k): (r / 255, g / 255, b / 255)
        for k, r, g, b in zip(t[key], t["r"], t["g"], t["b"])
    }


def check_hcp_mmp_group(group):
    """Raise unless ``group`` is one of HCP_MMP_GROUPS."""
    if group not in HCP_MMP_GROUPS:
        raise ValueError(
            f"HCP_MMP group must be one of {HCP_MMP_GROUPS}; got {group!r}"
        )


def get_morphometry(which, lr, space="onavg-ico32", **kwargs):
    """Group-based morphometry measure.

    Parameters
    ----------
    which : str
        Which morphometric measure to get. One of the following:
        'area', 'area.mid', 'area.pial', 'curv', 'curv.pial',
        'jacobian_white', 'sulc', 'thickness', 'volume'.
    lr : str
        Hemisphere, either 'l', 'r', or 'lr'.
    space : str, default='onavg-ico32'
        Surface space.

    Returns
    -------
    measure : ndarray
        Morphometric measure. Shape (n_vertices,).
    """
    assert which in MEASURES
    if lr == "lr":
        lh = get_morphometry(which, "l", space=space, **kwargs)
        rh = get_morphometry(which, "r", space=space, **kwargs)
        return np.concatenate([lh, rh])
    group = kwargs.get("group", "on1031")
    resample = kwargs.get("resample", "overlap-8div")
    avg_type = kwargs.get("avg_type", "trimmed")
    assert lr in "lr"
    fn = os.path.join(
        space, "morphometry", which, f"{lr}h", f"{group}_{avg_type}", f"{resample}.npy"
    )
    measure = core_dataset.get(fn, on_missing="raise")
    return measure


def get_parcellation(
    which, lr, space="onavg-ico32", prob=False, networks=None, **kwargs
):
    """Group-based parcellation.

    Parameters
    ----------
    which : str
        Which parcellation to get. One of the following:
        'aparc', 'aparc.DKTatlas', 'aparc.a2009s', 'HCP_MMP'.
    lr : str
        Hemisphere, either 'l', 'r', or 'lr'.
    space : str, default='onavg-ico32'
        Surface space.
    prob : bool, default=False
        Whether to load the probabilistic version of the parcellation.
    networks : str or None, default=None
        'HCP_MMP' only: return network keys instead of area keys, 'Glasser22' (1-22) or 'ColeAnticevic'
        (1-12, assigned per hemisphere). Unassigned vertices stay -1. With ``prob=True`` the area probability
        maps are summed within each network (row k-1 = network k): the fraction of the group's subjects whose
        area at that vertex belongs to network k. Neither grouping published a probabilistic version; this is
        derived. Names: get_parcellation_names(which, networks=...).
    **kwargs : dict
        Optional keyword arguments. ``group`` is the participant group: for
        'HCP_MMP', one of HCP_MMP_GROUPS: 'Q1-Q6_RelatedValidation210' (default),
        'Q1-Q6_RelatedParcellation210', or 'Q1-Q6_Related420'; for
        other parcellations, it defaults to 'on1031' with ``avg_type``
        defaulting to 'trimmed'. ``resample`` is the resampling method,
        default 'overlap-8div'.

    Returns
    -------
    parc : ndarray
        The cortical parcellation. It is a discrete parcellation if ``prob``
        is False, and a probabilistic parcellation if ``prob`` is True.
    """
    assert which in PARCELLATIONS
    if lr == "lr":
        lh = get_parcellation(
            which, "l", space=space, prob=prob, networks=networks, **kwargs
        )
        rh = get_parcellation(
            which, "r", space=space, prob=prob, networks=networks, **kwargs
        )
        # A probabilistic parcellation has shape (n_parcels, n_vertices), and
        # the vertex axis is the last one rather than the first one.
        return np.concatenate([lh, rh], axis=-1 if prob else 0)
    resample = kwargs.get("resample", "overlap-8div")
    assert lr in "lr"
    if prob:
        basename = f"{resample}_prob.npy"
    else:
        basename = f"{resample}_parc.npy"
    if which == "HCP_MMP":
        group = kwargs.get("group", HCP_MMP_DEFAULT_GROUP)
        check_hcp_mmp_group(group)
        group_dir = group  # HCP group labels: no onavg avg_type suffix
    else:
        group = kwargs.get("group", "on1031")
        avg_type = kwargs.get("avg_type", "trimmed")
        group_dir = f"{group}_{avg_type}"
    fn = os.path.join(space, "parcellations", which, f"{lr}h", group_dir, basename)
    parc = core_dataset.get(fn, on_missing="raise")
    if networks is not None:
        if which != "HCP_MMP":
            raise ValueError(f"networks is only available for 'HCP_MMP', not {which!r}")
        lut = _hcp_mmp_network_lookup(networks, lr)
        if prob:  # (180, n_vertices) area probabilities -> (n_networks, n_vertices)
            out = np.zeros((lut[1:].max(), parc.shape[1]), dtype=parc.dtype)
            np.add.at(out, lut[1:] - 1, parc)
            return out
        return np.where(parc > 0, lut[np.clip(parc, 0, None)], parc)
    return parc


def get_mask(lr, space="onavg-ico32", legacy=False, **kwargs):
    """Standard cortical mask.

    Parameters
    ----------
    lr : str
        Hemisphere, either 'l' or 'r' or 'lr'.
    space : str, default='onavg-ico32'
        Surface space.
    legacy : bool, default=False
        Whether to load the legacy version of the mask, e.g., "fsaverage" or
        "32k_fs_LR", instead of the group-based cortical mask.

    Returns
    -------
    mask : ndarray
        Mask of the cortical surface. Shape (n_vertices,).
    """
    if lr == "lr":
        lh = get_mask("l", space=space, legacy=legacy, **kwargs)
        rh = get_mask("r", space=space, legacy=legacy, **kwargs)
        M = np.concatenate([lh, rh], axis=0)
        return M
    assert lr in "lr"

    resample = kwargs.get("resample", "overlap-8div")
    which = kwargs.get("which", "aparc.a2009s")
    if space == "MEBRAIN" or space.startswith("mkavg-ico"):
        flavor = "MEBRAIN"
        which = "MEBRAIN"
        fn = os.path.join(
            space, "masks", which, f"{lr}h", f"{flavor}", f"{resample}.npy"
        )
    elif legacy:
        if space.startswith("fsavg-ico"):
            flavor = kwargs.get("flavor", "fsaverage")
        elif space.startswith("fslr-ico"):
            flavor = kwargs.get("flavor", "32k_fs_LR")
        fn = os.path.join(
            space, "masks", which, f"{lr}h", f"{flavor}", f"{resample}.npy"
        )
    else:
        group = kwargs.get("group", "on1031")
        avg_type = kwargs.get("avg_type", "trimmed")
        fn = os.path.join(
            space, "masks", which, f"{lr}h", f"{group}_{avg_type}", f"{resample}.npy"
        )
    mask = core_dataset.get(fn, on_missing="raise")
    return mask


def get_geometry(which, lr, space="onavg-ico32", vertices_only=False, **kwargs):
    """Surface geometry.

    Parameters
    ----------
    which : str
        Which geometry to get. One of the following:
        'sphere', 'sphere.reg', 'white', 'pial', 'inflated', 'midthickness',
        'faces'.
    lr : str
        Hemisphere, either 'l', 'r', or 'lr'.
    space : str, default='onavg-ico32'
        Surface space.
    vertices_only : bool, default=False
        Whether to return only the coordinates of the vertices.

    Returns
    -------
    coords : ndarray
        Coordinates of the vertices. Shape (n_vertices, 3).
        When ``which == 'faces'``, ``coords`` will not be returned.
    faces : ndarray
        Faces of the triangle surface mesh. Shape (n_faces, 3).
        Only returned when ``vertices_only == False``.
    """
    group = kwargs.get("group", "on1031")
    avg_type = kwargs.get("avg_type", "trimmed")
    if lr == "lr":
        if which == "faces":
            l_faces = get_geometry(which, "l", space=space, **kwargs)
            r_faces = get_geometry(which, "r", space=space, **kwargs)
            n_l = len(get_geometry("sphere", "l", space=space, vertices_only=True))
            return np.concatenate([l_faces, r_faces + n_l])
        l_coords = get_geometry(which, "l", space=space, vertices_only=True, **kwargs)
        r_coords = get_geometry(which, "r", space=space, vertices_only=True, **kwargs)
        coords = np.concatenate([l_coords, r_coords])
        if vertices_only:
            return coords
        l_faces = get_geometry("faces", "l", space=space, **kwargs)
        r_faces = get_geometry("faces", "r", space=space, **kwargs)
        faces = np.concatenate([l_faces, r_faces + len(l_coords)])
        return coords, faces
    assert lr in "lr"
    if not vertices_only:
        ffn = os.path.join(space, "geometry", "faces", f"{lr}h.npy")
        faces = core_dataset.get(ffn, on_missing="raise")
        if which == "faces":
            return faces

    if which in ["sphere", "sphere.reg"]:
        fn = os.path.join(space, "geometry", "sphere.reg", f"{lr}h.npy")
    else:
        fn = os.path.join(space, "geometry", which, f"{lr}h", f"{group}_{avg_type}.npy")
    coords = core_dataset.get(fn, on_missing="raise")
    if vertices_only:
        return coords

    return coords, faces


def get_distances(
    lr, source, target=None, mask=None, source_mask=None, target_mask=None, **kwargs
):
    """Distances between vertices.

    Parameters
    ----------
    lr : str
        Hemisphere, either 'l' or 'r'.
    source : str
        Source space.
    target : str, default=None
        Target space. If None, assuming it's the same as ``source``.
    mask : str, default=None
        Mask to apply to the distance matrix. If None or False, no mask is
        applied. If True, the group-based cortical mask is used.
        If a boolean array, it is used as the mask.
        ``source_mask`` and ``target_mask`` can be set separately, and they
        take precedence over ``mask``.
    source_mask : str, default=None
        Mask to apply to source space (rows) of the distance matrix.
    target_mask : str, default=None
        Mask to apply to target space (columns) of the distance matrix.

    Returns
    -------
    M : ndarray
        Distance matrix. The shape is (n_source_vertices, n_target_vertices).
    """
    group = kwargs.get("group", "on1031")
    avg_type = kwargs.get("avg_type", "trimmed")
    dist_type = kwargs.get("dist_type", "dijkstra")

    if target is None:
        target = source
    if source_mask is None:
        source_mask = mask
    if target_mask is None:
        target_mask = mask

    fn = os.path.join(
        source,
        "distances",
        f"to_{target}",
        f"{lr}h",
        f"{group}_{avg_type}",
        f"{dist_type}.npy",
    )

    assert target == source
    d = core_dataset.get(fn, on_missing="raise")
    ico = int(source.split("-ico")[1])
    nv = ico**2 * 10 + 2
    mat = np.zeros((nv, nv), dtype=d.dtype)
    idx1, idx2 = np.triu_indices(nv, 1)
    mat[idx1, idx2] = d
    mat = np.maximum(mat, mat.T)
    del d

    if source_mask is not None:
        if isinstance(source_mask, np.ndarray):
            mask1 = source_mask
        else:
            mask1 = get_mask(lr, source, **kwargs)
    if target_mask is not None:
        if isinstance(target_mask, np.ndarray):
            mask2 = target_mask
        else:
            mask2 = get_mask(lr, target, **kwargs)

    if source_mask is not None and target_mask is not None:
        M = mat[np.ix_(mask1, mask2)]
    elif source_mask is not None:
        M = mat[mask1, :]
    elif target_mask is not None:
        M = mat[:, mask2]
    else:
        M = mat

    return M


def smooth(lr, fwhm=None, sigma=None, space="onavg-ico32", mask=None, keep_sum=False):
    """Get a smoothing matrix.

    Parameters
    ----------
    lr : str
        Hemisphere(s), either 'l', 'r', or 'lr'.
    fwhm : float
        Full-width at half-maximum of the Gaussian kernel.
    sigma : float
        Standard deviation of the Gaussian kernel.
    space : str, default='onavg-ico32'
        Surface space where the data is in.
    mask : ndarray or bool or None, default=None
        Mask to apply to the smoothing matrix. If None or False, no mask is
        applied. If True, the standard cortical mask is used. If an ndarray,
        it is used as the mask.
    keep_sum : bool, default=False
        If True, keep the sum of the data. Useful for transforming area,
        volume, etc., where the total area/volume is preserved when
        ``keep_sum=True``.

    Returns
    -------
    M : sparse matrix
        Smoothing matrix. Can be applied to data matrix ``X`` as ``X @ M``.
    """
    if lr == "lr":
        lh = smooth("l", fwhm=fwhm, space=space, mask=mask, keep_sum=keep_sum)
        rh = smooth("r", fwhm=fwhm, space=space, mask=mask, keep_sum=keep_sum)
        M = sparse.block_diag([lh, rh], format="csr")
        return M

    if fwhm is None and sigma is None:
        raise ValueError("Either `fwhm` or `sigma` must be provided.")
    if fwhm is not None and sigma is not None:
        raise ValueError("Only one of `fwhm` or `sigma` must be provided.")

    if fwhm is not None:
        assert fwhm >= 0
        s2 = fwhm**2 / (4.0 * np.log(2))  # 2 * sigma^2
    else:
        assert sigma >= 0
        s2 = 2 * sigma**2

    if fwhm == 0:
        mask_arr = get_mask(lr, space=space)
        if mask:
            nv = mask_arr.sum()
        else:
            nv = mask_arr.shape[0]
        M = sparse.eye(nv, format="csr")
        return M

    d = get_distances(lr, space, space, mask=mask)
    weights = np.exp(-(d**2) / s2)  # ignoring the 1/(2*pi*sigma^2) factor
    mat = sparse.csr_matrix(weights)

    if keep_sum:
        with np.errstate(divide="ignore"):
            d = np.nan_to_num(np.reciprocal(mat.sum(axis=1).A.ravel()))
        M = sparse.diags(d) @ mat
    else:
        with np.errstate(divide="ignore"):
            d = np.nan_to_num(np.reciprocal(mat.sum(axis=0).A.ravel()))
        M = mat @ sparse.diags(d)

    return M


def get_mapping(
    lr,
    source,
    target,
    mask=None,
    nn=False,
    keep_sum=False,
    source_mask=None,
    target_mask=None,
    **kwargs,
):
    """Get mapping (transform) from one space to another.

    Parameters
    ----------
    lr : str
        Hemisphere(s), either 'l', 'r', or 'lr'.
    source : str
        Source space, the space where the data is currently in.
    target : str
        Target space, the space where the data will be transformed into.
    mask : bool or boolean array or None
        Mask to apply to the mapping. If None or False, no mask is applied.
        If True, the group-based cortical mask is used.
        If a boolean array, it is used as the mask.
        ``source_mask`` and ``target_mask`` can be set separately, and they
        take precedence over ``mask``.
    nn : bool
        If True, use nearest neighbor interpolation.
        If False, use overlap-area-based interpolation.
    keep_sum : bool
        If True, keep the sum of the data. Useful for transforming area,
        volume, etc., where the total area/volume is preserved when
        ``keep_sum=True``.
    source_mask : bool or boolean array or None
        Mask to apply to the source space. Similar to ``mask``.
    target_mask : bool or boolean array or None
        Mask to apply to the target space. Similar to ``mask``.

    Returns
    -------
    M : sparse matrix
        Mapping matrix. Can be applied to data in the source space to
        transform it into the target space. For example, if ``X`` is a
        data matrix in the source space, then ``X @ M`` is the data matrix
        in the target space.
    """
    if lr == "lr":
        lh = get_mapping(
            "l",
            source,
            target,
            mask=mask,
            nn=nn,
            keep_sum=keep_sum,
            source_mask=source_mask,
            target_mask=target_mask,
            **kwargs,
        )
        rh = get_mapping(
            "r",
            source,
            target,
            mask=mask,
            nn=nn,
            keep_sum=keep_sum,
            source_mask=source_mask,
            target_mask=target_mask,
            **kwargs,
        )
        M = sparse.block_diag([lh, rh], format="csr")
        return M

    species = (
        "macaque" if source.startswith("mkavg-") or source == "MEBRAIN" else "human"
    )
    if species == "human":
        group, avg_type = "on1031", "trimmed"
    elif species == "macaque":
        group, avg_type = "mk12", "average"
    else:
        raise ValueError(f"Unknown species: {species}")

    group = kwargs.get("group", group)
    resample = kwargs.get("resample", "overlap-8div")
    avg_type = kwargs.get("avg_type", avg_type)

    if source_mask is None:
        source_mask = mask
    if target_mask is None:
        target_mask = mask

    if source == target:
        ico = int(source.split("-ico")[1])
        nv = ico**2 * 10 + 2
        mat = sparse.diags(np.ones((nv,))).tocsr()
        # print(mat.shape, mat.data.shape, type(mat))
    else:
        fn1 = os.path.join(
            source,
            "mapping",
            f"to_{target}",
            f"{lr}h",
            f"{group}_{avg_type}",
            f"{resample}.npz",
        )
        fn2 = os.path.join(
            target,
            "mapping",
            f"to_{source}",
            f"{lr}h",
            f"{group}_{avg_type}",
            f"{resample}.npz",
        )
        local_fn1 = os.path.join(core_dataset.root, fn1)
        local_fn2 = os.path.join(core_dataset.root, fn2)
        mat1 = mat2 = None
        if os.path.exists(local_fn1):
            mat1 = core_dataset.get(fn1)
        elif os.path.exists(local_fn2):
            mat2 = core_dataset.get(fn2)
        else:
            try:
                mat1 = core_dataset.get(fn1, on_missing="ignore")
            except Exception:
                pass
            if mat1 is None:
                try:
                    mat2 = core_dataset.get(fn2, on_missing="ignore")
                except Exception:
                    pass
        assert mat1 is not None or mat2 is not None, f"Neither {fn1} nor {fn2} exists."
        mat = mat1 if mat1 is not None else mat2.T

    if source_mask is not None and source_mask is not False:
        if isinstance(source_mask, np.ndarray):
            mask1 = source_mask
        else:
            mask1 = get_mask(lr, source, **kwargs)
    if target_mask is not None and target_mask is not False:
        if isinstance(target_mask, np.ndarray):
            mask2 = target_mask
        else:
            mask2 = get_mask(lr, target, **kwargs)

    if source_mask is not None and target_mask is not None:
        mat = mat[np.ix_(mask1, mask2)]
    elif source_mask is not None:
        mat = mat[mask1, :]
    elif target_mask is not None:
        mat = mat[:, mask2]
    else:
        pass

    if nn:
        idx = mat.argmax(axis=0).A.ravel()
        m, n = mat.shape
        M = sparse.csr_matrix((np.ones((n,)), (idx, np.arange(n))), (m, n))
        return M

    if keep_sum:
        with np.errstate(divide="ignore"):
            d = np.nan_to_num(np.reciprocal(mat.sum(axis=1).A.ravel()))
        M = sparse.diags(d) @ mat
    else:
        with np.errstate(divide="ignore"):
            d = np.nan_to_num(np.reciprocal(mat.sum(axis=0).A.ravel()))
        M = mat @ sparse.diags(d)

    return M
