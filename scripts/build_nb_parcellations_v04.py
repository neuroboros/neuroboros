#!/usr/bin/env python3
"""
Resample label parcellations (Yeo, Schaefer, Kong, Gordon, Shen, Han, ...) to many neuroboros
target spaces and save them directly into the neuroboros data directory, next to the existing
anatomical parcellations (aparc, aparc.a2009s, HCP_MMP, ...).

Method (same as resample_HCP_MMP.py)
------------------------------------
For each hemisphere separately, every parcel k gets a binary source map (1 inside parcel k,
0 elsewhere) that is multiplied by the unmasked overlap-area mapping matrix
``M = nb.mapping('l' or 'r', source, target)`` of shape (n_src_vertices, n_tgt_vertices).
All parcels are done at once with a sparse one-hot matrix (identical to looping over parcels).
Each target vertex goes to the parcel with the highest probability (winner-take-all); vertices
that receive no probability from any parcel are unassigned (-1).

Output layout  (OUT_ROOT = ~/.neuroboros-data/core)
------------------------------------------------
<OUT_ROOT>/<target>/parcellations/<name>/<lh|rh>/on1031_trimmed/
    overlap-8div_prob.npy   float32 (K_h, nv_target): row k-1 = probability of parcel k
    overlap-8div_parc.npy   int (nv_target,): 1-indexed per-hemisphere parcel label, -1 unassigned
    → loadable with nb.parcellation('<name>', 'l', space='<target>') once spaces.py is updated

<OUT_ROOT>/<target>/other/<name>/
    label_key.csv                          hemi, hemi_label, original_id, source file, counts, network
    network_order.json / .csv              (Yeo only) label -> network name
    figures/<name>_<target>.png            parcellation figure
    figures/<name>_<target>_boundary.png   with boundary vertices in black
    <lh|rh>/on1031_trimmed/
        overlap-8div_parc_noboundary.npy   same as the official parc file
        overlap-8div_parc_boundary.npy     boundary vertices = -2, unassigned = -1
        overlap-8div_vertices.csv          one row per vertex (labels, boundary flag, network)
        overlap-8div_probs.csv             long format, non-zero probabilities only

Folder names (year dropped)
---------------------------
schaefer_400parcels_7networks, kong_400parcels_17networks, yeo_7networks, gordon_333,
shen_268, han_young_adult / han_middle_early / han_middle_late / han_older_early / han_older_late,
coleantecivic_parcels (CAB-NP parcels) and coleantecivic_networks (CAB-NP network assignments).
When a dlabel and a dscalar (or label.gii / func.gii) file give the same parcellation, only one
is used, in this order of preference: .dlabel.nii > .label.gii > .dscalar.nii > other .gii > other.

Per-hemisphere numbering
------------------------
If the hemispheres use different parcel IDs (Schaefer 1-400 / 401-800, Gordon, Shen with RH
first), each hemisphere is renumbered 1..K in ID order. If they share IDs (Yeo networks, files
that number each hemisphere from 1), the original IDs are kept. label_key.csv maps every
per-hemisphere label back to its original ID.

Source-space detection
----------------------
Every input file's per-hemisphere vertex count (for CIFTI: the cortical surface size from the
header, subcortex ignored) is looked up in GUESS_SEPARATE / GUESS_COMBINED. The detected space
is checked against SOURCE_SPACES: if they agree it is confirmed, if not the detected space is
used and a warning is logged. Masked inputs (cortex-only values) are expanded to the full mesh
with nb.mask before resampling.

Existing files
--------------
Before each target, the script logs which functional parcellations already exist and whether
their official files are complete and readable. Existing files are then OVERWRITTEN (use
--skip-existing to keep complete ones). Every file is written to a temporary name first and
only renamed into place once fully written, so a full disk never leaves a truncated file; if
the disk fills up, the run stops immediately with a clear message.

Usage
-----
python build_nb_parcellations_v04.py                          # all targets, all parcellations
python build_nb_parcellations_v04.py --targets onavg-ico32    # one target
python build_nb_parcellations_v04.py --parcellations coleantecivic2019 --skip-existing
python build_nb_parcellations_v04.py --detect-only            # only report each file's source space
"""

import os
import re
import sys
import json
import time
import errno
import shutil
import argparse
import traceback
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
import nibabel as nib

import matplotlib
matplotlib.use('Agg')                       # headless (SLURM)
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import neuroboros as nb


# ═══════════════════════════════════ Settings ═══════════════════════════════════
ROOT      = '/dartfs/rc/lab/H/HaxbyLab'
PARC_ROOT = os.path.join(ROOT, 'jason', 'haxby_parcellations')   # <parc>/original_parcellations
OUT_ROOT  = '/dartfs-hpc/rc/home/3/f005g23/.neuroboros-data/core'

# Targets from resample_HCP_MMP.py, plus the native fsLR 32k mesh
TARGETS = [
    'onavg-ico4', 'onavg-ico8', 'onavg-ico16', 'onavg-ico32', 'onavg-ico48',
    'onavg-ico64', 'onavg-ico128',
    'fsavg-ico32', 'fsavg-ico64', 'fsavg-ico128',
    'fslr-ico32', 'fslr-ico64', 'fslr-ico128',
    'fslr-ico57',
]

PARCELLATIONS = ['yeo2010', 'schaefer2018', 'kong2022', 'gordon2016', 'shen2013', 'han2018',
                 'coleantecivic2019']

DEFAULT_SOURCE_SPACE = 'fslr-ico57'
SOURCE_SPACES = {'yeo2010': 'fslr-ico57', 'schaefer2018': 'fslr-ico57', 'kong2022': 'fslr-ico57',
                 'gordon2016': 'fslr-ico57', 'shen2013': 'fsavg-ico128', 'han2018': 'fslr-ico57',
                 'coleantecivic2019': 'fslr-ico57'}
ANATOMICAL = {'aparc', 'aparc.DKTatlas', 'aparc.a2009s', 'HCP_MMP'}   # never touched or listed
PROB_DTYPE = np.float32

GROUP_DIR        = 'on1031_trimmed'   # = nb default group 'on1031' + avg_type 'trimmed'
RESAMPLE         = 'overlap-8div'
UNASSIGNED_LABEL = -1
BOUNDARY_LABEL   = -2                 # only used in the "other" boundary files

MIN_DISTANCE         = 5.0     # mm, neighbour radius for boundaries
MAX_DISTANCE         = 60.0    # radius is doubled on coarse meshes until most vertices have neighbours
MAX_NO_NEIGHBOR_FRAC = 0.10
SHARED_FRAC          = 0.5     # > this fraction of IDs shared between hemis → bilateral labels

SAVE_PROBS_CSV  = True
MAKE_PLOTS      = True
VERIFY_WITH_NB  = True         # also reload official files through neuroboros (needs updated spaces.py)

# Han et al. (2018) age-group prefixes → folder names
HAN_GROUPS = {'YA': 'young_adult', 'ME': 'middle_early', 'ML': 'middle_late',
              'OE': 'older_early', 'OL': 'older_late'}

# Optional manual folder names: {(parcellation, file_key): 'folder_name'}
VARIANT_NAMES = {}

# Yeo network colouring
NETWORK_PARCELLATIONS = {'yeo2010'}
SEED_SPACE   = 'onavg-ico32'   # space the seed vertices below are defined in
YEO_NETWORKS = ['Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention',
                'Limbic', 'Frontoparietal', 'Default Mode']   # no Auditory in Yeo-7

HEMIS = ('lh', 'rh')


# ═══════════════════════════════ Space / parcellation guessing tables ═══════════════════════════════
GUESS_SEPARATE = {
    # masked
    (154786, 154560): ("onavg-ico128", True),
    (9675, 9666): ("onavg-ico32", True),
    (9519, 9506): ("fsavg-ico32", True),
    (9504, 9483): ("fslr-ico32", True),
    (38698, 38628): ("onavg-ico64", True),
    (38073, 38040): ("fsavg-ico64", True),
    (38018, 37943): ("fslr-ico64", True),
    (21779, 21731): ("onavg-ico48", True),
    (30153, 30079): ("fslr-ico57", True),
    (152, 151): ("onavg-ico4", True),
    (603, 607): ("onavg-ico8", True),
    (2417, 2414): ("onavg-ico16", True),
    # masked, legacy
    (9372, 9370): ("fsavg-ico32", True, "fsaverage"),
    # non-masked
    (10242, 10242): ("onavg-ico32", False),
    (40962, 40962): ("onavg-ico64", False),
    (23042, 23042): ("onavg-ico48", False),
    (32492, 32492): ("fslr-ico57", False),
    (162, 162): ("onavg-ico4", False),
    (642, 642): ("onavg-ico8", False),
    (2562, 2562): ("onavg-ico16", False),
}

GUESS_PARCELLATION_SEPARATE = {
    (180, 180): "HCP_MMP",
}

GUESS_PARCELLATION_COMBINED = {
    180: ("HCP_MMP", True),  # bilateral: same values for L and R homologs
    360: ("HCP_MMP", False),  # separate: L (0-179) and R (180-359)
}

GUESS_COMBINED = {
    # masked
    309346: ("onavg-ico128", True, [154786]),
    19341: ("onavg-ico32", True, [9675]),
    19025: ("fsavg-ico32", True, [9519]),
    18987: ("fslr-ico32", True, [9504]),
    77326: ("onavg-ico64", True, [38698]),
    76113: ("fsavg-ico64", True, [38073]),
    75961: ("fslr-ico64", True, [38018]),
    43510: ("onavg-ico48", True, [21779]),
    60232: ("fslr-ico57", True, [30153]),
    303: ("onavg-ico4", True, [152]),
    1210: ("onavg-ico8", True, [603]),
    4831: ("onavg-ico16", True, [2417]),
    # masked, legacy
    18742: ("fsavg-ico32", True, [9372], "fsaverage"),
    # non-masked
    20484: ("onavg-ico32", False, [10242]),
    81924: ("onavg-ico64", False, [40962]),
    46084: ("onavg-ico48", False, [23042]),
    64984: ("fslr-ico57", False, [32492]),
    324: ("onavg-ico4", False, [162]),
    1284: ("onavg-ico8", False, [642]),
    5124: ("onavg-ico16", False, [2562]),
}

PLOT_MAPPING = {}


def _ico_of(space):
    return int(space.split('-ico')[1])


def _icosahedral_ico(n):
    """ico if n == 10*ico^2 + 2 (a full icosahedral mesh), else None."""
    ico = int(round(np.sqrt(max(n - 2, 0) / 10)))
    return ico if ico > 0 and 10 * ico ** 2 + 2 == n else None


def _refine_family(space, hint, is_cifti):
    """Full-mesh vertex counts are the same for onavg/fsavg/fslr at one resolution, so pick the
    family from the configured space (same ico), then CIFTI (always fsLR), then the table."""
    ico = _ico_of(space)
    if hint and _ico_of(hint) == ico:
        return hint
    if is_cifti:
        return f'fslr-ico{ico}'
    return space


def guess_source_space(n_lh, n_rh, hint=None, is_cifti=False):
    """(space, masked, legacy_flavor, how) for per-hemisphere vertex counts, or None."""
    entry = GUESS_SEPARATE.get((n_lh, n_rh))
    if entry is not None:
        space, masked = entry[0], entry[1]
        flavor = entry[2] if len(entry) > 2 else None
        if not masked:
            space = _refine_family(space, hint, is_cifti)
        return space, masked, flavor, 'GUESS_SEPARATE'
    if n_lh == n_rh and _icosahedral_ico(n_lh):
        ico = _icosahedral_ico(n_lh)
        family = hint.split('-ico')[0] if hint else ('fslr' if is_cifti else 'onavg')
        return f'{family}-ico{ico}', False, None, 'icosahedral vertex count'
    return None


def split_combined(arr, f, hint=None, is_cifti=False):
    """Split a concatenated LH+RH array using GUESS_COMBINED (or an even icosahedral split)."""
    n = arr.size
    entry = GUESS_COMBINED.get(n)
    if entry is not None:
        n_lh = entry[2][0]
        return {'lh': arr[:n_lh], 'rh': arr[n_lh:]}
    if n % 2 == 0 and _icosahedral_ico(n // 2):
        return {'lh': arr[:n // 2], 'rh': arr[n // 2:]}
    print(f'  ⚠️  {f}: {n} values — not a known LH+RH vertex count (GUESS_COMBINED). Skipping.')
    return {}


def unmask_to_full(arr, lr, space, flavor=None):
    """Put cortex-only values back on the full mesh (medial wall = 0)."""
    if flavor:
        mask = np.asarray(nb.mask(lr, space=space, legacy=True, flavor=flavor), bool)
    else:
        mask = np.asarray(nb.mask(lr, space=space, legacy=False), bool)
    if mask.sum() != arr.size:
        raise ValueError(f'{space} {lr}h mask has {mask.sum()} vertices, data has {arr.size}')
    full = np.zeros(mask.size, dtype=float)
    full[mask] = arr
    return full


def log_parcellation_guess(name, ids):
    """Note when the parcel counts match a known parcellation layout (e.g. HCP_MMP)."""
    sep = GUESS_PARCELLATION_SEPARATE.get((ids['lh'].size, ids['rh'].size))
    comb = GUESS_PARCELLATION_COMBINED.get(np.union1d(ids['lh'], ids['rh']).size)
    if sep:
        print(f'    Parcel counts {ids["lh"].size}/{ids["rh"].size} match the {sep} layout')
    elif comb:
        print(f'    {np.union1d(ids["lh"], ids["rh"]).size} parcel IDs match the {comb[0]} layout '
              f'({"bilateral" if comb[1] else "separate L/R"})')


# ═══════════════════════════════════ Colours ═══════════════════════════════════
# >>> Replace with your notebook's watershed_gradient so figures match <<<
WATERSHED_GRADIENT = mpl.colormaps['nipy_spectral']

networks_yeo17 = {
    "N1: VisCent":   (0.80, 0.00, 0.80),
    "N2: VisPeri":   (1.00, 0.00, 0.00),
    "N3: SomMotA":   (0.00, 0.50, 1.00),
    "N4: SomMotB":   (0.00, 1.00, 0.50),
    "N5: DorsAttnA": (0.00, 0.70, 0.00),
    "N6: DorsAttnB": (0.00, 0.90, 0.00),
    "N7: SalVentA":  (1.00, 0.00, 1.00),
    "N8: SalVentB":  (1.00, 0.50, 1.00),
    "N9: LimbicB":   (0.70, 0.70, 0.00),
    "N10: LimbicA":  (1.00, 0.80, 0.00),
    "N11: ContA":    (1.00, 0.65, 0.00),
    "N12: ContB":    (0.70, 0.00, 0.50),
    "N13: ContC":    (0.50, 0.70, 1.00),
    "N14: DefaultA": (1.00, 1.00, 0.00),
    "N15: DefaultB": (1.00, 0.00, 0.00),
    "N16: DefaultC": (0.00, 0.00, 1.00),
    "N17: TempPar":  (0.00, 0.00, 0.80),
    "Black":         (0.00, 0.00, 0.00),
}
YEO17_ORDER = [k for k in networks_yeo17 if k != 'Black']   # label 1 → N1, …, label 17 → N17

master_colors = {
    "Ventral Attention": (0.92, 0.30, 1.00),
    "Limbic":            (1.00, 0.88, 0.65),
    "Frontoparietal":    (1.00, 0.55, 0.00),
    "Default Mode":      (0.95, 0.20, 0.20),
    "Visual":            (0.42, 0.00, 0.78),
    "Somatomotor":       (0.00, 0.35, 1.00),
    "Dorsal Attention":  (0.00, 0.65, 0.25),
    "Auditory":          (0.769, 0.776, 0.784),
    "Black":             (0.00, 0.00, 0.00),
}

SEED_VERTICES_BY_SURFACE = {
    "fsavg": {
        "Default Mode":      {"Region": "Precuneus",                       "Vertex Number": 137},
        "Dorsal Attention":  {"Region": "Supramarginal/Superior Parietal", "Vertex Number": 1},
        "Frontoparietal":    {"Region": "Supramarginal/Inferior Parietal", "Vertex Number": 107},
        "Visual":            {"Region": "LOC",                             "Vertex Number": 82},
        "Somatomotor":       {"Region": "Postcentral",                     "Vertex Number": 0},
        "Ventral Attention": {"Region": "Insula",                          "Vertex Number": 59},
        "Limbic":            {"Region": "Middle Temporal Gyrus",           "Vertex Number": 112},
        "Auditory":          {"Region": "Superior Temporal Gyrus",         "Vertex Number": 18993},
    },
    "fslr": {
        "Default Mode":      {"Region": "Precuneus",                       "Vertex Number": 11},
        "Dorsal Attention":  {"Region": "Supramarginal/Superior Parietal", "Vertex Number": 6},
        "Frontoparietal":    {"Region": "Supramarginal/Inferior Parietal", "Vertex Number": 64},
        "Visual":            {"Region": "LOC",                             "Vertex Number": 7},
        "Somatomotor":       {"Region": "Postcentral",                     "Vertex Number": 275},
        "Ventral Attention": {"Region": "Insula",                          "Vertex Number": 271},
        "Limbic":            {"Region": "Middle Temporal Gyrus",           "Vertex Number": 290},
        "Auditory":          {"Region": "Superior Temporal Gyrus",         "Vertex Number": 1187},
    },
    "onavg": {
        "Default Mode":      {"Region": "Precuneus",                       "Vertex Number": 3216},
        "Dorsal Attention":  {"Region": "Supramarginal/Superior Parietal", "Vertex Number": 5803},
        "Frontoparietal":    {"Region": "Supramarginal/Inferior Parietal", "Vertex Number": 2969},
        "Visual":            {"Region": "LOC",                             "Vertex Number": 693},
        "Somatomotor":       {"Region": "Postcentral",                     "Vertex Number": 29},
        "Ventral Attention": {"Region": "Insula",                          "Vertex Number": 4679},
        "Limbic":            {"Region": "Middle Temporal Gyrus",           "Vertex Number": 5198},
        "Auditory":          {"Region": "Superior Temporal Gyrus",         "Vertex Number": 3415},
    },
}


# ═══════════════════════════════ Neighbours & boundaries ═══════════════════════════════
def pad_to_2d(nested_list, max_len, dtype=float, fill_value=-1):
    """Stack ragged rows into a 2D array, padding with fill_value (default -1)."""
    padded = np.full((len(nested_list), max_len), fill_value, dtype=dtype)
    for i, row in enumerate(nested_list):
        padded[i, :len(row)] = row
    return padded


def _build_neighbors(hemisphere, minimum_distance, nb_space, mask_lh, mask_rh):
    """Return a {global_cortex_idx: neighbor_array} dict for one hemisphere."""
    num_lh_vertices = int(mask_lh.sum())
    offset = num_lh_vertices if hemisphere == "r" else 0

    sls, _ = nb.sls(hemisphere, minimum_distance,
                    space=nb_space, center_space=nb_space, return_dists=True)

    N_dist   = max(len(sl) for sl in sls)
    sl_idx2d = pad_to_2d(sls, N_dist, dtype=int, fill_value=-1)

    nbrs = {}
    for i, row in enumerate(sl_idx2d):
        valid_indices   = row[row >= 0]
        valid_neighbors = valid_indices[valid_indices != i]
        nbrs[i + offset] = valid_neighbors + offset
    return nbrs


def get_boundary_vertices(assignments, neighbors_dict):
    """Label every cortical vertex as a boundary (1) or interior (0) vertex."""
    boundary = np.zeros(len(assignments), dtype=np.int8)
    for vtx, nbrs in neighbors_dict.items():
        if len(nbrs) == 0:
            continue
        if np.any(assignments[nbrs] != assignments[vtx]):
            boundary[vtx] = 1
    return boundary


def build_neighbors_adaptive(target, mask_lh, mask_rh):
    """Neighbours within MIN_DISTANCE mm; doubles the radius on coarse meshes where most
    vertices would otherwise have no neighbours (e.g. onavg-ico4)."""
    n_lh_ctx, n_rh_ctx = int(mask_lh.sum()), int(mask_rh.sum())
    d = MIN_DISTANCE
    while True:
        neighbors = {}
        for lr, n_expected in (('l', n_lh_ctx), ('r', n_rh_ctx)):
            nbrs_h = _build_neighbors(lr, d, target, mask_lh, mask_rh)
            assert len(nbrs_h) == n_expected, (
                f"{lr}h: {len(nbrs_h)} searchlights but {n_expected} cortex vertices")
            neighbors.update(nbrs_h)
        n_nbrs = np.array([len(v) for v in neighbors.values()])
        frac_none = (n_nbrs == 0).mean()
        if frac_none <= MAX_NO_NEIGHBOR_FRAC or d * 2 > MAX_DISTANCE:
            break
        print(f"    {100 * frac_none:.0f}% of vertices have no neighbours within {d} mm "
              f"→ trying {d * 2} mm")
        d *= 2
    max_idx = max((v.max() for v in neighbors.values() if len(v)), default=-1)
    assert max_idx < n_lh_ctx + n_rh_ctx, f"neighbour index {max_idx} out of range"
    print(f"    Neighbours within {d} mm: mean {n_nbrs.mean():.1f}, min {n_nbrs.min()}, "
          f"max {n_nbrs.max()} | vertices with none: {(n_nbrs == 0).sum()}")
    return neighbors, d


# ═══════════════════════════════ Network ordering helpers ═══════════════════════════════
def resolve_region_seeds(seed_vertices_by_space, surface_space):
    """Seeds for one surface family ('onavg', 'fsavg', 'fslr')."""
    return seed_vertices_by_space[surface_space]


def _build_discrete_colormap(color_dict, order, missing_color=(0.6, 0.6, 0.6)):
    colors = [color_dict.get(name, missing_color) for name in order]
    return ListedColormap(colors), colors, list(order)


def get_automatic_order(assignments, region_order_seeds, num_clusters,
                        X=None, networks_to_include=None):
    """Detect the best cluster → network mapping from seed vertices (from the notebook)."""
    assignments = np.asarray(assignments)
    candidates_pool = (list(region_order_seeds.keys()) if networks_to_include is None
                       else [n for n in networks_to_include if n in region_order_seeds])

    print("\n" + "=" * 64 + "\n  AUTOMATIC ORDERING  (data-driven, no fixed network list)\n" + "=" * 64)
    print(f"  Seeds scanned      : {candidates_pool}")
    print(f"  Expected clusters  : {num_clusters}\n")

    cluster_seed_map = defaultdict(list)
    for network_name in candidates_pool:
        seed_info   = region_order_seeds[network_name]
        vertex      = seed_info["Vertex Number"]
        region      = seed_info["Region"]
        cluster_num = int(assignments[vertex])
        cluster_seed_map[cluster_num].append((network_name, vertex, region))
        print(f"  seed  {network_name:<22s}  vertex {vertex:>6d} ({region:<35s})  → cluster {cluster_num:>2d}")
    print()

    centroids = {}
    if X is not None:
        for c in range(num_clusters):
            m = (assignments == c)
            if m.any():
                centroid = X[m].mean(axis=0)
                centroids[c] = centroid / (np.linalg.norm(centroid) + 1e-12)

    def _score(c, cands):
        scores = []
        for (name, vtx, region) in cands:
            seed_vec = X[vtx]
            scores.append((np.dot(centroids[c], seed_vec / (np.linalg.norm(seed_vec) + 1e-12)),
                           name, vtx, region))
        scores.sort(reverse=True)
        return scores

    cluster_to_network, dropped_networks = {}, []
    print("  ── Per-cluster resolution ──")
    for c in range(num_clusters):
        candidates = cluster_seed_map.get(c, [])
        if len(candidates) == 0:
            print(f"  cluster {c:>2d} | no seed → resolved later from dropped networks")
            continue
        if len(candidates) == 1:
            cluster_to_network[c] = candidates[0][0]
            print(f"  cluster {c:>2d} | unique seed: {candidates[0][0]}")
            continue
        if X is not None and c in centroids:
            scores = _score(c, candidates)
            winner = scores[0][1:]
            losers = [s[1:] for s in scores[1:]]
        else:
            winner, losers = candidates[0], candidates[1:]
        cluster_to_network[c] = winner[0]
        dropped_networks.extend(losers)
        print(f"  cluster {c:>2d} | {len(candidates)} seeds competing → winner={winner[0]}, "
              f"dropped={[l[0] for l in losers]}")

    unmatched = [c for c in range(num_clusters) if c not in cluster_to_network]
    for c in unmatched:
        if not dropped_networks:
            break
        if X is not None and c in centroids:
            best = _score(c, dropped_networks)[0][1:]
        else:
            best = dropped_networks[0]
        cluster_to_network[c] = best[0]
        dropped_networks = [d for d in dropped_networks if d[0] != best[0]]
        print(f"  cluster {c:>2d} | assigned dropped network: {best[0]}")

    for c in range(num_clusters):
        if c not in cluster_to_network:
            cluster_to_network[c] = f"Cluster_{c}"
            print(f"  cluster {c:>2d} | fallback label: Cluster_{c}")

    my_order = [cluster_to_network[c] for c in range(num_clusters)]
    print(f"\n  Final my_order: {my_order}\n" + "=" * 64 + "\n")
    return my_order, cluster_to_network



# ═══════════════════════════════ File loading (CIFTI + NIfTI + GIFTI) ═══════════════════════════════
LH_TOKENS = {'lh', 'l', 'left'}
RH_TOKENS = {'rh', 'r', 'right'}
CIFTI_CORTEX = {'CIFTI_STRUCTURE_CORTEX_LEFT': 'lh', 'CIFTI_STRUCTURE_CORTEX_RIGHT': 'rh'}
EXT_RE = re.compile(r'(\.(dlabel|dscalar|dtseries|pscalar|ptseries|label|func|shape))?\.(nii\.gz|nii|gii)$',
                    re.IGNORECASE)


def _split_name(filename):
    """'YA_L.label.gii' → ('lh', 'YA'); 'Gordon333.32k_fs_LR.dlabel.nii' → (None, 'Gordon333.32k_fs_LR').
    The extension (incl. dlabel/dscalar) is stripped, so dlabel/dscalar twins share a key."""
    stem   = EXT_RE.sub('', filename)
    tokens = re.split(r'[._-]', stem)
    for i, tok in enumerate(tokens):
        t = tok.lower()
        if t in LH_TOKENS or t in RH_TOKENS:
            return ('lh' if t in LH_TOKENS else 'rh'), '_'.join(tokens[:i] + tokens[i + 1:])
    return None, stem


def _file_priority(filename):
    """Lower = preferred when two files give the same parcellation."""
    f = filename.lower()
    if '.dlabel.' in f:
        return 0
    if f.endswith('.label.gii'):
        return 1
    if '.dscalar.' in f:
        return 2
    if f.endswith('.gii'):
        return 3
    return 4


def _load_cifti_cortex(img, fill_value=0):
    """{hemi: full-mesh array} for the cortical structures present in a CIFTI-2 file
    (medial wall = fill_value; subcortical/volume structures dropped)."""
    axes = [img.header.get_axis(i) for i in range(len(img.shape))]
    bm_dim = next(i for i, ax in enumerate(axes) if isinstance(ax, nib.cifti2.BrainModelAxis))
    bm = axes[bm_dim]

    data = np.asanyarray(img.dataobj)
    data = np.moveaxis(data, bm_dim, -1).reshape(-1, data.shape[bm_dim])
    if data.shape[0] > 1:
        print(f'    ⚠️  CIFTI has {data.shape[0]} maps — using the first one.')
    data = data[0]

    hemis, n_sub = {}, 0
    for struct, slc, model in bm.iter_structures():
        if struct in CIFTI_CORTEX:
            arr = np.full(bm.nvertices[struct], fill_value, dtype=float)
            arr[model.vertex] = data[slc]
            hemis[CIFTI_CORTEX[struct]] = arr
        else:
            n_sub += data[slc].size
    print(f'    CIFTI: {data.size} grayordinates → '
          + ', '.join(f'{h.upper()} {a.size}' for h, a in hemis.items())
          + f' surface vertices ({n_sub} subcortical/other grayordinates dropped)')
    return hemis


def _load_gii(path):
    return np.asarray(nib.load(path).darrays[0].data).ravel()


def load_parcellation_inputs(orig_dir, hint_space):
    """Load every .nii/.nii.gz/.gii file, pair hemispheres and detect each one's source space.
    Returns ({key: {'lh': arr, 'rh': arr}},       full-mesh source labels
             {key: {'lh': (priority, file), ...}},
             {key: source_space})."""
    entries = defaultdict(dict)                 # key → hemi → (priority, filename, array, is_cifti)

    for f in sorted(os.listdir(orig_dir)):
        if not EXT_RE.search(f):
            continue
        path = os.path.join(orig_dir, f)
        hemi_tok, key = _split_name(f)
        print(f'  Loading {f}' + (f'  [{hemi_tok}]' if hemi_tok else ''))
        is_cifti = False
        try:
            if f.lower().endswith('.gii'):
                arr = _load_gii(path)
                hemis = {hemi_tok: arr} if hemi_tok else split_combined(arr, f, hint_space)
            else:
                img = nib.load(path)
                if isinstance(img, nib.Cifti2Image):
                    is_cifti = True
                    hemis = _load_cifti_cortex(img)          # full cortical mesh per hemisphere
                    if hemi_tok and hemi_tok in hemis:       # e.g. YA_L.dlabel.nii
                        hemis = {hemi_tok: hemis[hemi_tok]}
                else:
                    arr = img.get_fdata().ravel()
                    hemis = {hemi_tok: arr} if hemi_tok else split_combined(arr, f, hint_space)
        except Exception as e:
            print(f'  ⚠️  {f}: could not load ({e}) — skipping.')
            continue

        for h, arr in hemis.items():
            prio = _file_priority(f)
            cur = entries[key].get(h)
            if cur is None or prio < cur[0]:
                if cur is not None:
                    print(f'    {f} [{h}] preferred over {cur[1]} (duplicate)')
                entries[key][h] = (prio, f, arr, is_cifti)
            else:
                print(f'    ⏭️  {f} [{h}]: duplicate of {cur[1]} — skipped')

    inputs, sources, spaces = {}, {}, {}
    print(f'\n  ── Source-space check (configured: {hint_space}) ──')
    for key, hs in entries.items():
        if set(hs) != set(HEMIS):
            print(f'  ⚠️  {key}: only found {sorted(hs)} — need both hemispheres. Skipping.')
            continue
        n_lh, n_rh = hs['lh'][2].size, hs['rh'][2].size
        is_cifti = hs['lh'][3] or hs['rh'][3]
        guess = guess_source_space(n_lh, n_rh, hint_space, is_cifti)
        if guess is None:
            print(f'  ⚠️  {key}: LH {n_lh} / RH {n_rh} vertices — unknown space '
                  f'(not in GUESS_SEPARATE). Skipping.')
            continue
        space, masked, flavor, how = guess
        if space == hint_space:
            verdict = 'matches configured space ✓'
        else:
            verdict = f'⚠️  configured {hint_space} — using detected {space}'
        print(f'  {key}: LH {n_lh} / RH {n_rh} vertices ({"CIFTI cortex" if is_cifti else "file"}) '
              f'→ {space}{" (masked" + (", legacy " + flavor if flavor else "") + ")" if masked else ""} '
              f'[{how}] — {verdict}')
        try:
            src_h = {h: (unmask_to_full(hs[h][2], h[0], space, flavor) if masked else hs[h][2])
                     for h in HEMIS}
        except Exception as e:
            print(f'  ⚠️  {key}: could not expand masked data to the full {space} mesh ({e}). Skipping.')
            continue
        inputs[key]  = src_h
        sources[key] = {h: hs[h][:2] for h in HEMIS}
        spaces[key]  = space
    return inputs, sources, spaces


def make_parc_name(parcellation, file_key):
    """Year-less folder name with the variant (resolution / age group) included:
    ('schaefer2018', 'Schaefer2018_400Parcels_7Networks_order')            → 'schaefer_400parcels_7networks'
    ('kong2022',     'Schaefer2018_400Parcels_Kong2022_17Networks_order')  → 'kong_400parcels_17networks'
    ('gordon2016',   'Gordon333.32k_fs_LR')                                → 'gordon_333'
    ('shen2013',     'shen268')                                            → 'shen_268'
    ('han2018',      'YA')                                                 → 'han_young_adult'"""
    if (parcellation, file_key) in VARIANT_NAMES:
        return VARIANT_NAMES[(parcellation, file_key)]
    base = re.sub(r'\d{4}$', '', parcellation).lower()
    low  = file_key.lower()
    tags = [n + ('parcels' if w.startswith('parcel') else 'networks')
            for n, w in re.findall(r'(\d+)_?(parcels?|networks?)', low)]

    if base == 'coleantecivic':
        # CAB-NP ships a parcel file and a network-assignment file → keep them apart
        kind = 'networks' if ('netassign' in low or 'network' in low) else 'parcels'
        return '_'.join([base, kind] + tags)

    if base == 'han':
        groups = [HAN_GROUPS[t.upper()] for t in re.split(r'[._\-\s]', file_key) if t.upper() in HAN_GROUPS]
        if not groups:
            print(f'    ⚠️  Han file "{file_key}" has no YA/ME/ML/OE/OL prefix — using its name.')
            groups = [re.sub(r'\W+', '_', low).strip('_')]
        return '_'.join([base, groups[0]] + tags)

    if not tags:
        nums = [n for n in re.findall(r'(?<!\d)(\d+)(?![\dk])', low)
                if not re.fullmatch(r'(19|20)\d\d', n)]
        tags = nums[:1]
    return '_'.join([base] + tags)


def name_inputs(parcellation, inputs, sources):
    """{key: name}, dropping keys whose name duplicates a preferred file (dlabel > dscalar ...)."""
    by_name = defaultdict(list)
    for key in inputs:
        prio = max(p for p, _ in sources[key].values())
        by_name[make_parc_name(parcellation, key)].append((prio, key))
    names = {}
    for name, cands in by_name.items():
        cands.sort()
        names[cands[0][1]] = name
        for prio, key in cands[1:]:
            if prio == cands[0][0]:
                alt = f'{name}_{key.lower()}'
                print(f'  ⚠️  {key} and {cands[0][1]} both map to "{name}" — saving as "{alt}"')
                names[key] = alt
            else:
                print(f'    ⏭️  {key} → "{name}" duplicates {cands[0][1]} (preferred) — skipped')
    return names


# ═══════════════════════════════ Resampling (per hemisphere) ═══════════════════════════════
def resample_hemi(labels_h, M_h):
    """Sparse one-hot (n_src, n_parcels) → probabilities (n_parcels, n_tgt) = onehot.T @ M_h."""
    labels = np.rint(labels_h).astype(int).ravel()
    assert labels.size == M_h.shape[0], f"{labels.size} labels vs {M_h.shape[0]} source vertices"
    parcel_ids = np.unique(labels[labels > 0])
    in_parc    = np.flatnonzero(labels > 0)
    cols       = np.searchsorted(parcel_ids, labels[in_parc])
    onehot = sp.csr_matrix((np.ones(in_parc.size, np.float64), (in_parc, cols)),
                           shape=(labels.size, parcel_ids.size))
    return (onehot.T @ M_h).tocsr(), parcel_ids


def hemi_id_lists(src_h, shared_frac=SHARED_FRAC, verbose=True):
    """Original parcel IDs per hemisphere; per-hemi label k ↔ ids[hemi][k-1].
    Bilateral labels (mostly shared IDs) keep their original numbering."""
    ids = {}
    for h in HEMIS:
        s = np.rint(src_h[h]).astype(int).ravel()
        ids[h] = np.unique(s[s > 0])
    n_shared = np.intersect1d(ids['lh'], ids['rh']).size
    n_small  = max(min(ids['lh'].size, ids['rh'].size), 1)
    if n_shared / n_small > shared_frac:
        full = np.arange(1, max(ids['lh'].max(initial=0), ids['rh'].max(initial=0)) + 1)
        if verbose:
            print(f'    Bilateral labels ({n_shared}/{n_small} IDs shared) — keeping original numbering 1–{full.size}')
        return {'lh': full, 'rh': full}, True
    if verbose:
        for h in HEMIS:
            print(f'    {h.upper()}: source IDs {ids[h].min()}–{ids[h].max()} ({ids[h].size}) → labels 1–{ids[h].size}'
                  + (f'  [{n_shared} ID(s) appear in both hemis]' if n_shared else ''))
    return ids, False


def hemi_prob_and_parc(probs_h, parcel_ids_h, ids_h):
    """Dense (K_h, nv_h) probabilities (row k-1 = label k) and WTA parc (1..K_h, -1 unassigned)."""
    nv_h   = probs_h.shape[1]
    prob_h = np.zeros((ids_h.size, nv_h), dtype=PROB_DTYPE)
    if parcel_ids_h.size and ids_h.size:
        idx   = np.searchsorted(parcel_ids_h, ids_h).clip(max=parcel_ids_h.size - 1)
        valid = parcel_ids_h[idx] == ids_h
        if valid.any():
            prob_h[valid] = probs_h[idx[valid]].toarray()
    assert prob_h.min(initial=0) >= 0, "prob map has negative values"
    if ids_h.size == 0:
        return prob_h, np.full(nv_h, UNASSIGNED_LABEL, dtype=int)
    parc = np.where(prob_h.max(axis=0) > 0, prob_h.argmax(axis=0) + 1, UNASSIGNED_LABEL)
    return prob_h, parc.astype(int)


def optimize_dtype(arr):
    try:
        return nb.utils.optimize_dtype(arr)
    except Exception:
        return arr.astype(np.int16 if np.abs(arr).max(initial=0) < 32767 else np.int32)


# ═══════════════════════════════ Saving with verification ═══════════════════════════════
class DiskFullError(RuntimeError):
    pass


def _is_disk_full(e):
    return isinstance(e, OSError) and e.errno in (errno.ENOSPC, errno.EDQUOT)


def _atomic_write(path, write_fn):
    """Write to <path>.tmp, then rename over <path> (overwrites the old file).
    A full disk raises DiskFullError and leaves no truncated file behind."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        if _is_disk_full(e):
            raise DiskFullError(f'Disk/quota full while writing {path}') from e
        raise


def save_npy_verified(path, arr):
    """Overwrite path with arr, then reload and confirm it matches before moving on."""
    def _w(tmp):
        with open(tmp, 'wb') as fh:
            np.save(fh, arr)
    _atomic_write(path, _w)
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise IOError(f'{path} was not written')
    back = np.load(path, mmap_mode='r')
    if back.shape != arr.shape or back.dtype != arr.dtype or not np.array_equal(back, arr):
        raise IOError(f'{path} does not match what was saved (shape {back.shape} vs {arr.shape})')


def save_csv_verified(path, df):
    _atomic_write(path, lambda tmp: df.to_csv(tmp, index=False))
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise IOError(f'{path} was not written')
    with open(path) as fh:
        n_lines = sum(1 for _ in fh) - 1
    if n_lines != len(df):
        raise IOError(f'{path}: {n_lines} rows on disk vs {len(df)} expected')


def check_file(path):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise IOError(f'{path} was not written')


_nb_check_disabled = False

def verify_with_neuroboros(name, target, expected):
    """Reload the official files through neuroboros' own loader (needs the updated spaces.py)."""
    global _nb_check_disabled
    if not VERIFY_WITH_NB or _nb_check_disabled:
        return
    try:
        from neuroboros.spaces import get_parcellation
        for h in HEMIS:
            parc = np.asarray(get_parcellation(name, h[0], space=target, prob=False))
            prob = np.asarray(get_parcellation(name, h[0], space=target, prob=True))
            if not np.array_equal(parc, expected[h]['parc']) or prob.shape != expected[h]['prob'].shape:
                raise IOError(f'neuroboros returned different data for {name} {h}')
        print(f'    ✓ neuroboros loads {name} ({target}) correctly')
    except (AssertionError, ValueError) as e:
        _nb_check_disabled = True
        print(f'    ⚠️  neuroboros check skipped ({type(e).__name__}: {e}). Install the updated '
              f'spaces.py to enable it; files were still verified directly.')
    except ImportError:
        _nb_check_disabled = True


# ═══════════════════════════════ Inventory of existing outputs ═══════════════════════════════
def _official_status(pdir):
    """'complete' if all 4 official files exist and load, else a short reason."""
    problems = []
    for h in HEMIS:
        for fn in (f'{RESAMPLE}_prob.npy', f'{RESAMPLE}_parc.npy'):
            path = os.path.join(pdir, h, GROUP_DIR, fn)
            if not os.path.isfile(path):
                problems.append(f'{h}/{fn[len(RESAMPLE) + 1:]} missing')
                continue
            try:
                np.load(path, mmap_mode='r').shape
            except Exception:
                problems.append(f'{h}/{fn[len(RESAMPLE) + 1:]} unreadable (truncated?)')
    return 'complete' if not problems else '; '.join(problems)


def log_inventory(target, expected_names=None, title='Existing parcellations'):
    """Print which functional parcellations exist for a target and whether they're complete.
    Returns {name: status}."""
    pdir = os.path.join(OUT_ROOT, target, 'parcellations')
    status = {}
    if os.path.isdir(pdir):
        for d in sorted(os.listdir(pdir)):
            if d not in ANATOMICAL and os.path.isdir(os.path.join(pdir, d)):
                status[d] = _official_status(os.path.join(pdir, d))
    n_ok = sum(v == 'complete' for v in status.values())
    print(f'\n  ── {title} [{target}]: {len(status)} found, {n_ok} complete ──')
    for name, st in status.items():
        print(f"    {'✓' if st == 'complete' else '✗'} {name:<42s} {st}")
    if expected_names:
        missing = sorted(set(expected_names) - set(status))
        if missing:
            print(f'    Not produced yet: {missing}')
    try:
        du = shutil.disk_usage(OUT_ROOT)
        print(f'    Free space on {OUT_ROOT}: {du.free / 1e9:.1f} GB '
              f'(home-directory quotas may be lower than this)')
    except Exception:
        pass
    return status


def folder_size_gb(path):
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total / 1e9


# ═══════════════════════════════ Target context & mapping cache ═══════════════════════════════
_xfm_cache = {}
_ctx_cache = {}


def get_xfm(src_space, target, hemi):
    """Unmasked per-hemisphere mapping: nb.mapping('l' or 'r', source, target)."""
    key = (src_space, target, hemi)
    if key not in _xfm_cache:
        _xfm_cache[key] = sp.csr_matrix(nb.mapping(hemi[0], source=src_space, target=target))
        print(f"  Mapping [{hemi[0]}] {src_space} → {target}: {_xfm_cache[key].shape}")
    return _xfm_cache[key]


def get_target_context(target, need_neighbors=True):
    """Masks and (optionally) boundary neighbours for a target space."""
    ctx = _ctx_cache.get(target)
    if ctx is not None and (ctx['neighbors'] is not None or not need_neighbors or ctx['nbr_failed']):
        return ctx
    if ctx is None:
        ctx = dict(target=target, mask_lh=None, mask_rh=None, full_mask=None,
                   neighbors=None, nbr_failed=False, distance=None)
        try:
            ctx['mask_lh'] = np.asarray(nb.mask('l', space=target, legacy=False), bool)
            ctx['mask_rh'] = np.asarray(nb.mask('r', space=target, legacy=False), bool)
            ctx['full_mask'] = np.concatenate([ctx['mask_lh'], ctx['mask_rh']])
            print(f"  Mask [{target}]: LH {ctx['mask_lh'].sum()}/{ctx['mask_lh'].size}, "
                  f"RH {ctx['mask_rh'].sum()}/{ctx['mask_rh'].size}")
        except Exception as e:
            print(f"  ⚠️  nb.mask failed for {target} ({e}) — boundaries and figures skipped.")
            ctx['nbr_failed'] = True
        _ctx_cache[target] = ctx
    if need_neighbors and ctx['full_mask'] is not None and ctx['neighbors'] is None:
        try:
            ctx['neighbors'], ctx['distance'] = build_neighbors_adaptive(
                target, ctx['mask_lh'], ctx['mask_rh'])
        except Exception as e:
            print(f"  ⚠️  Neighbour search failed for {target} ({e}) — boundaries skipped.")
            ctx['nbr_failed'] = True
    return ctx


# ═══════════════════════════════ Yeo network mapping ═══════════════════════════════
_network_map_cache = {}


def get_network_map(parcellation, file_key, src_h, src_space, ids):
    """{per-hemi label: network name}. Yeo-17: fixed. Yeo-7: seed-based automatic order,
    computed once at SEED_SPACE (where the seeds are defined) and reused for every target."""
    if parcellation not in NETWORK_PARCELLATIONS:
        return None
    key = (parcellation, file_key)
    if key in _network_map_cache:
        return _network_map_cache[key]

    max_label = max(ids['lh'].size, ids['rh'].size)
    net_map = None
    if 7 < max_label <= len(YEO17_ORDER):
        net_map = {k: YEO17_ORDER[k - 1] for k in range(1, max_label + 1)}
        print(f'    Yeo-17 fixed label → network mapping ({max_label} labels)')
    elif max_label <= len(YEO_NETWORKS):
        print(f'    Yeo-7: computing seed-based network order at {SEED_SPACE}')
        ctx = get_target_context(SEED_SPACE, need_neighbors=False)
        if ctx['full_mask'] is None:
            print('    ⚠️  No mask for seed space — no network colours.')
        else:
            parcs = []
            for h in HEMIS:
                probs_h, pids_h = resample_hemi(src_h[h], get_xfm(src_space, SEED_SPACE, h))
                parcs.append(hemi_prob_and_parc(probs_h, pids_h, ids[h])[1])
            labels_ctx = np.concatenate(parcs)[ctx['full_mask']]
            net_ids = np.unique(labels_ctx[labels_ctx > 0])
            clusters = np.full(labels_ctx.shape, -1, dtype=int)
            pos = labels_ctx > 0
            clusters[pos] = np.searchsorted(net_ids, labels_ctx[pos])

            seeds = resolve_region_seeds(SEED_VERTICES_BY_SURFACE, SEED_SPACE.split('-')[0])
            bad = [n for n in YEO_NETWORKS if seeds[n]['Vertex Number'] >= clusters.size]
            assert not bad, f"Seed vertices out of range for {SEED_SPACE}: {bad}"
            _, c2n = get_automatic_order(assignments=clusters, region_order_seeds=seeds,
                                         num_clusters=net_ids.size, X=None,
                                         networks_to_include=YEO_NETWORKS)
            net_map = {int(net_ids[c]): n for c, n in c2n.items()}
    else:
        print(f'    {max_label} labels — no network colour scheme, using watershed gradient.')

    _network_map_cache[key] = net_map
    return net_map


# ═══════════════════════════════ Plotting ═══════════════════════════════
def boundary_cmap(base_cmap, n_labels, boundary_color='black'):
    """Colors for values -1..n_labels: -1 → black, 0..n_labels → samples of base_cmap."""
    base   = mpl.colormaps.get_cmap(base_cmap)
    colors = base(np.linspace(0, 1, n_labels + 1))
    return ListedColormap(np.vstack([mpl.colors.to_rgba(boundary_color), colors]))


def safe_plot(values, fn, **kwargs):
    """nb.plot exactly as in the notebook (no space argument); a plotting error never stops the run."""
    try:
        nb.plot(values, fn=fn, **kwargs)
    except Exception as e:
        print(f"    ⚠️  Plot failed ({os.path.basename(fn)}): {e}")
    finally:
        plt.close('all')


def make_figures(name, target, parc_full, boundary, full_mask, net_map, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    fn_plain = os.path.join(fig_dir, f'{name}_{target}.png')
    fn_bnd   = os.path.join(fig_dir, f'{name}_{target}_boundary.png')
    for fn in (fn_plain, fn_bnd):                       # redo: drop old figures first
        if os.path.exists(fn):
            os.remove(fn)

    plot_ctx = parc_full[full_mask].copy()
    plot_ctx[plot_ctx < 0] = 0                          # unassigned → 0 for plotting

    if net_map:
        net_ids = np.unique(plot_ctx[plot_ctx > 0])
        K = net_ids.size
        colors = networks_yeo17 if any(n in networks_yeo17 for n in net_map.values()) else master_colors
        order  = [net_map.get(int(l), f'Label_{l}') for l in net_ids] + ['Black']
        cmap, _, _ = _build_discrete_colormap(colors, order)
        values = np.full(plot_ctx.shape, K, dtype=int)
        pos = plot_ctx > 0
        values[pos] = np.searchsorted(net_ids, plot_ctx[pos])
        values_bnd = values.copy()
        if boundary is not None:
            values_bnd[boundary] = K
        safe_plot(values, fn_plain, cmap=cmap, vmin=0, vmax=K,
                  title=f'{name} ({target})')
        if boundary is not None:
            safe_plot(values_bnd, fn_bnd, cmap=cmap, vmin=0, vmax=K,
                      title=f'{name} with boundaries ({target})')
    else:
        safe_plot(plot_ctx, fn_plain, cmap=WATERSHED_GRADIENT,
                  title=f'{name} ({target})')
        if boundary is not None:
            plot_bnd = plot_ctx.copy()
            plot_bnd[boundary] = -1
            n_max = int(max(plot_bnd.max(), 1))
            safe_plot(plot_bnd, fn_bnd, cmap=boundary_cmap(WATERSHED_GRADIENT, n_max),
                      vmin=-1, vmax=n_max, title=f'{name} with boundaries ({target})')




# ═══════════════════════════════ Per-file processing ═══════════════════════════════
def official_paths(target, name, hemi):
    d = os.path.join(OUT_ROOT, target, 'parcellations', name, hemi, GROUP_DIR)
    return d, os.path.join(d, f'{RESAMPLE}_prob.npy'), os.path.join(d, f'{RESAMPLE}_parc.npy')


def process_file(parcellation, file_key, name, src_h, src_files, src_space, target, skip_existing=False):
    print(f'\n  ── {name}  ({file_key}) ──')

    pdir = os.path.join(OUT_ROOT, target, 'parcellations', name)
    if os.path.isdir(pdir):
        st = _official_status(pdir)
        if skip_existing and st == 'complete':
            print('    Complete official files exist — skipping (--skip-existing).')
            return
        print(f'    Existing files ({st}) — overwriting.')

    M   = {h: get_xfm(src_space, target, h) for h in HEMIS}
    nv  = {h: M[h].shape[1] for h in HEMIS}
    ctx = get_target_context(target, need_neighbors=True)
    if ctx['full_mask'] is not None:
        for h in HEMIS:
            m = ctx['mask_lh'] if h == 'lh' else ctx['mask_rh']
            assert m.size == nv[h], f"{h} mask has {m.size} vertices, mapping has {nv[h]}"

    # ── Resample each hemisphere with its own mapping ──
    print(f'    Source space: {src_space}')
    ids, bilateral = hemi_id_lists(src_h)
    log_parcellation_guess(name, ids)
    out = {}
    for h in HEMIS:
        probs_h, pids_h = resample_hemi(src_h[h], M[h])
        prob_h, parc_h  = hemi_prob_and_parc(probs_h, pids_h, ids[h])
        out[h] = dict(prob=prob_h, parc=parc_h)
        present  = np.unique(parc_h[parc_h > 0])
        expected = np.searchsorted(ids[h], np.intersect1d(ids[h], pids_h)) + 1
        lost = np.setdiff1d(expected, present)
        print(f"    {h.upper()}: {prob_h.shape[0]} parcels → {present.size} after WTA | "
              f"{(parc_h == UNASSIGNED_LABEL).sum()} unassigned vertices"
              + (f" | ⚠️ lost labels: {lost.tolist()}" if lost.size else ""))

    # ── Official files (saved + verified before moving on) ──
    for h in HEMIS:
        _, f_prob, f_parc = official_paths(target, name, h)
        out[h]['parc'] = optimize_dtype(out[h]['parc'])
        save_npy_verified(f_prob, out[h]['prob'])
        save_npy_verified(f_parc, out[h]['parc'])
    print(f"    ✓ Saved + verified official files → {os.path.join(OUT_ROOT, target, 'parcellations', name)}")
    verify_with_neuroboros(name, target, out)

    # ── Boundaries (cortex-only index space) ──
    parc_full = np.concatenate([out['lh']['parc'], out['rh']['parc']]).astype(int)
    boundary  = None
    if ctx['neighbors'] is not None:
        boundary = get_boundary_vertices(parc_full[ctx['full_mask']], ctx['neighbors']).astype(bool)
        print(f"    {boundary.sum()} boundary vertices ({100 * boundary.mean():.1f}% of cortex, "
              f"{ctx['distance']} mm neighbours)")

    # ── Network map (Yeo) ──
    net_map = get_network_map(parcellation, file_key, src_h, src_space, ids)

    # ── "other" folder ──
    other_dir = os.path.join(OUT_ROOT, target, 'other', name)
    os.makedirs(other_dir, exist_ok=True)

    rows = []
    for h in HEMIS:
        s = np.rint(src_h[h]).astype(int).ravel()
        n_src = dict(zip(*np.unique(s[s > 0], return_counts=True)))
        p = out[h]['parc']
        n_tgt = dict(zip(*np.unique(p[p > 0], return_counts=True)))
        for k, oid in enumerate(ids[h], start=1):
            rows.append(dict(hemi=h, hemi_label=k, original_id=int(oid),
                             source_file=src_files[h][1],
                             n_source_vertices=int(n_src.get(oid, 0)),
                             n_target_vertices=int(n_tgt.get(k, 0)),
                             network=(net_map or {}).get(k, '')))
    save_csv_verified(os.path.join(other_dir, 'label_key.csv'), pd.DataFrame(rows))

    if net_map:
        fn_json = os.path.join(other_dir, 'network_order.json')
        with open(fn_json, 'w') as fh:
            json.dump({int(k): v for k, v in net_map.items()}, fh, indent=2)
        check_file(fn_json)
        save_csv_verified(os.path.join(other_dir, 'network_order.csv'),
                          pd.DataFrame(sorted(net_map.items()), columns=['hemi_label', 'network']))

    n_lh_ctx = int(ctx['mask_lh'].sum()) if ctx['mask_lh'] is not None else 0
    for h in HEMIS:
        d = os.path.join(other_dir, h, GROUP_DIR)
        parc_h = out[h]['parc'].astype(int)
        prob_h = out[h]['prob']
        mask_h = ctx['mask_lh'] if h == 'lh' else ctx['mask_rh']

        bnd_h = None
        if boundary is not None:
            bnd_ctx = boundary[:n_lh_ctx] if h == 'lh' else boundary[n_lh_ctx:]
            bnd_h = np.zeros(nv[h], dtype=bool)
            bnd_h[mask_h] = bnd_ctx
        parc_bnd_h = parc_h.copy()
        if bnd_h is not None:
            parc_bnd_h[bnd_h] = BOUNDARY_LABEL

        save_npy_verified(os.path.join(d, f'{RESAMPLE}_parc_noboundary.npy'), optimize_dtype(parc_h))
        if bnd_h is not None:
            save_npy_verified(os.path.join(d, f'{RESAMPLE}_parc_boundary.npy'), optimize_dtype(parc_bnd_h))

        orig_ids = np.full(nv[h], UNASSIGNED_LABEL)
        pos = parc_h > 0
        orig_ids[pos] = ids[h][parc_h[pos] - 1]
        df = pd.DataFrame({'hemi_vertex': np.arange(nv[h]),
                           'parc': parc_h,                 # 1..K_h, -1 unassigned
                           'original_id': orig_ids})       # ID in the source file, -1 unassigned
        if mask_h is not None:
            cortex_index = np.full(nv[h], -1)
            cortex_index[mask_h] = np.arange(mask_h.sum()) + (0 if h == 'lh' else n_lh_ctx)
            df.insert(1, 'in_cortex', mask_h)
            df.insert(2, 'cortex_index', cortex_index)   # index used by nb.plot / seed vertices
        if bnd_h is not None:
            df['boundary'] = bnd_h
            df['parc_boundary'] = parc_bnd_h             # -2 boundary, -1 unassigned
        if net_map:
            df['network'] = df['parc'].map(net_map)
        save_csv_verified(os.path.join(d, f'{RESAMPLE}_vertices.csv'), df)

        if SAVE_PROBS_CSV:
            r, c = np.nonzero(prob_h)
            save_csv_verified(os.path.join(d, f'{RESAMPLE}_probs.csv'),
                              pd.DataFrame({'hemi_label': r + 1, 'original_id': ids[h][r],
                                            'hemi_vertex': c, 'probability': prob_h[r, c]}))

    # ── Figures ──
    if MAKE_PLOTS and ctx['full_mask'] is not None:
        fig_dir = os.path.join(other_dir, 'figures')
        make_figures(name, target, parc_full, boundary, ctx['full_mask'], net_map, fig_dir)
        for suffix in ('', '_boundary') if boundary is not None else ('',):
            fn = os.path.join(fig_dir, f'{name}_{target}{suffix}.png')
            if not os.path.isfile(fn):
                print(f'    ⚠️  Figure not written: {os.path.basename(fn)}')
    print(f"    ✓ Saved + verified other files → {other_dir}")


# ═══════════════════════════════ Main ═══════════════════════════════
def main():
    global OUT_ROOT, PARC_ROOT, SAVE_PROBS_CSV, MAKE_PLOTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--targets', nargs='+', default=TARGETS)
    ap.add_argument('--parcellations', nargs='+', default=PARCELLATIONS)
    ap.add_argument('--parc-root', default=PARC_ROOT)
    ap.add_argument('--out-root', default=OUT_ROOT)
    ap.add_argument('--skip-existing', action='store_true',
                    help='keep parcellations whose official files are already complete (default: overwrite)')
    ap.add_argument('--no-plots', action='store_true')
    ap.add_argument('--detect-only', action='store_true',
                    help='only load the input files and report each one\'s detected source space')
    ap.add_argument('--no-probs-csv', action='store_true')
    args = ap.parse_args()

    PARC_ROOT, OUT_ROOT = args.parc_root, args.out_root
    MAKE_PLOTS     = not args.no_plots
    SAVE_PROBS_CSV = not args.no_probs_csv
    print(f'Input : {PARC_ROOT}\nOutput: {OUT_ROOT}\nTargets: {args.targets}\n'
          f'Parcellations: {args.parcellations}')
    if not os.path.isdir(OUT_ROOT):
        sys.exit(f'Output root {OUT_ROOT} does not exist.')

    inputs_cache = {}            # parcellation → {file_key: (name, {hemi: labels}, sources, src_space)}
    failures, saved = [], []

    for target in args.targets:
        t0 = time.time()
        print(f'\n{"#" * 78}\n#  TARGET: {target}\n{"#" * 78}')
        if not os.path.isdir(os.path.join(OUT_ROOT, target)):
            print(f'  ⚠️  {target} folder not in {OUT_ROOT} — it will be created.')
        expected = {v[0] for c in inputs_cache.values() for v in c.values()}
        if not args.detect_only:
            log_inventory(target, expected or None, title='Before processing')

        for parcellation in args.parcellations:
            print(f'\n{"=" * 70}\n  PARCELLATION: {parcellation}  →  {target}\n{"=" * 70}')
            orig_dir  = os.path.join(PARC_ROOT, parcellation, 'original_parcellations')
            hint_space = SOURCE_SPACES.get(parcellation, DEFAULT_SOURCE_SPACE)
            if not os.path.isdir(orig_dir):
                print(f'  ⚠️  No original_parcellations folder at {orig_dir} — skipping {parcellation}.')
                continue

            if parcellation not in inputs_cache:
                inputs, sources, spaces = load_parcellation_inputs(orig_dir, hint_space)
                names = name_inputs(parcellation, inputs, sources)
                inputs_cache[parcellation] = {k: (names[k], inputs[k], sources[k], spaces[k])
                                              for k in names}
                print()
                for k, (n, _, s, sp_) in inputs_cache[parcellation].items():
                    print(f'  {n:<40s} [{sp_}] ← ' + ' + '.join(sorted({s[h][1] for h in HEMIS})))
            if args.detect_only:
                continue
            files = inputs_cache[parcellation]
            if not files:
                print(f'  ⚠️  No usable files in {orig_dir} — skipping {parcellation}.')
                continue

            for file_key, (name, src_h, src_files, src_space) in files.items():
                try:
                    process_file(parcellation, file_key, name, src_h, src_files, src_space, target,
                                 skip_existing=args.skip_existing)
                    saved.append((target, name))
                except DiskFullError as e:
                    print(f'\n❌ {e}\n   Free up space (or use --no-probs-csv) and rerun; '
                          f'completed parcellations can be kept with --skip-existing.')
                    log_inventory(target, title='State when the disk filled up')
                    sys.exit(2)
                except Exception as e:
                    traceback.print_exc()
                    failures.append((target, name, 'process', str(e)))

        for key in [k for k in _xfm_cache if k[1] == target and target != SEED_SPACE]:
            del _xfm_cache[key]
        if args.detect_only:
            print('\n--detect-only: source spaces reported above; nothing was written.')
            return
        expected = {v[0] for c in inputs_cache.values() for v in c.values()}
        log_inventory(target, expected, title='After processing')
        tdir = os.path.join(OUT_ROOT, target)
        print(f'  Disk used: parcellations {folder_size_gb(os.path.join(tdir, "parcellations")):.2f} GB '
              f'(incl. anatomical), other {folder_size_gb(os.path.join(tdir, "other")):.2f} GB')
        print(f'\n  Target {target} finished in {(time.time() - t0) / 60:.1f} min')

    print(f'\n{"#" * 78}\nDone. {len(saved)} parcellation/target combination(s) processed or already present.')
    names = sorted({n for _, n in saved})
    print(f'Parcellation names: {names}')
    if failures:
        print(f'{len(failures)} failure(s):')
        for f in failures:
            print('  ', f)
        sys.exit(1)


if __name__ == '__main__':
    main()
