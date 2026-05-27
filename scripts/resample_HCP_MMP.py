"""
Resample HCP MMP parcellation to different target spaces.

The source data (Glasser et al., 2016; https://balsa.wustl.edu/study/show/RVVG) provides
probabilistic maps on the HCP 32k_fs_LR surface (fslr-ico57 in neuroboros): for each
parcel k (1-180) and each hemisphere, the value at a cortical vertex is the fraction of
the 210 subjects for whom that vertex was assigned to parcel k (steps of 1/210).
Medial-wall vertices are zero.

Resampling pipeline
-------------------
1. Load source maps into a (181, 64984) array ``parcels``, where row 0 is an all-
   zero background row, rows 1-180 hold parcel k for the left (cols 0-32491) then
   right (cols 32492-64983) hemisphere, and all medial-wall vertices are zero.

2. Obtain the overlap-area mapping matrix M of shape (64984, nv_target*2) via
   ``nb.mapping('lr', 'fslr-ico57', target)``.  Each column of M contains the
   fractional overlap weights from source vertices to a single target vertex, so
   multiplying ``parcels @ M`` propagates probability mass proportionally to the
   surface area shared between source and target vertices.

3. The resampled probability array ``prob`` has shape (181, nv_target*2).  For
   each target vertex the assigned parcel is determined by argmax over rows 0-180:
   argmax = 0 means no parcel dominates (unassigned → stored as -1), argmax = k
   means parcel k has the highest probability (stored as k, 1-indexed).

Groups
------
``Q1-Q6_RelatedParcellation210``  - parcellation discovery set (210 subjects).
``Q1-Q6_RelatedValidation210``    - parcellation validation set (210 subjects).
``Q1-Q6_Related420``              - combined estimate: mean of the two 210-subject
                                    probability maps, equivalent to a 420-subject
                                    average.

Saved files (per hemisphere, per group)
----------------------------------------
``overlap-8div_prob.npy``  - float64, shape (180, nv_target); row k-1 is the
                             resampled probability for parcel k (background row
                             excluded).
``overlap-8div_parc.npy``  - integer, shape (nv_target,); 1-indexed parcel label
                             or -1 for unassigned vertices.
"""

import os

import nibabel as nib
import numpy as np

import neuroboros as nb

NV57 = 32492  # vertices per hemisphere on 32k surface
GLASSER_BASE = "Glasser_et_al_2016_HCP_MMP1.0_v6_RVVG/HCP_PhaseTwo"


def load_mask(root=GLASSER_BASE, group="Q1-Q6_RelatedParcellation210"):
    """Boolean mask of shape (64984,): True for cortical vertices."""
    glasser = f"{root}/{group}/MNINonLinear/fsaverage_LR32k"
    fn = (
        f"{glasser}/{group}.MyelinMap_BC_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.dscalar.nii"
    )
    ax = nib.load(fn).header.get_axis(1)
    mask = np.zeros((2, NV57), dtype=bool)
    for i, hemi in enumerate(
        ["CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT"]
    ):
        _, _, bm = next(s for s in ax.iter_structures() if s[0] == hemi)
        mask[i, bm.vertex] = True
    return mask.ravel()


def load_parcels(mask, root=GLASSER_BASE, group="Q1-Q6_RelatedParcellation210"):
    """Parcellation probability maps of shape (181, 64984).

    Row 0 is zeros (background). Rows 1-180 hold parcel k for both hemispheres:
    L in columns 0-32491, R in columns 32492-64983. argmax(axis=0) gives 0
    (unassigned) or 1-180 (parcel label) for each vertex.
    """
    glasser = f"{root}/{group}/MNINonLinear/fsaverage_LR32k"
    n = len(mask)
    parcels = np.zeros((181, n), dtype=np.float64)
    for i, h in enumerate(["L", "R"]):
        fn = f"{glasser}/{group}.{h}.CorticalAreas_dil_Final_Final_Areas.32k_fs_LR.dscalar.nii"
        data = nib.load(fn).get_fdata()  # (180, n_cortical)
        hemi_mask = mask[i * NV57 : (i + 1) * NV57]
        parcels[1:, i * NV57 : (i + 1) * NV57][:, hemi_mask] = data
    return parcels


def save_parcellation(
    parcels,
    group="Q1-Q6_RelatedParcellation210",
    target="onavg-ico32",
    name="HCP_MMP",
    resample="overlap-8div",
):
    """Convert parcel probability maps to target space and save as prob/parc npy files."""
    data_root = nb.io.DATA_ROOT
    M = nb.mapping("lr", "fslr-ico57", target)  # (64984, nv_target*2)
    prob = parcels @ M  # (181, nv_target*2)
    nv_target = prob.shape[1] // 2

    for i, lr in enumerate(["l", "r"]):
        prob_h = prob[:, i * nv_target : (i + 1) * nv_target]  # (181, nv_target)
        raw = prob_h.argmax(axis=0)  # 0=unassigned, 1-180=parcel label
        parc = nb.utils.optimize_dtype(np.where(raw > 0, raw, -1))

        out_dir = os.path.join(
            data_root, "core", target, "parcellations", name, f"{lr}h", group,
        )  # fmt: skip
        prob_out = prob_h[1:]  # (180, nv_target), no background row
        assert prob_out.min() >= 0, "prob map has negative values"
        nb.save(os.path.join(out_dir, f"{resample}_prob.npy"), prob_out)
        nb.save(os.path.join(out_dir, f"{resample}_parc.npy"), parc)
        print(f"Saved {lr}h → {out_dir}")


def check_legacy(mask, parcels):
    legacy_idx = np.load("legacy/glasser_parcellation_indices.npy", allow_pickle=True)
    legacy_mask = np.concatenate(
        [np.load(f"legacy/mask_{lr}h_32k.npy") for lr in ["l", "r"]]
    )
    assert np.array_equal(mask, legacy_mask), "masks differ"

    mask2 = mask.reshape(2, NV57)
    n_match = 0
    for i in range(360):
        h = i // 180  # 0=L, 1=R
        cortical_vals = parcels[i % 180 + 1, h * NV57 : (h + 1) * NV57][mask2[h]]
        argmax_cifti = (29696 if h else 0) + cortical_vals.argmax()
        n_match += argmax_cifti in legacy_idx[i]
    print(f"argmax match: {n_match}/360")


if __name__ == "__main__":
    targets = [
        "onavg-ico4",
        "onavg-ico8",
        "onavg-ico16",
        "onavg-ico32",
        "onavg-ico48",
        "onavg-ico64",
        "onavg-ico128",
        "fsavg-ico32",
        "fsavg-ico64",
        "fsavg-ico128",
        "fslr-ico32",
        "fslr-ico64",
        "fslr-ico128",
    ]
    groups = ["Q1-Q6_RelatedParcellation210", "Q1-Q6_RelatedValidation210"]

    all_parcels = {}
    for group in groups:
        mask = load_mask(group=group)
        assert mask[:NV57].sum() == 29696
        assert mask[NV57:].sum() == 29716

        parcels = load_parcels(mask, group=group)
        print(f"parcels shape: {parcels.shape}  group: {group}")
        all_parcels[group] = parcels

        if group == "Q1-Q6_RelatedParcellation210":
            check_legacy(mask, parcels)

        for target in targets:
            save_parcellation(parcels, group=group, target=target)

    # Combined group: mean of the two 210-subject probability maps = 420-subject estimate
    combined_group = "Q1-Q6_Related420"
    combined_parcels = (
        all_parcels["Q1-Q6_RelatedParcellation210"]
        + all_parcels["Q1-Q6_RelatedValidation210"]
    ) * 0.5
    print(f"parcels shape: {combined_parcels.shape}  group: {combined_group}")
    for target in targets:
        save_parcellation(combined_parcels, group=combined_group, target=target)
