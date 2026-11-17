"""
Resample published cortical gradients from fsLR 32k (fslr-ico57 in neuroboros) to onavg spaces.

Sources (downloaded into scripts/source_cortical_maps/ by --download, see its README.md):

- ``Margulies2016``: the first 10 functional-connectivity gradients of Margulies et al. 2016 (PNAS 113:12574),
  as distributed by neuromaps (source 'margulies2016', desc 'fcgradient01'..'fcgradient10', fsLR 32k, OSF
  project 4mw3a; md5 checked against neuromaps' osf.json). Gradient 1 is identical (r = 1.000) to
  ``G1.fMRI.dscalar.nii`` in the Sydnor repository.
- ``SA_axis``: the vertex-level sensorimotor-association axis of Sydnor et al. 2021 (Neuron 109:2820),
  PennLINC/S-A_ArchetypalAxis (commit edaad95), FSLRVertex/SensorimotorAssociation_Axis_{LH,RH}.fslr32k.func.gii.
  Values are ranks 1-59412 over the cortical vertices of both hemispheres (1 = most sensorimotor).

Both sources set the HCP medial-wall vertices (2796 L, 2776 R) to 0; those are treated as missing.

Resampling: M = nb.mapping(lr, 'fslr-ico57', space) holds the area-overlap weights from source to target
vertices (each column sums to 1). With v the source cortex mask, the target value is the overlap-weighted mean
over cortical source vertices, (x * v) @ M / (v @ M). Target vertices whose cortical coverage v @ M is below
MIN_COVERAGE are set to NaN. Resampled S-A ranks are therefore area-weighted mean ranks, not re-ranked.

Saved files (core nb-data root, read by nb.gradient):
    <space>/gradients/Margulies2016/{lh,rh}/overlap-8div.npy   float32, (10, nv): row k-1 = gradient k
    <space>/gradients/SA_axis/{lh,rh}/overlap-8div.npy         float32, (nv,)

The source files are not redistributed with neuroboros (the Sydnor repository has no licence, and no licence is
listed for the neuromaps copy of the Margulies gradients). --download fetches them into source_cortical_maps/
from their original locations and checks every file against a fixed md5 (DOWNLOADS below); files already present
are only re-checked.

    python resample_cortical_gradients.py --download                 # fetch sources, resample to onavg-ico32
    python resample_cortical_gradients.py                            # sources already present
    python resample_cortical_gradients.py onavg-ico64 onavg-ico16    # other spaces
"""
import argparse
import hashlib
import os
import urllib.request

import nibabel as nib
import numpy as np

import neuroboros as nb

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source_cortical_maps")
NV57 = 32492
HCP_MEDIAL_WALL = {"l": 2796, "r": 2776}  # 32492 minus the 29696 / 29716 CIFTI cortex vertices
MIN_COVERAGE = 0.5  # judgment call: a target vertex needs half its area on fsLR cortex to get a value

SYDNOR_COMMIT = "edaad95a9c2eb3b5c44567907b8d4343d357651d"  # PennLINC/S-A_ArchetypalAxis, 2023-12-26
MARGULIES_OSF = {  # (desc, hemi): (OSF file id in project 4mw3a, md5), from neuromaps' osf.json (2026-11-12)
    ("fcgradient01", "L"): ("60c22687f3ce9401fa24dc77", "f3ef7c4601a4e8f44cafc59afb8aaa30"),
    ("fcgradient01", "R"): ("60c2267df3ce94020124c6f6", "463b8be657529800fd41ea3fc418c363"),
    ("fcgradient02", "L"): ("60c226f918f70b020ba7caac", "837c8e7aa719495b0fb5a07645b23983"),
    ("fcgradient02", "R"): ("60c226ee18f70b020aa7a77e", "1b1334761da62401f5519753500a49b0"),
    ("fcgradient03", "L"): ("60c226b5da0ff201fd393ea0", "f6f3264ff311c4d629ffdb7d499f7a76"),
    ("fcgradient03", "R"): ("60c226bf18f70b020ba7c982", "946d6c41a5cc46c7391323778fb580db"),
    ("fcgradient04", "L"): ("60c226c918f70b0205a7bbab", "19b3a5d41e991edfbaf5ba6639670170"),
    ("fcgradient04", "R"): ("60c226d318f70b020aa7a744", "1d675f6bacb94dc2d5b14f0381837cf0"),
    ("fcgradient05", "L"): ("60c2270cda0ff201f8397977", "608d60276eebccafd47ab6277be9601c"),
    ("fcgradient05", "R"): ("60c22702f3ce94020124c985", "7029a8ba95fb7df0f9c6d88afed3d842"),
    ("fcgradient06", "L"): ("60c22690f3ce9401ee24eae3", "4335bbd73eed786bbdcd5888dca9edce"),
    ("fcgradient06", "R"): ("60c2269bf3ce94020124c79f", "cf0e2e94ba7a29e11277aa63990dfc89"),
    ("fcgradient07", "L"): ("60c226a2da0ff201f3396029", "d2bbcc4162774cf8ec8fb5355f72a3bf"),
    ("fcgradient07", "R"): ("60c226adf3ce9401ee24eaec", "895187a70f8e2a8b194be91f0b1590a5"),
    ("fcgradient08", "L"): ("60c2271e18f70b020aa7a82c", "f357c7145a4363c087ebadaa4b92bfbb"),
    ("fcgradient08", "R"): ("60c22715da0ff201f2393840", "4a835ce339d010064beddcb89fec43b5"),
    ("fcgradient09", "L"): ("60c226db18f70b0206a7f136", "cab550d9eea2e1636eb6974bb4689271"),
    ("fcgradient09", "R"): ("60c226e618f70b0206a7f152", "d539c68735f42c2ce70eda0dfb0c714c"),
    ("fcgradient10", "L"): ("60c22731da0ff201fd394037", "755b9d61162091dbaa5a15de3dec6c02"),
    ("fcgradient10", "R"): ("60c22726f3ce94020124ca4a", "fcb889621c85b6871ac275d3d0d2773c"),
}
SYDNOR_MD5 = {"L": "b2a2f6f96e5efb947136bfa1359e4e61", "R": "ad6be5c851e89baa56c4fddd44b3a2dd"}


def margulies_fn(k, hemi):
    return os.path.join(SRC, "margulies2016_fsLR32k",
                        f"source-margulies2016_desc-fcgradient{k:02d}_space-fsLR_den-32k_hemi-{hemi}_feature.func.gii")


def sydnor_fn(hemi):
    return os.path.join(SRC, "S-A_ArchetypalAxis", "FSLRVertex",
                        f"SensorimotorAssociation_Axis_{hemi}H.fslr32k.func.gii")


def downloads():
    """[(url, local path, md5)] for every source file."""
    out = [(f"https://files.osf.io/v1/resources/4mw3a/providers/osfstorage/{fid}",
            margulies_fn(int(desc[-2:]), hemi), md5) for (desc, hemi), (fid, md5) in MARGULIES_OSF.items()]
    out += [(f"https://raw.githubusercontent.com/PennLINC/S-A_ArchetypalAxis/{SYDNOR_COMMIT}/FSLRVertex/"
             f"SensorimotorAssociation_Axis_{hemi}H.fslr32k.func.gii", sydnor_fn(hemi), md5)
            for hemi, md5 in SYDNOR_MD5.items()]
    return out


def md5sum(fn):
    with open(fn, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def download_sources():
    for url, fn, md5 in downloads():
        if not os.path.exists(fn):
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            tmp = fn + ".part"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, fn)
            print(f"downloaded {os.path.relpath(fn, SRC)}")
        if md5sum(fn) != md5:
            raise SystemExit(f"{fn}: md5 {md5sum(fn)} != expected {md5}; delete it and rerun --download")
    print(f"all {len(downloads())} source files present and checked")


def load_gifti(fn):
    data = nib.load(fn).darrays[0].data.astype(np.float64)
    assert data.shape == (NV57,), (fn, data.shape)
    return data


def margulies(lr):
    return np.stack([load_gifti(margulies_fn(k, lr.upper())) for k in range(1, 11)])


def sa_axis(lr):
    return load_gifti(sydnor_fn(lr.upper()))


def resample(x, lr, space):
    """x: (..., 32492) with medial wall 0 -> (..., nv_target) overlap-weighted mean over cortex."""
    x = np.atleast_2d(x)
    valid = np.all(x != 0, axis=0)
    # the two sources share the HCP 32k medial wall (2796 L / 2776 R vertices); anything else would mean a
    # changed source file
    assert (~valid).sum() == HCP_MEDIAL_WALL[lr], (lr, (~valid).sum())
    M = nb.mapping(lr, "fslr-ico57", space)
    coverage = np.asarray(valid.astype(float) @ M).ravel()
    out = np.asarray((x * valid) @ M) / np.where(coverage > 0, coverage, 1)
    out[:, coverage < MIN_COVERAGE] = np.nan
    return out.astype(np.float32), coverage


def main(spaces):
    root = nb.spaces.core_dataset.root
    for space in spaces:
        for name, loader in [("Margulies2016", margulies), ("SA_axis", sa_axis)]:
            for lr in "lr":
                out, coverage = resample(loader(lr), lr, space)
                if name == "SA_axis":
                    out = out[0]
                fn = os.path.join(root, space, "gradients", name, f"{lr}h", "overlap-8div.npy")
                os.makedirs(os.path.dirname(fn), exist_ok=True)
                np.save(fn, out)
                mask = nb.mask(lr, space=space).astype(bool)
                n_nan_in_mask = int(np.isnan(np.atleast_2d(out)[0][mask]).sum())
                print(f"{space} {name} {lr}h -> {fn} shape {out.shape}; "
                      f"NaN in nb.mask cortex: {n_nan_in_mask}/{mask.sum()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spaces", nargs="*", default=["onavg-ico32"])
    ap.add_argument("--download", action="store_true", help="fetch and md5-check the source files first")
    a = ap.parse_args()
    if a.download:
        download_sources()
    missing = [fn for _, fn, _ in downloads() if not os.path.exists(fn)]
    if missing:
        raise SystemExit(f"{len(missing)} source files missing (e.g. {missing[0]}); run with --download")
    main(a.spaces)
