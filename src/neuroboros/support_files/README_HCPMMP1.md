# HCP-MMP1.0 area names and groupings

These tables are used by `nb.parcellation("HCP_MMP", lr, networks=...)`, `nb.parcellation_names(...)` and
`nb.parcellation_colors(...)`. The
parcels themselves are in nb-data (`<space>/parcellations/HCP_MMP/{lh,rh}/<group>/`).

| file | content |
|---|---|
| `atlas-HCPMMP1_keylabel.tsv` | `key` 1-180 = the parcel value in `nb.parcellation("HCP_MMP", ...)` (same in both hemispheres); `label` (area name, e.g. `V1`); `long_name`; `glasser22_key` / `glasser22_label` (the 22 cortices). |
| `atlas-HCPMMP1_ColeAnticevic_keylabel.tsv` | `hemi` (L/R), `key` 1-180, `label` (`L_V1`, ...), `network_key` 1-12, `network_label`. Assigned per hemisphere: 13 areas differ between L and R. |
| `atlas-ColeAnticevic_networks.tsv` | `network_key`, `network_label`, and the ColeLab colour (`r`, `g`, `b`, 0-255). |
| `atlas-HCPMMP1_colors.tsv` | `hemi`, `key`, `label`, `r`, `g`, `b` (0-255): the label-table colours of the published HCP-MMP1.0 dlabel (BALSA RVVG `Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii`). L and R colours differ for every area. |
| `atlas-HCPMMP1_Glasser22_colors.tsv` | `glasser22_key` 1-22, `glasser22_label`, `r`, `g`, `b` (0-255): the region colours of the 22 cortices. |
| `LICENSE_ColeAnticevicNetPartition.txt` | The CAB-NP licence (non-commercial), which must accompany redistributed CAB-NP data. |

## Sources

- **Area order and names:** these are the map names of the BALSA RVVG probability files
  (`Q1-Q6_Related*.{L,R}.CorticalAreas_dil_Final_Final_Areas.32k_fs_LR.dscalar.nii`). Map k is area k, and the
  neuroboros labels are the argmax of those maps after resampling (checked for 210P and 210V, 2026-11-03).
  `long_name` comes from a community `HCP-MMP1_UniqueRegionList.csv` whose origin is not recorded.
- **Glasser22:** the 22 cortices are the primary (bold) text sections of Glasser et al. 2016, *Supplementary
  Neuroanatomical Results*, Supplementary Table 1. The grouping comes from the same `HCP-MMP1_UniqueRegionList.csv`
  (`Cortex_ID`). It matches the bold section for 168/180 areas; the other 12 could not be machine-parsed because
  their table cells wrap.
- **Glasser22 colours:** sampled (2026-11-12) from the numbered swatches under Figure 1 of the *Supplementary
  Neuroanatomical Results* (page 2, embedded 1950x1176 image), median of the swatch pixels after excluding the
  digit. Every sampled channel was within 2 of a multiple of 32 (or 255/160), i.e. JPEG noise around a designed
  palette, so the stored values are snapped to those levels. The figure's own data are at http://balsa.wustl.edu/QnXj.
- **ColeAnticevic:** Cole-Anticevic Brain-wide Network Partition v1.1.6
  (https://github.com/ColeLab/ColeAnticevicNetPartition), from `cortex_parcel_network_assignments.txt`,
  `network_labelfile.txt` and the LabelKey. ColeLab's README (v1.1.3) says the Q1-Q6_RelatedValidation210 Glasser
  version was used. Networks are assigned per parcel, so they apply to any HCP_MMP group.

## Citations

- Glasser MF, Coalson TS, Robinson EC, et al. (2016). A multi-modal parcellation of human cerebral cortex.
  *Nature* 536, 171-178. https://doi.org/10.1038/nature18933
- Ji JL, Spronk M, Kulkarni K, Repovš G, Anticevic A, Cole MW (2019). Mapping the human brain's cortical-subcortical
  functional network organization. *NeuroImage* 185, 35-57. https://doi.org/10.1016/j.neuroimage.2018.10.006

HCP-derived data are subject to the WU-Minn HCP Consortium Open Access Data Use Terms (see nb-data's
`parcellations/HCP_MMP/README.md`).
