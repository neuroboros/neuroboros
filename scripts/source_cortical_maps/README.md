# Source files for resample_cortical_gradients.py

Fetched by `python ../resample_cortical_gradients.py --download` (fixed URLs and md5s in the script); not
redistributed with neuroboros (see `.gitignore`), since neither source carries a licence that clearly permits it. `resample_cortical_gradients.py` writes the resampled maps into
the core nb-data root (`<space>/gradients/...`), which `nb.gradient` reads.

- `margulies2016_fsLR32k/`: Margulies et al. 2016, *Situating the default-mode network along a principal gradient
  of macroscale cortical organization*, PNAS 113:12574-12579, doi:10.1073/pnas.1608282113. Gradients 1-10 on fsLR
  32k as distributed by neuromaps (Markello et al. 2022, Nat Methods): OSF project 4mw3a, file ids and md5
  checksums from neuromaps' `osf.json` (`neuromaps_osf.json`, fetched from netneurolab/neuromaps main); all 20
  files matched their checksums.
- `S-A_ArchetypalAxis/FSLRVertex/`: two files from https://github.com/PennLINC/S-A_ArchetypalAxis at commit edaad95
  (2023-12-26). Sydnor et al. 2021, *Neurodevelopment of the association cortices: Patterns, mechanisms, and
  implications for psychopathology*, Neuron 109:2820-2846, doi:10.1016/j.neuron.2021.06.016. Used:
  `FSLRVertex/SensorimotorAssociation_Axis_{LH,RH}.fslr32k.func.gii`. Its `FSLRVertex/G1.fMRI.dscalar.nii` equals
  Margulies gradient 1 (r = 1.000, left hemisphere checked).

Checks after resampling to onavg-ico32 (2026-11-12): vertex Spearman rho(G1, S-A axis) = 0.85; the S-A axis averaged
within Glasser-360 (210P) parcels vs Sydnor's published parcel ranks, rho = 0.96 (R_H has no covered vertices).
