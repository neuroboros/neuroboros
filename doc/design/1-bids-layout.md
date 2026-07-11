# Problem statement

This is primarily a write up for myself and @cindyhfls on current state and potential improvement.

There is an issue of scalability with current layout which neuroboros expects. ATM it is a hierarchy of files which is not grouped by subj/session but by other criteria. E.g. we have for a relatively small budapest hotel dataset

```shell
[d31548v@ndoli 20.2.7]$ pwd
/dartfs/rc/lab/H/HaxbyLab/feilong/nb-data/budapest/20.2.7

[d31548v@ndoli 20.2.7]$ ls
anatomy  confounds  resampled  xforms
```

and in `confounds/` we have  1323 files for 21 subject

```shell
[d31548v@ndoli 20.2.7]$ ls confounds/| nl | tail -n 1
  1323	sub-sid000560_ses-budapest_task-movie_run-5_desc-mask_timeseries.npy

[d31548v@ndoli 20.2.7]$ ls confounds/| sed -e 's,_ses-.*,,g' | sort | uniq -c | nl | tail -n1
    21	     63 sub-sid000560
```

so it would be logical to expect about 50 times more (60,000) for a dataset with e.g. 1000 subjects of the similar kind.  Similarish (just 3 times less) situation with the folders with split per area data

```shell
[d31548v@ndoli 20.2.7]$ ls resampled/onavg-ico64/l-cerebrum/1step_pial_overlap/*npy | nl | tail -n 1
   441	resampled/onavg-ico64/l-cerebrum/1step_pial_overlap/sub-sid000560_ses-budapest_task-movie_run-05.npy
```
so it would grow with the number of subjects.

# Proposed solution (WiP)


For that reason, BIDS decided that most logical top level grouping should be subject/session since then it would not matter and scale for any sized study.

So let's consider what filename patterns we have across folders neuroboros expects per each subject:

```shell
[d31548v@ndoli 20.2.7]$ for d in *; do find $d  -iname *sid000021* | head -n1 ; done
anatomy/onavg-ico64/overlap/area.mid/sid000021_rh.npy
confounds/sub-sid000021_ses-1_task-visualmemory_run-6_desc-mask_timeseries.npy
resampled/onavg-ico64/l-cerebrum/1step_pial_overlap/sub-sid000021_ses-budapest_task-movie_run-04.npy
xforms/onavg-ico64/sid000021_area_lh.npz
```

- we can use ["BIDS common derivatives"](https://bids-specification.readthedocs.io/en/stable/derivatives/introduction.html) to structure at the highest level
- I feel like BIDS already has entities to cover most if not all necessary annotations above
  - `onavg-ico64` - space - `_space-onavg+ico64` and then have `spaces.tsv` and .json to describe them
  - `l-cerebrum` - `_seg-l+cereburm` https://bids-specification.readthedocs.io/en/stable/derivatives/imaging.html#discrete-segmentations
  -  `1step_pial_overlap` -
  - ... to be complete... suggest via comments and suggestion more there
