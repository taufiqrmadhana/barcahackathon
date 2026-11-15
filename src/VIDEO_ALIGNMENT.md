# Video cadence and aligned crop experiments

The current verified checkpoint is [macro v4](MACRO_V4.md): 0.950003 macroF1 and
0.752941 spellF1 on the public five-fold predictions. Its spell alternative remains
0.942279 / 0.800000. These are selected on the same 17 public possessions; private
and unseen-match scores are unknown.

## What changed in the input comparison

The physical pipeline uses `fused_full` ball tracking, while the historical FILM
features use raw ATD ball coordinates to centre their crops. Among the 6,431 public
frames with both coordinates, median displacement is 0.526 m and the 95th percentile
is 4.559 m. Displacement exceeds 0.2 m on 5,217 public frames. The fused track also
fills 229 public frames missing from raw tracking. No possession labels enter the
crop extraction.

The new comparisons keep the same native 1 Hz training grid, 301 training rows,
five time folds, 125-frame purge, physical baseline and independent fold decoding.
Raw/fused 1 Hz caches use identical 1,021 sample indices: 301 public and 720 elsewhere.
Changing crop location alone gives a small gain, but that view still underperforms
the denser existing checkpoint.

| Video head, by itself | macroF1 | spellF1 |
|---|---:|---:|
| Raw crop, 1 Hz, C=1 | 0.938077 | 0.626667 |
| Fused crop, 1 Hz, C=1 | 0.938782 | 0.626667 |
| Fused crop, 1 Hz, C=3 | 0.940064 | 0.626667 |
| Raw crop, 10 Hz, C=1 | 0.946477 | 0.776471 |
| Raw crop, 10 Hz, C=3 | 0.946829 | 0.741176 |

The 10 Hz cache preserves every previous 5 Hz feature value exactly and adds 1,500
observations, for 3,721 samples total. All crop mean residuals are finite. Its best
25% ordinary mixture with the current primary reaches only 0.948759 macroF1.
Using the C=3 view as another conditional team expert after the motion expert gives
0.949663 / 0.752941, a gain of roughly one public frame; it is not promoted.

The matched raw/fused denser comparison uses exactly the existing 1,739-index dense
plan (1,019 public observations, 301 training samples). Fused C=3 alone scores only
0.945989 / 0.637500. As a conditional H/A expert after the existing motion expert,
it reaches 0.950003 / 0.752941 and improves macroF1 on 3/4/5/6 time-fold partitions.
This selective expert is deployed; ordinary mixtures are not.

## Reproducible extraction and provenance

`extract_aligned_video.py` accepts a frame plan and an explicit raw or fused crop
source. It reads tracking, camera and video inputs, without loading possession
labels. Use a separate cache alias for each crop source:

```bash
TFHUB_CACHE_DIR=cache/film_model TF_NUM_INTRAOP_THREADS=6 TF_NUM_INTEROP_THREADS=2 \
  OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=3 .venv-v3/bin/python src/extract_aligned_video.py \
  --frames src/cv/frames/valdi_film_dense.txt --cache-kind film_fused_dense \
  --ball-source fused_full --chunk 8
```

The new sidecar is `<feature-cache>.npz.json`. It records raw-ball and crop-ball
hashes, crop source, cache kind and legacy feature parameters. Appending undocumented
existing rows or mixing crop sources is rejected. Historical raw caches retain
their existing path without a sidecar.

Fitting records the aligned cache's full file hash in the checkpoint and rejects a
change during fitting. Inference checks coordinates, parameters, metadata and cached
observations against that snapshot. Different crop sources can be valid ensemble
members; their held-out heads must still use identical time partitions and the
same physical baseline. Older models and caches preserve their exact prediction path.

All 41 tests pass, including changed crop coordinates, changed raw input, mismatched
parameters, altered cached observations, removed metadata and legacy compatibility.
The previous macro-v3 production CSV retains its exact byte fingerprint.
Models, videos, feature stores and prediction CSVs remain gitignored.
