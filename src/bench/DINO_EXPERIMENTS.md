# Frozen image features: 66 rejected variants

The [official DINOv2 ViT-S/14 backbone](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8)
was used locally as a frozen feature extractor. It was not fine-tuned on the
competition. These experiments did not beat the corresponding v6/v7 checkpoint,
so the default remains [macro v7](../MACRO_V7.md).

The fixed 2,221-frame plan supplies two ball-centred crops, 128 and 320 pixels,
resized to 224. Fused ball geometry locates the crop, with the existing unlabelled
tracking fallback when needed. Each crop produces a 384-dimensional class token.
A second pass averages 4x4 patch tokens around the ball and above it, using the
projected ball's position inside the clamped crop. Extraction reads no GT.

The second pass reproduces every original class-token value exactly. Every
requested frame is present, and all patch pools are finite. Cache manifests
record the source revision, model hash, frame plan, camera parameters and ball
fingerprints. Original weights have SHA-256
`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.

The classifier combines the 34 previous context/video features with the frozen
embedding. PCA keeps 16 or 32 components and whitens them; imputation, PCA and
logistic fitting use only that head's training rows. A second screen standardises
all visual dimensions with stronger logistic regularisation, without PCA.
Every head uses the existing 301 native rows and 125-frame purge. Candidate
experts, primary replacements and auxiliary mixtures are independently decoded
per held-out head before public stitching. Before each screen, this decoder
must reproduce production exactly over all 199,319 frames.

| Screen | Baseline | Variants | Best macroF1 / spellF1 |
|---|---|---:|---:|
| Class tokens, PCA | v6 | 18 | 0.957088 / 0.875000 |
| Tokens above the ball, PCA | v7 | 18 | 0.957604 / 0.875000 |
| Both local patch pools, PCA | v7 | 18 | 0.957604 / 0.875000 |
| Class tokens, all dimensions | v7 | 6 | 0.955165 / 0.837500 |
| Tokens above ball, all dimensions | v7 | 6 | 0.956020 / 0.760000 |

The tied variants add no public benefit. Some change unscored match frames,
which supplies no evidence for promotion. A separate nested calibration screen
fits three purged inner folds inside each outer dead-head training set. Sigmoid
and isotonic variants also regress; the best reaches 0.954006 / 0.823529.
All selection reuses the same 17 public possessions. These results do not measure
private or unseen-match performance.

## Reproduce

Prepare the provided assets and macro-v7 caches as documented in `src/README.md`.
The model checkout is a separate repository under the ignored cache directory:

```bash
git clone https://github.com/facebookresearch/dinov2.git cache/models/dinov2_repo
git -C cache/models/dinov2_repo checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=3 .venv-cv/bin/python src/bench/dino_extract.py
OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=3 .venv-cv/bin/python src/bench/dino_extract.py \
  --patch-pools --out cache/improve_nov17/dino_tokens_5hz.npz
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/dino_bench.py \
  --params src/configs/macro_v6.json --out cache/improve_nov17/dino_bench
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 .venv/bin/python src/bench/dino_bench.py \
  --params src/configs/macro_v7.json --cache cache/improve_nov17/dino_tokens_5hz.npz \
  --pool above --out cache/improve_nov17/dino_above_v7
```

Use `--pool pools` for both local pools. The full-dimension screens use
`--components 0 --C .03 --crops 128 320`, with class or above-ball features.
Extraction resumes only when metadata matches. The model, embeddings, crops
and predicted labels remain ignored. Aggregate results and fingerprints are
in `results/macro_v7_visual_2026-11-17.json`.
