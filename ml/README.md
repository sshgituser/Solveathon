# SerIQ E02 ML

Local research prototype, not a clinical assessment. The task is binary clip classification (`incomplete`, `complete`) given the pose trajectory of one E02 clip. Missing pose produces `label=None` with a reason, not an incomplete classification. See the [forearm-only results](../reports/ml/forearm-v1/experiment.md) and [evaluation notes](../reports/ml/forearm-v1/README.md). The historical strict-triplet attempts remain in `reports/ml/`; fixture parameters are not trained artifacts.

## Environment

Run commands from the checkout root. Point `$uv` to your `uv.exe`, `$private` to an absolute directory **outside** the checkout; `$root` is the existing dataset's `2_Extending the Elbow` directory. Use Git Bash on Windows:

```bash
export UV_CACHE_DIR="$private/uv-cache"
"$uv" venv --python 3.11 "$private/.venv"
py="$private/.venv/Scripts/python.exe"
"$uv" pip install --python "$py" -r ml/requirements.txt
"$uv" pip install --python "$py" --index https://download.pytorch.org/whl/cu126 \
  --index-strategy unsafe-best-match 'torch==2.8.0+cu126'
```

The CUDA wheel may need Microsoft's Visual C++ runtime. On a Windows machine where system installation is not allowed, extract the official Microsoft VCLibs x64 Desktop package to a private directory and register its DLL directory *in this process only* with `os.add_dll_directory`. For example, use this Bash function instead of plain `python` once `msvc` is set to that directory:

```bash
seriq() {
  "$py" -Bu -c 'import os,sys,runpy; h=os.add_dll_directory(sys.argv.pop(1));runpy.run_module(sys.argv.pop(1),run_name="__main__")' "$msvc" "$@"
}
seriq ml.train --help
```

Use [`ATTRIBUTION.md`](../artifacts/seriq/ATTRIBUTION.md) to obtain the exact official pose and SLiM files into `$private`. Set `$pose` to the appropriate private model file (e.g. RTMPose's extracted `end2end.onnx`) and `$slim` to the pinned SLiM XSub `.safetensors`. Never check these upstream model files into Git. MoveNet is an optional diagnostic: requires a working LiteRT or TensorFlow Lite interpreter for the official TFLite file.

## Data and strict-triplet diagnostics

The local manifest contains identifiable inventory keys; keep it private (`ml/manifest.json` is ignored even if a legacy local copy exists). To prepare a fresh inventory outside the checkout, run:

```bash
"$py" -B -c 'import sys; from ml.data import build_manifest,verify_counts,dump_json; rows=build_manifest(sys.argv[1]);verify_counts(rows);dump_json(sys.argv[2],rows)' \
  "$root" "$private/manifest-rebuilt.json"
sha256sum "$private/manifest-rebuilt.json"
```

Expected `127 = 77 train + 20 validation + 20 sealed test + 10 quarantine`. The two held-out splits each contain one distinct volunteer. Ten conflicting `Complete/Train` files are quarantined, not renamed. Content SHA-256, path and participant code serve inventory/provenance only. RGB, filenames, clip IDs, FPS and duration are not model features. The models use observed pose sampled at up to 64 phase positions; neither uses RGB embeddings or pose imputation.

```bash
# The first evidence is TRAIN ONLY; repeat for candidate extractors using --engine.
"$py" -B -m ml.extract smoke --root "$root" --private "$private" --engine rtmpose \
  --manifest "$private/manifest-rebuilt.json" --pose-model "$pose" --out "$private/smoke-rtmpose.json"
# Expand only to development partitions; decoded frame pixels are never written.
"$py" -B -m ml.extract dev --root "$root" --private "$private" --engine rtmpose \
  --manifest "$private/manifest-rebuilt.json" --pose-model "$pose" --out "$private/coverage-rtmpose.json"
```

The default `seriq-e02-arm-triplet-v1` still requires at least 20/64 valid shoulder–elbow–wrist triplets on one side with nonzero limb lengths. All three strict attempts were blocked; do not lower the threshold or clip coordinates to the image boundary to make that policy pass. Strict and forearm results are not interchangeable.

## Forearm-only (`seriq-e02-forearm-2d-v1`)

The selected extractor is the pinned RTMPose Body7 ONNX. Both models see only the x/y of the detector's right elbow/wrist in a common image-plane scale and a paired validity mask. `z=0` is padding, not a 3D observation; shoulder, elbow angle and other coordinates are absent. Samples are never synthesized to fill a gap or copied from the left. The first usable elbow anchors the whole track and the median projected forearm length sets one scale for the whole clip; translation over time and camera-plane orientation are kept. The policy checks 40/64 paired observations, six in every quarter, relative segment length, motion continuity, nontrivial motion and separated rival tracks. A real second visible arm can make the single-arm prototype abstain. No equivalence is claimed for mirrored or other views.

The [evaluation notes](../reports/ml/forearm-v1/README.md) give the CUDA invocation, sealed handling and actual hashes. Both checkpoints and upstream weights stay private. Validation selection uses balanced accuracy, macro F1, BCE, then earlier epoch, with thresholds selected on validation only. The test requires a successful common freeze and an exclusive private lock. Do not re-run it after the lock is created.

## Python API (when trained artifacts exist)

```python
from ml.api import load_artifact

model_a = load_artifact(private_artifacts / 'A-tcn.safetensors', expected_sha256=frozen_a_sha)
model_b = load_artifact(private_artifacts / 'B-slim-last-block.safetensors',
                        slim_base=private_slim_file, expected_sha256=frozen_b_sha)
# xyz [T,33,3] with x right, y up, z away and one common scale;
# observed [T,33] Boolean from the SAME extractor/version as training.
result = model_a.predict_pose(xyz, observed)
```

Check `model.metadata` for `pose_policy`, `pose_engine`, `pose_model_sha256`, `threshold`, `input_channels`, `pose_source_side`, `pose_dimensions`, `phase_sampling`, `confidence_kind` and `low_confidence_below`. For a video adapter call `extract_one(path, extractor, repeat_short=False)` with this policy, preserve RTMPose x-right/y-up coordinates in common scale and pass its boolean mask. Do not preprocess or pre-gate with `pose_sequence` for `predict_pose`: it already dispatches to the artifact policy. For fewer than 64 native samples, pass each one once; repeated-slot input cannot be reliably detected from coordinates alone.

`label`, `p_complete` and `reason` are unchanged. In forearm mode `valid_arm_frames` stays `[0, 0]` because no triplets/angles are used. `valid_pair_frames` contains raw co-observed counts `[left, right]`; `usable_pair_frames` counts retained right observations for a scorable clip. If `reason` is set, all of `label`, `p_complete` and `confidence` are `None`. Reasons: `no_visible_forearm_pair`, `insufficient_visible_forearm_pair`, `degenerate_forearm_pair`, `inconsistent_forearm_trajectory`, `ambiguous_forearm_tracks`, `incomplete_forearm_coverage`, or `static_forearm_trajectory`; a file adapter can report `decode_failure` independently of the pose-only API. On successful classification, `confidence` is a normalized margin from the decision threshold `t`: `(p-t)/(1-t)` for `p>=t`, `(t-p)/t` otherwise. If it falls below `metadata['low_confidence_below']` (`0.10`), show a neutral low-confidence state instead of trusting the hard label. This margin and `p_complete` are uncalibrated, not probabilities of exercise correctness. B contains just the adapted last block plus head, overlaid on the SHA-256-verified full encoder.

## Tests

```bash
export SERIQ_SLIM_BASE="$slim"
export SERIQ_SLIM_SRC="$private/upstream-slim"  # exact code revision in attribution
export SERIQ_CHECK_DEVICE=cuda
seriq pytest -q -p no:cacheprovider ml/tests --basetemp "$private/pytest-seriq"
```

Unit tests for checkpoint fidelity use synthetic poses without reading the sealed clips. The parity test compares all 205 tokens of the full SLiM encoder with the upstream graph and an explicit dense attention reference while preserving RoPE. Test fixtures are never packaged as learned weights.
