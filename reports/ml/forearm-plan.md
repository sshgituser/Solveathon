# Evaluate an observed 2D forearm without opening test prematurely

This is a living execution plan; historical strict-policy reports are unchanged.

## Purpose / Big Picture

Determine whether one consistently tracked 2D elbow–wrist pair can classify E02 when the shoulder is cropped. Train both paths locally, or retain an explicit block. The result is not an elbow-angle measurement or medical assessment.

## Progress

- [x] (2026-11-24 08:32Z) Read strict implementation and inherited aggregates; verify pinned local weights, cache membership, device and development video digests.
- [x] (2026-11-24 08:40Z) Verified fault tests RED then GREEN; full CUDA-enabled suite: 30 passed.
- [x] (2026-11-24 08:40Z) Trained both models, verified reload, passed predeclared validation gate and checked both artifact digests and the source-hashed freeze.
- [x] (2026-11-24 08:42Z) Ran the single sealed evaluation on 20 clips; checked exact cache membership and freeze → lock → aggregate ordering.

## Surprises & Discoveries

Of 2,220 train slots where both elbow–wrist pairs are labeled visible, every left-labelled pair overlaps the right-labelled pair: maximum corresponding-point distance is 0.061 projected right forearm lengths. These are compatible with duplicate labels, not evidence of two measured arms. The right-labelled pair is present in at least 40/64 slots in 76/77 train clips. All 97 dev files match manifest digests and have at least 64 native frames; inherited nearest-frame sampling did not repeat native frames because of clip length.

## Decision Log

On 2026-11-24, before fitting or inspecting forearm validation, use only detector right elbow/wrist for the full clip; no left fallback or swapping. Audit the left channel to detect a persistent separated rival, but do not treat it as a second observation. Subtract the first accepted elbow once per clip, scale once by the median projected forearm length to a nominal 0.30 unit, keep camera-plane orientation and displacements, and mask every other slot including depth.

Predeclared sealed admission: each validation class retains at least 8/10 clips. Each model must achieve research minima of 0.70 conditional balanced accuracy, 0.65 full-denominator balanced recall, 0.60 full-denominator recall in each class and strictly beat a train-fit mask-only control on full-denominator balanced recall. These are experiment admission gates, not clinical performance targets. Keep threshold and epoch selection rules otherwise unchanged. If either model fails, keep the results and do not open test or write a sealed freeze.

## Outcomes & Retrospective

Both learned artifacts passed validation and tied at 20/20 on the one-volunteer sealed test. No transfer advantage was measured; the simpler A is sufficient for the evaluated prototype. The test established a fixed-configuration sample result, not independent custody of labels or population accuracy. The fixed-side policy is not validated for other viewpoints or camera mirroring, and the observation itself cannot establish elbow extension without upper-arm orientation. See `reports/ml/forearm-v1/README.md` for commands, hypotheses, metrics and proof.

## Context and Orientation

`ml/data.py` maps 33 raw joints to common masked 64×25 coordinate arrays, `ml/models.py` consumes them, and `ml/api.py` serializes and selects preprocessing by pose policy. `ml/train.py` never opens test under `fit`. `reports/ml/coverage-*.json` records three previous strict-triplet blocks. The private work directory outside Git holds pose caches, the official RTMPose ONNX and SLiM base, and locally fitted parameters.

## Plan of Work

Add a named 2D right-pair policy with unit tests for partial/missing/contradictory observations, include policy metadata and API tests, then audit geometry and track coverage on development partitions. Train the scratch temporal network and the SLiM final encoder block with binary head using identical observable coordinates. Add an audit of what the mask alone can predict. Leave sealed test inaccessible without a successful, source- and weight-hashed freeze.

## Concrete Steps

Run the MSVC-bound `seriq pytest -q -p no:cacheprovider ml/tests` and `seriq ml.train fit ... --pose-policy seriq-e02-forearm-2d-v1` from the checkout root, as detailed in `ml/README.md` and the experiment report. Run `seriq ml.train test ...` only after successful freeze and validation.

## Validation and Acceptance

Fault cases abstain with explicit reasons and leave missing slots zero/masked, while a valid pair retains displacement and populates exactly two encoder joints. Strict legacy pose keeps its old gate. Compare held-out metrics and aggregate geometry, reload both artifact hashes, and prove the final pretrained encoder block and classifier head changed.

## Idempotence and Recovery

Use a separate private artifact directory and report subdirectory; never overwrite historical candidate reports. A failed validation run can be inspected without accessing test. A test attempt creates an exclusive private lock before video access; do not remove it to try again.

## Artifacts and Notes

Fixed manifest counts: 77 train, 20 validation, 20 sealed test, 10 quarantine. Validation and test each represent one volunteer. The pinned RTMPose digest is `5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c`; the pinned SLiM digest is `c25c70db9ca5c1ca3715b80acbd21851fa04f4b3178e4c82f96d45492774b5bf`.

## Interfaces and Dependencies

Keep `load_artifact(...).predict_pose(xyz, mask)` positional arguments unchanged. Extend metadata, add abstention diagnostics, and bind the private official SLiM file when loading B.

Final revision: recorded the completed fit, one-time evaluation, and limits of the experimental result without changing the frozen implementation or historical strict attempts.
