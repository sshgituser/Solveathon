"""Prove the SerIQ PC engine runs on the GPU and only on the GPU.

Builds (or reuses) a batch-1 copy of the pinned RTMPose graph, opens it with
CUDA as the only execution provider, and validates from an ORT profile that
every graph node ran on CUDA. Optionally decodes an allowed development clip
and, when a private artefact is supplied, runs the frozen classifier on
``cuda:0`` via the same ``pose_sequence`` policy the product uses.

Writes a receipt outside the checkout. No raw video, no per-frame predictions,
no home paths, no human identifiers leave the machine.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.gpu import (CudaPose, CudaUnavailable, assert_cuda_only, assert_model_on_cuda,
                    cuda_stack, enable_cuda_dlls, open_session, prepare_static_graph,
                    sha256)


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True,
                              allow_nan=False) + '\n', encoding='utf8')
    tmp.replace(path)


def clip_frames(path, limit=64):
    """Decode up to *limit* evenly spaced frames of an allowed clip."""
    import cv2
    from ml.extract import phase_indices
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened():
            raise ValueError('clip cannot be opened')
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            raise ValueError('clip has no reliable frame count')
        cursor, out, frame = -1, [], None
        for target in phase_indices(n, repeat_short=False):
            while cursor < target:
                cursor += 1
                if cursor == target:
                    ok, frame = cap.read()
                else:
                    ok = cap.grab()
                if not ok:
                    raise ValueError('clip ended early')
            out.append(frame)
        return out
    finally:
        cap.release()


def _image_feed(frame):
    import cv2
    import numpy as np
    h, w = frame.shape[:2]
    scale = min(192 / w, 256 / h) / 1.25
    left, top = (192 - scale * w) / 2, (256 - scale * h) / 2
    transform = np.array([[scale, 0, left], [0, scale, top]], np.float32)
    image = cv2.warpAffine(frame, transform, (192, 256), flags=cv2.INTER_LINEAR)
    mean = np.asarray([123.675, 116.28, 103.53], np.float32)
    std = np.asarray([58.395, 57.12, 57.375], np.float32)
    rgb = (image[:, :, ::-1].astype(np.float32) - mean) / std
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def pose_parity(session, frames, reference_path):
    """Compare the strict graph against an archival reference of the original graph.

    The reference was captured once from the original export running on CUDA with
    its two shape-only nodes on CPU; it is stored with the input hash. Verification
    never re-executes a mixed graph: a missing or mismatched reference is an error.
    """
    import numpy as np
    data = [_image_feed(frame) for frame in frames]
    digest = hashlib.sha256(b''.join(d.tobytes() for d in data)).hexdigest()
    reference_path = Path(reference_path)
    if not reference_path.is_file():
        raise ValueError(f'archival reference is missing: {reference_path.name}; capture it before verification')
    with np.load(reference_path, allow_pickle=False) as saved:
        if str(saved['input_sha256']) != digest:
            raise ValueError('stored reference was built for different frames')
        reference = [saved['simcc_x'], saved['simcc_y']]

    deltas, same_bin, total, same_mask, mask_total = [], 0, 0, 0, 0
    for frame_data, ref_x, ref_y in zip(data, reference[0], reference[1]):
        cuda_x, cuda_y = session.run(None, {'input': frame_data})
        for cuda, ref in ((cuda_x[0], ref_x), (cuda_y[0], ref_y)):
            deltas.append(float(np.abs(cuda - ref).max()))
            same_bin += int((cuda.argmax(axis=-1) == ref.argmax(axis=-1)).sum())
            total += int(cuda.argmax(axis=-1).size)
        score_cuda = np.minimum(cuda_x.max(axis=-1), cuda_y.max(axis=-1))[0]
        score_ref = np.minimum(ref_x.max(axis=-1), ref_y.max(axis=-1))[0]
        same_mask += int(((score_cuda >= .50) == (score_ref >= .50)).sum())
        mask_total += int(score_cuda.size)
    return dict(frames=len(data), reference='archival-mixed-gpu',
                max_abs_delta=max(deltas) if deltas else None,
                simcc_bin_agreement=same_bin / total if total else None,
                visibility_agreement=same_mask / mask_total if mask_total else None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pose-model', required=True, type=Path,
                    help='pinned public RTMPose end2end.onnx')
    ap.add_argument('--graph', type=Path,
                    help='batch-1 strict graph; built next to --workdir when absent')
    ap.add_argument('--workdir', type=Path, required=True,
                    help='private directory outside the checkout for the built graph')
    ap.add_argument('--clip', type=Path, help='allowed development clip')
    ap.add_argument('--reference', type=Path,
                    help='archival reference (.npz) of the original graph; required with --clip')
    ap.add_argument('--weights', type=Path, help='private A/B safetensors')
    ap.add_argument('--slim-base', type=Path, help='pinned SLiM base for B')
    ap.add_argument('--expected-sha256', help='expected digest of --weights')
    ap.add_argument('--out', required=True, type=Path, help='receipt path outside the checkout')
    args = ap.parse_args(argv)

    workdir = args.workdir.resolve()
    out = args.out.resolve()
    if out.is_relative_to(ROOT) or workdir.is_relative_to(ROOT):
        ap.error('receipts and built graphs must live outside the checkout')

    enable_cuda_dlls()
    receipt = dict(schema='seriq-gpu-receipt-v1',
                   generated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                   stack=cuda_stack(),
                   pose=dict(original_sha256=sha256(args.pose_model)),
                   classifier=None, clip=None, verdict='failed')

    graph = args.graph
    if graph is None:
        graph = workdir / 'end2end.b1.onnx'
        if not graph.exists():
            receipt['pose']['transform'] = prepare_static_graph(args.pose_model, graph)
        else:
            receipt['pose']['transform'] = dict(reused=True)
    receipt['pose']['graph_sha256'] = sha256(graph)

    session = open_session(graph)
    import numpy as np
    shape = session.get_inputs()[0].shape
    feed_shape = [1 if isinstance(d, str) or d is None else d for d in shape]
    delegation = assert_cuda_only(session, {session.get_inputs()[0].name:
                                            np.zeros(feed_shape, np.float32)},
                                  label=graph.name)
    receipt['pose']['delegation'] = delegation
    receipt['pose']['input_shape'] = feed_shape
    receipt['pose']['outputs'] = [[o.name, o.shape] for o in session.get_outputs()]

    pose = CudaPose(session, graph_sha256=receipt['pose']['graph_sha256'])
    receipt['pose']['warmup'] = pose.warmup()

    if args.clip:
        if not args.reference:
            ap.error('--clip requires --reference with the archival pose reference')
        frames = clip_frames(args.clip)
        parity = pose_parity(session, frames, args.reference)
        xyz, masks = [], []
        for frame in frames:
            coords, observed = pose(frame)
            xyz.append(coords)
            masks.append(observed)
        xyz, masks = np.stack(xyz), np.stack(masks)
        from ml.data import FOREARM_POLICY, pose_sequence
        sequence = pose_sequence(xyz, masks, FOREARM_POLICY)
        receipt['clip'] = dict(frames=len(frames), sha256=sha256(args.clip),
                               name=args.clip.name, parity=parity,
                               pair_frames=list(sequence.pair_frames or (0, 0)),
                               reason=sequence.reason)

        if args.weights:
            import torch
            from ml.api import load_artifact
            predictor = load_artifact(args.weights, slim_base=args.slim_base,
                                      device='cuda:0', expected_sha256=args.expected_sha256)
            assert_model_on_cuda(predictor.model, 'cuda:0')
            result = predictor.predict_pose(xyz, masks)
            receipt['classifier'] = dict(
                sha256=sha256(args.weights), kind=predictor.metadata['kind'],
                device='cuda:0', parameter_devices=sorted(
                    {str(t.device) for t in predictor.model.parameters()}),
                result=result)
    elif args.weights:
        ap.error('--weights requires --clip to exercise the real pipeline')

    ok = (receipt['pose']['delegation'].get('non_cuda') == []
          and receipt['pose']['delegation'].get('by_provider', {}).get('CUDAExecutionProvider'))
    if receipt['clip'] and receipt['classifier']:
        ok = ok and receipt['classifier']['result'].get('label') in ('complete', 'incomplete')
    receipt['verdict'] = 'gpu_verified' if ok else 'failed'
    dump(out, receipt)
    print(json.dumps(dict(verdict=receipt['verdict'], pose_nodes=receipt['pose']['delegation']['nodes'],
                          clip=bool(receipt['clip']), classifier=bool(receipt['classifier'])),
                     ensure_ascii=False))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
