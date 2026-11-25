"""Assemble the private runtime bundle for a trained SerIQ artefact.

The historical ``reports/ml/forearm-v1/freeze.json`` is a receipt of the previous
machine and is never amended. A new runtime contract gets a new private bundle:
hashes of the pose graph, the classifier, the training manifest and the predicate
under which the artefact was produced.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.data import sha256, verify_training_inputs


def read_metadata(path):
    from safetensors import safe_open
    with safe_open(str(path), framework='pt') as handle:
        return json.loads((handle.metadata() or {}).get('seriq', '{}'))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifacts', required=True, type=Path, help='directory with the trained safetensors')
    ap.add_argument('--name', required=True, help='artifact file name, e.g. A-tcn.safetensors')
    ap.add_argument('--pose-model', required=True, type=Path)
    ap.add_argument('--graph', required=True, type=Path, help='batch-1 strict pose graph')
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--slot', default='scratch', choices=('scratch', 'transfer'),
                    help='API slot this artefact serves; only scratch/A is accepted today')
    ap.add_argument('--source-commit', default=None,
                    help='40-hex commit the artefact was produced from (default: git HEAD)')
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(argv)

    if Path(args.name).name != args.name or (args.slot == 'scratch' and args.name != 'A-tcn.safetensors'):
        ap.error('use a bare artifact filename; scratch requires A-tcn.safetensors')
    artifact = args.artifacts / args.name
    meta = read_metadata(artifact)
    commit = args.source_commit
    if not commit:
        import subprocess
        try:
            commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], check=True,
                                    capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            raise SystemExit('--source-commit is required outside a git checkout')
    if not re.fullmatch('[0-9a-f]{40}', commit or ''):
        raise SystemExit('--source-commit must be a 40-character hex commit id')
    if args.slot == 'scratch' and (meta.get('kind') != 'A' or meta.get('variant') != 'A'):
        raise SystemExit('scratch slot is the standalone A artefact; refusing another kind')
    if args.slot == 'transfer' and not meta.get('slim_base_sha256'):
        raise SystemExit('transfer slot requires a SLiM-based artefact')
    original, graph, manifest = (sha256(p) for p in (args.pose_model, args.graph, args.manifest))
    verify_training_inputs(meta, manifest=manifest, original=original, graph=graph)
    bundle = dict(
        schema='seriq-bundle-v1',
        created_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        source_commit=commit,
        model=dict(id=meta.get('kind'), variant=meta.get('variant'), file=args.name, slot=args.slot,
                   sha256=sha256(artifact),
                   threshold=meta.get('threshold'), version=meta.get('version'),
                   base_sha256=meta.get('slim_base_sha256'), corpus_scope=meta.get('corpus_scope'),
                   pose_policy=meta.get('pose_policy'), device='cuda:0'),
        pose=dict(engine=meta.get('pose_engine'), policy=meta.get('pose_policy'),
                  original_sha256=original, export_sha256=graph),
        runtime=dict(device_class='nvidia-cuda', version='onnxruntime-1.27.0+torch-2.11.0+cu130'),
        training_manifest_sha256=manifest,
        artifact_bytes=artifact.stat().st_size,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
                        encoding='utf8')
    print(json.dumps(dict(bundle=args.out.name, model=bundle['model']['sha256'],
                          pose=bundle['pose']['export_sha256']), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
