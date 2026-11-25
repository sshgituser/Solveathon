"""Export an already locally trained E02/A checkpoint to static FP32 ONNX on CUDA."""
from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

from tools.android_from_source import CHANNELS, disk, read_bundle, read_json, write_json


def export(bundle_path: Path, artifact: Path, out: Path) -> None:
    import onnx
    import torch

    from ml.api import load_artifact
    from ml.gpu import assert_model_on_cuda

    # Port of the private export-a-onnx.py; keep the exact forearm math and CUDA trace.
    class CompactA(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.conv = model.conv
            self.head = model.head

        def forward(self, features):
            valid = features[:, :, 4:5] == 1
            xy = features[:, :, :4].masked_fill(~valid, 0)
            mask = valid.to(xy.dtype)
            enc = self.conv(torch.cat((xy, mask), dim=-1).transpose(1, 2)).transpose(1, 2)
            count = mask.sum(dim=1)
            mean = (enc * mask).sum(dim=1) / count.clamp_min(1)
            peak = enc.masked_fill(~valid, -1e4).amax(dim=1).masked_fill(count == 0, 0)
            span = (xy.masked_fill(~valid, -1e4).amax(dim=1) -
                    xy.masked_fill(~valid, 1e4).amin(dim=1)).masked_fill(count == 0, 0)
            return self.head(torch.cat((mean, peak, span), dim=1)).squeeze(-1)

    receipt = out.with_suffix(out.suffix + '.json')
    if out.suffix != '.onnx' or out.exists() or receipt.exists():
        raise ValueError('choose a new .onnx output with no existing sidecar')
    target = out.resolve()
    if target.is_relative_to(Path(__file__).resolve().parents[1]) or any((p / '.git').exists() for p in target.parents):
        raise ValueError('the ONNX and its receipt must stay outside any Git checkout')
    bundle_pin = disk(bundle_path)
    bundle = read_bundle(bundle_path)
    record = bundle['model']
    artifact_pin = disk(artifact)
    if artifact.name != record['file'] or artifact_pin != {
        'bytes': bundle['artifact_bytes'], 'sha256': record['sha256'],
    }:
        raise ValueError('checkpoint differs from the local training bundle')
    # A limited diagnostic fit may also report exposure_complete: A has no B/C families.
    # Require the full published E02 development split and its actual training result.
    training_path = artifact.parent / 'receipt.json'
    training_pin = disk(training_path)
    training = read_json(training_path)
    if not (training.get('schema') == 'seriq-train-gpu-receipt-v1'
            and training.get('status') == 'trained' and training.get('variant') == 'A'
            and training.get('sha256') == artifact_pin['sha256'] and training.get('artifact') == artifact.name
            and training.get('manifest_sha256') == bundle['training_manifest_sha256']
            == '8c73b2a35584639f3a2ea057878d39cf105c52ffa4d77a4a41639c2606134931'
            and training.get('manifest_counts') == {'train:0': 47, 'train:1': 30, 'validation:0': 10, 'validation:1': 10}
            and training.get('extraction', {}).get('clips') == 97
            and training.get('train', {}).get('n') == 77 and training.get('validation', {}).get('n') == 20
            and training.get('training', {}).get('device') == 'cuda:0'):
        raise ValueError('Android production export requires a complete local E02/A development run')
    model = load_artifact(artifact, device='cuda:0', expected_sha256=record['sha256'])
    metadata = model.metadata
    expected = {
        'kind': 'A', 'variant': 'A', 'version': record['version'],
        'threshold': record['threshold'], 'pose_policy': record['pose_policy'],
        'runtime_device': 'cuda:0', 'pose_graph_sha256': bundle['pose']['export_sha256'],
        'manifest_sha256': bundle['training_manifest_sha256'], 'exposure_complete': True,
    }
    if any(metadata.get(k) != v for k, v in expected.items()) or metadata.get('slim_base_sha256'):
        raise ValueError('checkpoint provenance/threshold differs from the local training bundle')
    net = CompactA(model.model).eval()
    assert_model_on_cuda(net, 'cuda:0')
    if any(t.is_floating_point() and t.dtype != torch.float32 for t in net.state_dict().values()):
        raise ValueError('expected FP32 weights')
    sample = torch.zeros((1, 64, 5), device='cuda:0', dtype=torch.float32)
    sample[:, :, 4] = 1
    stream = BytesIO()
    with torch.no_grad():
        torch.onnx.export(net, (sample,), stream, input_names=['features'], output_names=['logit'],
                          opset_version=20, dynamo=False)
    data = stream.getvalue()
    model_proto = onnx.load_model_from_string(data)
    onnx.checker.check_model(model_proto)
    graph = model_proto.graph
    for tensors, name, shape in ((graph.input, 'features', [1, 64, 5]), (graph.output, 'logit', [1])):
        if (len(tensors) != 1 or tensors[0].name != name or
                tensors[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT or
                [d.dim_value for d in tensors[0].type.tensor_type.shape.dim] != shape):
            raise ValueError(f'wrong exported FP32 ONNX tensor: {name}')
    if not graph.node or graph.node[-1].op_type != 'Squeeze' or list(graph.node[-1].output) != ['logit']:
        raise ValueError('classifier export changed; expected trailing logit Squeeze')
    if disk(artifact) != artifact_pin or disk(bundle_path) != bundle_pin or disk(training_path) != training_pin:
        raise ValueError('checkpoint or bundle changed during export')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as stream:
        stream.write(data)
    write_json(receipt, {
        'schema': 'qolda-android-local-a-onnx-v1', 'graph': disk(out), 'bundle': bundle_pin,
        'artifact': artifact_pin, 'training_receipt': training_pin, 'channels': CHANNELS, 'device': str(sample.device),
        'nodes': len(graph.node), 'torch': str(torch.__version__), 'onnx': onnx.__version__,
    })
    print(out)
    print(receipt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--artifact', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    export(args.bundle, args.artifact, args.out)


if __name__ == '__main__':
    main()
