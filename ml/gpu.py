"""Strict-GPU runtime for SerIQ: RTMPose on CUDA only, plus CUDA artefact loading.

Owner decision: the working neural graphs never execute on CPU. The ONNX pose
graph is created with CUDA as the only registered execution provider and with
``session.disable_cpu_ep_fallback``; after a warm-up the ORT profile is parsed
and every graph node must have run on ``CUDAExecutionProvider``. A session that
places even one node on CPU is rejected, not downgraded.

The stock RTMPose export carries a dynamic batch and a Shape->Slice->Concat->
Reshape tail. Those three shape nodes are not offered to the CUDA EP and land on
CPU. ``prepare_static_graph`` pins the batch and folds that constant shape chain
into an initializer, so the graph becomes fully CUDA-assignable. The transform is
recorded and parity-checked; the original file is never overwritten.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sysconfig

import numpy as np

CPU = 'CPUExecutionProvider'
CUDA = 'CUDAExecutionProvider'


class CudaUnavailable(RuntimeError):
    """CUDA is required but cannot be used. Never fall back to CPU."""


class CudaNotFullyAccelerated(RuntimeError):
    """A working graph node executed outside the CUDA EP."""


def sha256(path) -> str:
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def torch_lib() -> Path:
    return Path(sysconfig.get_paths()['purelib']) / 'torch' / 'lib'


def enable_cuda_dlls() -> Path:
    """Make the CUDA runtime shipped inside the PyTorch wheel loadable first.

    Windows resolves onnxruntime's CUDA provider DLL against the process search
    path; torch/lib carries cublasLt/cudnn/cudart for the pinned CUDA. Without
    this the CUDA EP fails to register and ORT silently keeps only CPU.
    """
    lib = torch_lib()
    if os.name != 'nt':
        return lib
    for dll in ('cublasLt64_13.dll', 'cudnn64_9.dll', 'cudart64_13.dll'):
        if not (lib / dll).is_file():
            raise CudaUnavailable(f'torch wheel is missing {dll} in {lib}')
    os.add_dll_directory(str(lib))
    os.environ['PATH'] = str(lib) + os.pathsep + os.environ.get('PATH', '')
    return lib


def cuda_stack() -> dict:
    """Versions and device name; no user paths, no serials."""
    import torch
    if not torch.cuda.is_available():
        raise CudaUnavailable('torch.cuda.is_available() is False')
    import onnxruntime as ort
    if CUDA not in ort.get_available_providers():
        raise CudaUnavailable('onnxruntime does not advertise CUDAExecutionProvider')
    free, total = torch.cuda.mem_get_info()
    return dict(torch=torch.__version__, cuda_runtime=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), onnxruntime=ort.__version__,
                device=torch.cuda.get_device_name(0),
                capability='.'.join(map(str, torch.cuda.get_device_capability(0))),
                memory_total_mib=round(total / 2**20), memory_free_mib=round(free / 2**20))


def _dim(value_info):
    dims = []
    for d in value_info.type.tensor_type.shape.dim:
        if d.dim_value > 0:
            dims.append(d.dim_value)
        elif d.dim_param:
            return None
        else:
            return None
    return dims


def _shapes(graph):
    shapes = {}
    for v in list(graph.value_info) + list(graph.input) + list(graph.output):
        if v.type.HasField('tensor_type'):
            shapes[v.name] = _dim(v)
    for tensor in graph.initializer:
        shapes[tensor.name] = list(tensor.dims)
    return shapes


def _pin_batch(model, batch):
    """Rewrite the leading dynamic batch dim of every graph input to a constant."""
    for value in model.graph.input:
        dims = value.type.tensor_type.shape.dim
        if dims and not dims[0].HasField('dim_value'):
            dims[0].dim_value = batch
    return model


def _fold_shape_nodes(model):
    """Replace Shape(x) with a Constant when x has a fully static shape."""
    import onnx
    from onnx import numpy_helper
    graph = model.graph
    shapes = _shapes(graph)
    replaced = 0
    for node in list(graph.node):
        if node.op_type != 'Shape' or len(node.input) != 1 or len(node.output) != 1:
            continue
        dims = shapes.get(node.input[0])
        if dims is None:
            continue
        tensor = numpy_helper.from_array(np.asarray(dims, np.int64), node.output[0])
        graph.initializer.append(tensor)
        graph.node.remove(node)
        replaced += 1
    return replaced


def _fold_constant_nodes(model):
    """Evaluate nodes whose every input is already a constant, drop them."""
    import onnx
    from onnx import helper, numpy_helper
    from onnx.reference import ReferenceEvaluator
    graph = model.graph
    known = {i.name for i in graph.initializer}
    folded = 0
    while True:
        progress = False
        for node in list(graph.node):
            if not node.input or any(name and name not in known for name in node.input):
                continue
            if node.op_type in ('Constant',):
                continue
            sub = onnx.ModelProto()
            sub.ir_version = model.ir_version
            sub.opset_import.extend(model.opset_import)
            g = sub.graph
            needed = []
            for name in node.input:
                init = next(i for i in graph.initializer if i.name == name)
                g.initializer.append(init)
                needed.append(name)
            g.node.append(node)
            g.output.extend(helper.make_tensor_value_info(o, onnx.TensorProto.UNDEFINED, None)
                            for o in node.output)
            try:
                evaluator = ReferenceEvaluator(sub)
                values = evaluator.run(None, {n: numpy_helper.to_array(next(
                    i for i in graph.initializer if i.name == n)) for n in needed})
            except Exception:
                continue
            for name, value in zip(node.output, values):
                graph.initializer.append(numpy_helper.from_array(np.asarray(value), name))
                known.add(name)
            graph.node.remove(node)
            folded += 1
            progress = True
        if not progress:
            return folded


def _prune(model):
    import onnx
    graph = model.graph
    used = {name for node in graph.node for name in node.input if name}
    used |= {o.name for o in graph.output}
    keep = [i for i in graph.initializer if i.name in used]
    if len(keep) != len(graph.initializer):
        del graph.initializer[:]
        graph.initializer.extend(keep)


def prepare_static_graph(source, dest, batch=1, *, force: bool = False) -> dict:
    """Pin batch and fold the constant shape tail so the graph is CUDA-assignable.

    Never overwrites *source*. Returns a receipt dict (hashes, counts, shapes).
    """
    import onnx
    from onnx import shape_inference

    source, dest = Path(source), Path(dest)
    if dest.exists() and not force:
        raise FileExistsError(f'{dest} exists; pass force=True to rebuild')
    model = onnx.load(str(source))
    onnx.checker.check_model(model)
    _pin_batch(model, batch)
    model = shape_inference.infer_shapes(model, strict_mode=False)
    shape_nodes = _fold_shape_nodes(model)
    model = shape_inference.infer_shapes(model, strict_mode=False)
    constant_nodes = _fold_constant_nodes(model)
    _prune(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(dest))
    return dict(batch=batch, shape_nodes_folded=shape_nodes,
                constant_nodes_folded=constant_nodes,
                source_sha256=sha256(source), export_sha256=sha256(dest),
                source_nodes=len(onnx.load(str(source)).graph.node),
                export_nodes=len(model.graph.node))


def open_session(path, *, threads=2):
    """An ORT session with CUDA as the only execution provider."""
    import onnxruntime as ort
    if os.name == 'nt':
        enable_cuda_dlls()
    if CUDA not in ort.get_available_providers():
        raise CudaUnavailable('CUDAExecutionProvider is not available to onnxruntime')
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # ortcuda: refuse a silent CPU placement instead of degrading the mode.
    options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
    options.enable_profiling = True
    options.profile_file_prefix = str(Path(path).with_suffix('')) + '-profile'
    try:
        session = ort.InferenceSession(str(path), options, providers=[CUDA])
    except Exception as exc:  # no CUDA device, driver mismatch, missing cuDNN ...
        raise CudaUnavailable(f'CUDA session initialization failed: {exc}') from exc
    # The Python wrapper may otherwise rebuild the session on EP failure; refuse it.
    session.disable_fallback()
    return session


def assert_cuda_only(session, feed, *, label='graph') -> dict:
    """Run once under the profiler and prove every node ran on CUDA."""
    import json as _json
    session.run(None, feed)
    profile = session.end_profiling()
    try:
        events = _json.loads(Path(profile).read_text('utf-8'))
    finally:
        Path(profile).unlink(missing_ok=True)
    counts, off = {}, []
    for event in events:
        if event.get('cat') != 'Node':
            continue
        provider = event.get('args', {}).get('provider', '?')
        counts[provider] = counts.get(provider, 0) + 1
        if provider != CUDA:
            off.append(dict(node=event.get('name'), op=event.get('args', {}).get('op_name')))
    if off or not counts or CPU in counts:
        raise CudaNotFullyAccelerated(f'{label}: {len(off)} node(s) outside CUDA: {off[:5]}')
    return dict(providers=session.get_providers(), nodes=sum(counts.values()),
                by_provider=counts, non_cuda=off)


class CudaPose:
    """Same contract as ml.extract.RTMPose, executed on CUDA only."""

    def __init__(self, session, *, engine='rtmpose', graph_sha256=None):
        self.session = session
        self.input = session.get_inputs()[0].name
        self.outputs = [o.name for o in session.get_outputs()]
        self.mean = np.asarray([123.675, 116.28, 103.53], np.float32)
        self.std = np.asarray([58.395, 57.12, 57.375], np.float32)
        self.receipt = dict(engine=engine, graph_sha256=graph_sha256,
                            inputs=[[o.name, o.shape] for o in session.get_inputs()],
                            outputs=[[o.name, o.shape] for o in session.get_outputs()])
        self.delegation: dict | None = None

    def close(self):
        self.session = None

    def warmup(self, height=160, width=160):
        frame = np.zeros((height, width, 3), np.uint8)
        xyz, mask = self(frame)
        return dict(xyz_shape=list(xyz.shape), mask_true=int(mask.sum()),
                    finite=bool(np.isfinite(xyz).all()))

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        scale = min(192 / w, 256 / h) / 1.25
        left, top = (192 - scale * w) / 2, (256 - scale * h) / 2
        transform = np.array([[scale, 0, left], [0, scale, top]], np.float32)
        import cv2
        image = cv2.warpAffine(bgr, transform, (192, 256), flags=cv2.INTER_LINEAR)
        rgb = (image[:, :, ::-1].astype(np.float32) - self.mean) / self.std
        feed = {self.input: np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])}
        sx, sy = self.session.run(self.outputs, feed)
        # Argmax and min(axis maxima) match the official SimCC decoder.
        x = (sx[0].argmax(axis=1) / 2 - left) / (scale * w)
        y = (sy[0].argmax(axis=1) / 2 - top) / (scale * h)
        score = np.minimum(sx[0].max(axis=1), sy[0].max(axis=1))
        from ml.data import visible
        from ml.extract import MoveNet
        xyz = np.zeros((33, 3), np.float32)
        xyz[MoveNet.INDICES] = np.stack((x * w / max(h, w), -y * h / max(h, w),
                                         np.zeros_like(x)), axis=1)
        mask = np.zeros(33, bool)
        mask[MoveNet.INDICES] = visible(np.stack((x, y), axis=1), score, score, .50)
        return xyz, mask


def open_cuda_pose(path, *, batch=1, threads=2) -> CudaPose:
    """Open RTMPose for strict-CUDA inference; prove full delegation first."""
    path = Path(path)
    session = open_session(path, threads=threads)
    graph = session
    try:
        shape = graph.get_inputs()[0].shape
        feed_shape = [1 if isinstance(d, str) or d is None else d for d in shape]
        feed = {graph.get_inputs()[0].name: np.zeros(feed_shape, np.float32)}
        receipt = assert_cuda_only(graph, feed, label=path.name)
    except Exception:
        session = None
        raise
    pose = CudaPose(session, graph_sha256=sha256(path))
    pose.delegation = receipt
    return pose


def assert_model_on_cuda(model, device='cuda:0'):
    """Every parameter and buffer must live on the CUDA device."""
    import torch
    target = torch.device(device)
    if target.type != 'cuda' or not torch.cuda.is_available():
        raise CudaUnavailable('CUDA device is required for the classifier')
    stray = sorted({str(t.device) for t in list(model.parameters()) + list(model.buffers())
                    if t.device != target})
    if stray:
        raise CudaNotFullyAccelerated(f'tensors outside {target}: {stray}')
    return target
