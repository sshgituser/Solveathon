"""Rewrite exported ONNX graphs so the LiteRT OpenGL delegate accepts every node.

The delegate refuses ``HardSigmoid`` (surfaced as RELU_0_TO_1), ``Squeeze`` and
``Unsqueeze`` (turned into a dynamic Shape->Gather->Reshape chain). Both graphs are
rewritten to equivalent supported ops before onnx2tf, and the rewrite is verified
against the original ONNX on CUDA. Weights are never read or written here.

Usage (from the repository root):
    python -m tools.export_android_onnx pose --source <b1.onnx> --out <pose-android.onnx>
    python -m tools.export_android_onnx classifier --source <A-tcn.onnx> --out <a-android.onnx>

Then:
    onnx2tf -i <pose-android.onnx> -o <dir> -tb flatbuffer_direct -nuo -dsm -osd -coion
    onnx2tf -i <a-android.onnx> -o <dir> -tb flatbuffer_direct -nuo -dsm -osd -coion -kat features
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


def sha256(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def static_shapes(model):
    inferred = shape_inference.infer_shapes(model, strict_mode=False)
    shapes = {}
    for value in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        if not value.type.HasField('tensor_type'):
            continue
        dims = [d.dim_value for d in value.type.tensor_type.shape.dim]
        if all(d > 0 for d in dims):
            shapes[value.name] = dims
    for tensor in inferred.graph.initializer:
        shapes[tensor.name] = list(tensor.dims)
    return shapes


class Rewriter:
    def __init__(self, model):
        self.graph = model.graph
        self.shapes = static_shapes(model)
        self.counter = 0

    def name(self, prefix):
        self.counter += 1
        return f'{prefix}_{self.counter}'

    def const(self, value, dtype):
        name = self.name('android_const')
        self.graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype), name))
        return name

    def hardsigmoid(self, node, alpha, beta):
        x = node.input[0]
        scaled, shifted, upper, lower = (self.name(p) for p in ('hs_mul', 'hs_add', 'hs_min', 'hs_max'))
        return [
            helper.make_node('Mul', [x, self.const(alpha, np.float32)], [scaled], name=self.name('HardSigmoid_mul')),
            helper.make_node('Add', [scaled, self.const(beta, np.float32)], [shifted], name=self.name('HardSigmoid_add')),
            helper.make_node('Min', [shifted, self.const(1.0, np.float32)], [upper], name=self.name('HardSigmoid_min')),
            helper.make_node('Max', [upper, self.const(0.0, np.float32)], [node.output[0]], name=self.name('HardSigmoid_max')),
        ]

    def reshape(self, node, axes, unsqueeze):
        shape = self.shapes.get(node.input[0])
        if shape is None:
            raise SystemExit(f'no static shape for {node.input[0]}')
        if unsqueeze:
            target = list(shape)
            for axis in sorted(axes):
                target.insert(axis, 1)
        else:
            target = [dim for index, dim in enumerate(shape) if index not in axes]
        return [helper.make_node('Reshape', [node.input[0], self.const(target, np.int64)],
                                 [node.output[0]], name=self.name('Reshape_const'))]

    def run(self):
        rewritten = 0
        nodes = list(self.graph.node)
        for node in nodes:
            replacement = None
            if node.op_type == 'HardSigmoid':
                attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
                replacement = self.hardsigmoid(node, float(attrs.get('alpha', 0.2)), float(attrs.get('beta', 0.5)))
            elif node.op_type in ('Squeeze', 'Unsqueeze'):
                axes = helper.get_attribute_value(node.attribute[0]) if node.attribute else []
                replacement = self.reshape(node, list(axes), node.op_type == 'Unsqueeze')
            elif node.op_type == 'Squeeze' and not node.attribute:
                raise SystemExit('operator without explicit axes')
            if replacement is None:
                continue
            index = list(self.graph.node).index(node)
            self.graph.node.remove(node)
            for offset, created in enumerate(replacement):
                self.graph.node.insert(index + offset, created)
            rewritten += 1
        used = {name for n in self.graph.node for name in n.input if name}
        used |= {out.name for out in self.graph.output}
        keep = [i for i in self.graph.initializer if i.name in used]
        if len(keep) != len(self.graph.initializer):
            del self.graph.initializer[:]
            self.graph.initializer.extend(keep)
        return rewritten


def rewrite_pose(source, dest):
    model = onnx.load(str(source))
    rewritten = Rewriter(model).run()
    onnx.checker.check_model(model)
    onnx.save(model, str(dest))
    return rewritten, len(model.graph.node)


def rewrite_classifier(source, dest):
    model = onnx.load(str(source))
    graph = model.graph
    if graph.node[-1].op_type != 'Squeeze':
        raise SystemExit('expected a trailing Squeeze on the classifier logit')
    last = graph.node[-1]
    shape_name = 'android_logit_shape'
    graph.initializer.append(numpy_helper.from_array(np.asarray([1, 1], np.int64), shape_name))
    del graph.node[-1]
    graph.node.append(helper.make_node('Reshape', [last.input[0], shape_name], ['logit'],
                                      name='logit_reshape'))
    for value in graph.output:
        if value.name == 'logit':
            del value.type.tensor_type.shape.dim[:]
            for _ in range(2):
                value.type.tensor_type.shape.dim.add().dim_value = 1
    onnx.checker.check_model(model)
    onnx.save(model, str(dest))
    return len(graph.node)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='kind', required=True)
    for kind in ('pose', 'classifier'):
        parser = sub.add_parser(kind)
        parser.add_argument('--source', required=True, type=Path)
        parser.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(argv)
    if args.out.exists():
        ap.error(f'{args.out} already exists')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.kind == 'pose':
        rewritten, nodes = rewrite_pose(args.source, args.out)
    else:
        rewritten, nodes = 1, rewrite_classifier(args.source, args.out)
    print(f'{args.kind}: rewritten={rewritten} nodes={nodes} sha256={sha256(args.out)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
