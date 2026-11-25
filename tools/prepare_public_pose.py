"""Acquire the public RTMPose model from its original publisher on the training machine."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

URL = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip'
ZIP_SHA = 'f7fbb6c5c11a1bb70f3d445e4ddec5d144ea89ad5649c081ef976f9b24a0b741'
MODEL_SHA = '5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c'
STATIC_SHA = '2e0a9c0c5287bfa8897440975c32e98e26be90e88ad9ddd9e72fba8644b68b78'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def prepare(out, local=None):
    out = out.resolve()
    if out.exists() or any((p / '.git').exists() for p in (out, *out.parents)):
        raise ValueError('Choose a NEW output outside all Git working trees')
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / 'pose'; stage.mkdir()
        archive = local
        if archive is None:
            archive = Path(tmp) / 'publisher.zip'
            with urlopen(URL, timeout=90) as src, archive.open('xb') as dst:
                shutil.copyfileobj(src, dst)
        if digest(archive) != ZIP_SHA:
            raise ValueError('Publisher archive SHA-256 mismatch')
        original = stage / 'end2end.onnx'
        with ZipFile(archive) as z:
            (entry,) = [i for i in z.infolist() if PurePosixPath(i.filename).name == original.name]
            with z.open(entry) as src, original.open('xb') as dest:
                shutil.copyfileobj(src, dest)
        if digest(original) != MODEL_SHA:
            raise ValueError('Original ONNX digest mismatch')
        from ml.gpu import prepare_static_graph
        graph = stage / 'end2end.b1.onnx'
        prepare_static_graph(original, graph)
        if digest(graph) != STATIC_SHA:
            raise ValueError('Unexpected converted graph: verify pinned ONNX version; do not change the pin')
        receipt = {'publisher_url':URL,'archive_sha256':ZIP_SHA,'original_sha256':MODEL_SHA,'static_sha256':STATIC_SHA}
        (stage/'source.json').write_text(json.dumps(receipt, indent=2)+'\n',encoding='utf-8')
        stage.rename(out)
        return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--publisher-archive', type=Path, help='Optional archive downloaded directly from the pinned publisher URL')
    a = parser.parse_args()
    print(json.dumps(prepare(a.out, a.publisher_archive)))
