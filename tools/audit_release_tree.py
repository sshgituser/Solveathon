"""Read-only privacy inventory of one Git tree (not history). No filesystem/media traversal."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import PurePosixPath
import re
import subprocess

from tools.build_source_candidate import BINARY, DENY_DIRS, ROOT, SENSITIVE

PAYLOAD = {'.mp4','.webm','.mov','.avi','.mkv','.f32','.npz','.npy','.safetensors','.pt','.pth','.onnx',
           '.tflite','.pkl','.pickle','.apk','.aar','.zip','.exe','.dll','.so','.db','.sqlite','.sqlite3',
           '.keystore','.jks','.pem','.p12','.pfx','.key','.7z','.tar','.gz'}
WRAPPER = 'tools/android-engine/gradle/wrapper/gradle-wrapper.jar'
HOME = re.compile(rb'(?i)(?:[a-z]:[\\/]+users[\\/]+|/(?:home|Users)/)(?![<${%~]|Public(?:[\\/]|$))([a-z0-9_.-]+)[\\/]')


def git(*args, data=None):
    return subprocess.check_output(['git', *args], cwd=ROOT, input=data)


def audit(ref):
    commit=git('rev-parse','--verify','--end-of-options',ref+'^{commit}').decode().strip()
    entries=git('ls-tree','-rz','--long',commit).split(b'\0')[:-1]
    counts=Counter(); hits=Counter(); selected=[]
    for entry in entries:
        header,raw=entry.split(b'\t',1); mode,kind,oid,size=header.split(); name=raw.decode(); p=PurePosixPath(name)
        if kind!=b'blob' or mode not in (b'100644',b'100755'):
            hits['unexpected_tree_entry']+=1;continue
        if (p.suffix.lower() in PAYLOAD or any(part.lower() in DENY_DIRS|{'raw-data','pose-cache','holdout','quarantine'} for part in p.parts)
                or p.name in {'local.properties','key.properties','id_ed25519','id_rsa'} or
                (p.name.startswith('.env') and p.name!='.env.example')):
            hits['forbidden_path_not_opened']+=1;continue
        if int(size)>16*1024*1024:
            hits['unreviewed_large_blob_not_opened']+=1;continue
        selected.append((name,oid,int(size)))
    # Read selected object IDs in one process. Never emit matching content or user paths.
    out=git('cat-file','--batch',data=b''.join(oid+b'\n' for _,oid,_ in selected));offset=0
    for name,oid,size in selected:
        end=out.index(b'\n',offset);header=out[offset:end].split()
        if header!=[oid,b'blob',str(size).encode()]:raise RuntimeError('unexpected Git batch header')
        start=end+1;data=out[start:start+size];offset=start+size+1
        if out[offset-1:offset]!=b'\n':raise RuntimeError('short Git batch read')
        if name in BINARY:
            counts['public_binary_asset']+=1
            if not data.startswith(b'glTF' if name.endswith('.glb') else b'\x00\x01\x00\x00'):hits['asset_signature']+=1
        elif name==WRAPPER:
            counts['gradle_wrapper']+=1
            if hashlib.sha256(data).hexdigest()!='cb0da6751c2b753a16ac168bb354870ebb1e162e9083f116729cec9c781156b8':hits['wrapper_digest']+=1
        elif b'\0' in data:
            hits['unreviewed_binary']+=1
        else:
            counts['text_blobs_scanned']+=1
            if any(rule.search(data) for rule in SENSITIVE):hits['credential_pattern_files']+=1
            if HOME.search(data):hits['literal_home_path_files']+=1
    if offset!=len(out):raise RuntimeError('incorrect Git batch length')
    return {'schema':'qolda-git-tree-privacy-v1','commit':commit,'tracked_entries':len(entries),
            'scanned':dict(counts),'findings':dict(hits),'history_scanned':False,'worktree_scanned':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref',required=True)
    r=audit(parser.parse_args().ref);print(json.dumps(r,sort_keys=True));raise SystemExit(int(bool(r['findings'])))
