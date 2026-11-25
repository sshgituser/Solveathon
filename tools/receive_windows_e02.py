"""Run one bounded file-API acceptance on the packaged interpreter, not a working server."""
import argparse
import hashlib
import http.client
from http.server import HTTPServer
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread

MANIFEST='8c73b2a35584639f3a2ea057878d39cf105c52ffa4d77a4a41639c2606134931'

def require(ok, message):
    if not ok:raise ValueError(message)


def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def run(args):
    eng=args.engine.resolve(strict=True);runtime=eng/'runtime'; exe=(eng/'python/python.exe').resolve(strict=True)
    require(Path(sys.executable).resolve()==exe and sys.flags.isolated and all(Path(p).resolve().is_relative_to(eng) for p in sys.path),'not the isolated packaged interpreter')
    from backend import api, runtime as backend
    app=eng/'app';require(Path(api.__file__).resolve().is_relative_to(app) and Path(backend.__file__).resolve().is_relative_to(app),'backend loaded outside package')
    manifest=args.dev_manifest.resolve(strict=True);raw=manifest.read_bytes();require(hashlib.sha256(raw).hexdigest()==MANIFEST,'unapproved development manifest')
    validation=[r for r in json.loads(raw)['rows'] if r['split']=='validation'];inputs=[]
    require(len(validation)==20 and all(r['participant']=='05' for r in validation)
            and all(sum(r['label']==k for r in validation)==10 for k in (0,1)), 'unexpected validation cohort')
    selected=validation if args.full_validation else [next(r for r in validation if r['label']==k) for k in (0,1)]
    for r in selected:
        f=manifest.parent/r['path'];require(f.absolute()==f.resolve(strict=True) and f.stat().st_nlink==1,'unexpected development clip');inputs.append((f,r['sha256'],('incomplete','complete')[r['label']]))
    blank=args.blank.absolute();require(re.fullmatch(r'[0-9a-f]{64}',args.blank_sha256) and blank==blank.resolve(strict=True) and blank.stat().st_nlink==1 and 0<blank.stat().st_size<=64*1024*1024 and digest(blank)==args.blank_sha256,'static fixture differs from its specified pin')
    bundle=json.loads((runtime/'bundle.json').read_bytes());key=bundle['model']['sha256'];version=bundle['model']['version']+'+'+key[:12]
    require(bundle['training_manifest_sha256']==MANIFEST and digest(runtime/'A-tcn.safetensors')==key,'not the approved local rebuild')
    scratch=args.scratch.resolve(strict=True);require(not any((p/'.git').exists() for p in (scratch,*scratch.parents)) and not scratch.is_relative_to(eng),'scratch must be private and separate')
    with TemporaryDirectory(dir=scratch) as tmp:
        engines=backend.load_bundle_engines(runtime/'bundle.json', runtime, runtime/'end2end.onnx',Path(tmp),str(runtime/'ffprobe.exe'))
        require(callable(engines.get('scratch')) and not engines.get('transfer'),'unexpected E02 capabilities')
        results=[]
        with HTTPServer(('127.0.0.1',0),api.make_handler(engines,app)) as server:
            def request(path,body=None):
                worker=Thread(target=server.handle_request,daemon=True);worker.start();conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=70)
                try:
                    conn.request('POST' if body is not None else 'GET',path,body,{'Content-Type':'video/mp4','Origin':f'http://127.0.0.1:{server.server_port}'})
                    response=conn.getresponse();data=response.read();require(response.status==200 and response.getheader('Cache-Control')=='no-store','unexpected response');return data
                finally:
                    conn.close();worker.join(5)
            require(request('/')==(app/'frontend/index.html').read_bytes(),'wrong bundled frontend')
            health=json.loads(request('/api/v1/health'));require(health['models']['scratch'] and not health['models']['transfer'],'health mismatch')
            for f,pin,expected in inputs+[(blank,args.blank_sha256,'no_pose')]:
                body=f.read_bytes();require(hashlib.sha256(body).hexdigest()==pin,'unexpected input bytes')
                reply=json.loads(request('/api/v1/analyze?model=scratch&mode=single',body));counts=reply['counts']
                require(reply['model']=='scratch' and reply['model_version']==version,'wrong model')
                require(reply['version']=='seriq.v1' and all(type(counts[k]) is int and counts[k]>=0 for k in ('total','complete','incomplete','uncertain')) and counts['total']==len(reply['attempts'])==sum(counts[k] for k in ('complete','incomplete','uncertain')),'invalid attempt accounting')
                if expected=='no_pose':require(counts['total']==0 and not reply['attempts'] and 'no_pose' in reply['warnings'],'unexpected static scene result')
                results.append({'expected':expected,'counts':{k:counts[k] for k in ('total','complete','incomplete','uncertain')},'matches_label':(counts['total']==1 and counts.get(expected)==1 and reply['attempts'][0]['label']==expected) if expected!='no_pose' else True})
        require(not any(Path(tmp).iterdir()),'upload scratch was not empty after analysis')
    receipt={'schema':'qolda-windows-local-e02-file-v1','model_version':version,'weights_sha256':key,'manifest_sha256':MANIFEST,'engine_inventory_sha256':digest(eng/'manifest.json'),'isolated_python':True,'http_status':200,'served_bundled_index':True,'results':results,'scratch_empty':True,'exe_ui_exercised':False,'clean_machine_verified':False}
    if args.full_validation:
        # Do not export per-clip decisions. Preserve refusals instead of dropping them.
        labels=('incomplete','complete');columns=(*labels,'uncertain','no_attempt','multiple_attempts')
        matrix=[[0]*len(columns) for _ in labels]
        for item in results[:-1]:
            counts=item['counts'];n=counts['total']
            outcome='no_attempt' if not n else 'multiple_attempts' if n>1 else next(k for k in columns[:3] if counts[k]==1)
            matrix[labels.index(item['expected'])][columns.index(outcome)]+=1
        del receipt['results']
        receipt.update(schema='qolda-windows-e02-full-validation-v1',class_order=labels,columns=columns,
                       confusion=matrix,subjects=1,independent_evaluation=False,blank_no_pose=True,
                       all_match=all(item['matches_label'] for item in results))
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('engine','dev-manifest','blank','scratch'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--blank-sha256',required=True,help='SHA-256 of the explicitly selected local static fixture')
    parser.add_argument('--full-validation',action='store_true',help='replay all 20 canonical validation files; emit aggregate counts only')
    parser.add_argument('--inside',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.inside:
        receipt=run(args);print(json.dumps(receipt,sort_keys=True))
        matched=receipt['all_match'] if args.full_validation else all(r['matches_label'] for r in receipt['results'])
        sys.exit(0 if matched else 2)
    else:
        exe=args.engine.resolve(strict=True)/'python/python.exe';sys.exit(subprocess.run([str(exe),'-B',str(Path(__file__).resolve()),*sys.argv[1:],'--inside'],timeout=235).returncode)
