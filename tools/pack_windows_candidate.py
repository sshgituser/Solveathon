"""Produce one private, removable-media-ready Windows bundle without touching running files."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

from build_windows_engine import ROOT, application


def digest(data):
    return hashlib.sha256(data).hexdigest()


def member_name(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or any(x in ('.', '..') or ':' in x or '\\' in x for x in p.parts):
        raise ValueError('invalid manifest member')
    return p.as_posix()


def pack(engine, desktop, ref, out):
    commit = subprocess.check_output(['git','rev-parse','--verify',ref+'^{commit}'],cwd=ROOT,text=True).strip()
    if out.exists() or out.resolve().is_relative_to(ROOT):
        raise ValueError('output must be a new private ZIP outside the repo')
    base = json.loads((engine/'manifest.json').read_bytes())
    if base['schema'] != 'qolda-private-windows-engine-v1': raise ValueError('wrong engine manifest')
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp); app=stage/'app'; application(commit, app)
        target = stage/'release.zip'; checks={}; members={}
        with ZipFile(target,'x',compression=ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            def add(name, path, expected=None):
                name=member_name(name)
                if name in checks: raise ValueError('duplicate release member')
                chk=hashlib.sha256(); size=0
                with path.open('rb') as src, z.open(name,'w',force_zip64=True) as dst:
                    for chunk in iter(lambda: src.read(1<<20),b''):
                        chk.update(chunk); dst.write(chunk); size+=len(chunk)
                record=[size,chk.hexdigest()]
                if expected is not None and record!=expected: raise ValueError('source digest mismatch: '+name)
                checks[name]=record
                return record
            for name,expected in base['files'].items():
                name=member_name(name)
                if name.startswith('app/'): continue
                if name.split('/')[0] not in ('python','runtime','licenses','start-engine.cmd'):
                    raise ValueError('unexpected portable runtime entry')
                members[name]=add('engine-win64/'+name,engine/name,expected)
            for f in sorted(app.rglob('*')):
                if f.is_file():
                    name='app/'+f.relative_to(app).as_posix();members[name]=add('engine-win64/'+name,f)
            meta=dict(base,code_commit=commit,files=members)
            raw=(json.dumps(meta,sort_keys=True,separators=(',',':'))+'\n').encode();z.writestr('engine-win64/manifest.json',raw);checks['engine-win64/manifest.json']=[len(raw),digest(raw)]
            lines=(desktop/'SHA256SUMS.txt').read_text(encoding='ascii').splitlines()
            for row in lines:
                sha, name=row.split('  ',1); name=member_name(name); f=desktop/name
                add('windows/'+name,f,[f.stat().st_size,sha])
            add('windows/SHA256SUMS.txt',desktop/'SHA256SUMS.txt')
            readme=('Qolda: private Windows x64 demo kit. Do not publish this ZIP (contains local models).\r\n'
                    'Extract the entire archive onto a writable local drive, preserving windows/ and engine-win64/.\r\n'
                    'Requires a compatible NVIDIA driver and Microsoft Edge WebView2 Runtime. No Python install or pip required.\r\n'
                    'Launch windows/Qolda.exe. Browser only: run engine-win64/start-engine.cmd and open http://127.0.0.1:8765/.\r\n'
                    'An existing backend on port 8765 is not replaced. The owner must choose whether to switch versions.\r\n'
                    'A runs on GPU only; missing hardware is reported as unavailable. Local history is optional.\r\n'
                    'Live camera API is included; frontend from the code revision below, not a future UI.\r\n'
                    'Code: '+commit+'\r\n').encode('utf-8');z.writestr('START-HERE.txt',readme);checks['START-HERE.txt']=[len(readme),digest(readme)]
        with ZipFile(target) as z:
            if len(z.infolist())!=len(checks) or set(z.namelist())!=set(checks): raise ValueError('payload inventory differs')
            for e in z.infolist():
                chk=hashlib.sha256()
                with z.open(e) as src:
                    for chunk in iter(lambda:src.read(1<<20),b''): chk.update(chunk)
                if checks[e.filename] != [e.file_size,chk.hexdigest()]: raise ValueError('packed bytes differ: '+e.filename)
        with target.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
        target.rename(out)
    print(json.dumps({'schema':'qolda-combined-private-win64-v1','zip':out.name,'source_commit':commit,'bytes':out.stat().st_size,'sha256':sha,'verified_members':len(checks),'models':'A only','published':False},indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('engine','desktop','out'): ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--ref',required=True)
    a=ap.parse_args();pack(a.engine,a.desktop,a.ref,a.out)
