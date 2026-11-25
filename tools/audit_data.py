"""Read-only SerIQ data audit. Writes aggregates, never filenames or frames."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile
import zlib

SUBJECTS = {'01':'train','02':'train','03':'train','04':'train','05':'validation','08':'test'}
CODES = {'00':'incomplete','01':'complete'}
NAME = re.compile(r'(\d{2})_(\d{2})_(\d{2})_(\d+)\.mp4', re.I)

def inspect_bytes(path: Path) -> tuple[str,int]:
    h=hashlib.sha256(); crc=0
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block); crc=zlib.crc32(block,crc)
    return h.hexdigest(),crc & 0xffffffff

def audit(root: Path, archive: Path | None = None) -> dict:
    root=Path(root); base=root/'Exercise'
    if not base.is_dir(): raise ValueError('expected dataset root containing Exercise')
    files=sorted(p for p in base.rglob('*') if p.is_file())
    if not files: raise ValueError('no files found')
    if any(p.suffix.lower()!='.mp4' or p.is_symlink() for p in files): raise ValueError('unexpected input member')
    zip_members=None; zip_sha=None
    if archive:
        archive=Path(archive)
        with archive.open('rb') as stream: zip_sha=hashlib.file_digest(stream,'sha256').hexdigest()
        with ZipFile(archive) as z:
            infos=[i for i in z.infolist() if not i.is_dir()]; zip_members={i.filename.partition('/')[2]:i for i in infos}
        if len(zip_members)!=len(infos) or set(zip_members)!={p.relative_to(root).as_posix() for p in files}: raise ValueError('archive and local inventory differ')
    splits={s:{'incomplete':0,'complete':0,'total':0} for s in ('train','validation','test')}; groups={s:set() for s in splits}; quarantine=Counter(); digest_rows=[]; seen=set(); e02=0
    for f in files:
        rel=f.relative_to(root).as_posix(); parts=f.relative_to(root).parts
        target=len(parts)==5 and parts[1]=='2_Extending the Elbow'
        if not target and zip_members is None: continue
        digest,crc=inspect_bytes(f)
        if zip_members is not None:
            member=zip_members[rel]
            if f.stat().st_size!=member.file_size or crc!=member.CRC: raise ValueError('archive CRC/size mismatch')
        if not target: continue
        e02+=1
        if digest in seen: raise ValueError('duplicate E02 video content')
        seen.add(digest); m=NAME.fullmatch(f.name)
        if not m: raise ValueError('invalid E02 filename')
        sid,exercise,code,_=m.groups(); label=parts[2].lower(); orig=parts[3]
        if exercise!='02' or code not in CODES or label not in CODES.values() or sid not in SUBJECTS: raise ValueError('unrecognized E02 group or label schema')
        split=SUBJECTS[sid]
        if orig!=('Test' if split=='test' else 'Train'): raise ValueError('unexpected original partition')
        if CODES[code]!=label:
            quarantine['label_conflict']+=1; digest_rows.append([digest,sid,'quarantine','label_conflict']); continue
        groups[split].add(sid); splits[split][label]+=1; splits[split]['total']+=1; digest_rows.append([digest,sid,split,label])
    if not e02: raise ValueError('no E02 clips')
    serial=json.dumps(sorted(digest_rows),ensure_ascii=True,separators=(',',':')).encode('ascii')
    return {'schema':'seriq.audit.v1','data_doi':'10.17632/49h9dcwx5v.1','all_mp4':len(files),'e02_total':e02,'unique_e02_sha256':len(seen),'quarantined':dict(sorted(quarantine.items())),'splits':splits,'subjects':{s:sorted(v) for s,v in groups.items()},'assignment_digest_sha256':hashlib.sha256(serial).hexdigest(),'archive_sha256':zip_sha,'archive_crc_size_paths_matched':len(files) if zip_members is not None else None}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--zip',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--expect-v1',action='store_true');a=ap.parse_args(); r=audit(a.root,a.zip)
    if a.expect_v1:
        expected={'train':{'incomplete':47,'complete':30,'total':77},'validation':{'incomplete':10,'complete':10,'total':20},'test':{'incomplete':10,'complete':10,'total':20}}
        if (r['all_mp4'],r['e02_total'],r['unique_e02_sha256'],r['quarantined'],r['splits'])!=(491,127,127,{'label_conflict':10},expected): raise SystemExit('V1 cohort mismatch; no receipt written')
    text=json.dumps(r,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(text,encoding='utf-8',newline='\n')
    print(text,end='')

if __name__=='__main__': main()
