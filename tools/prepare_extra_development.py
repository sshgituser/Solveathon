"""Stage public V1 development clips only. Filter Test, protected IDs and label conflicts before opening any payload."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ('An upper limb stroke rehabilitation exercise video', 'Exercise')
EXERCISES = {'E01': ('1_Lifting an Object', {'01','02','03','04'}, {(0,'train'):40,(1,'train'):42,(0,'validation'):11,(1,'validation'):10}),
             'E03': ('3_Lifting the Wrist', {'02','03','04','06'}, {(0,'train'):41,(1,'train'):40,(0,'validation'):10,(1,'validation'):10})}

E02 = {'E02': ('2_Extending the Elbow', {'01','02','03','04'}, {(0,'train'):47,(1,'train'):30,(0,'validation'):10,(1,'validation'):10})}

def stage(archive, target, selection='extras'):
    if target.resolve().is_relative_to(ROOT) or target.exists():
        raise ValueError('require a new private output outside the checkout')
    target.parent.mkdir(exist_ok=True, parents=True)
    reports = {}; seen = set()
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp, ZipFile(archive) as z:
        work = Path(tmp)/'stage';work.mkdir()
        exercises={'extras': EXERCISES, 'E02': E02}[selection]
        for ex,(folder,train,expected) in exercises.items():
            dest = work/ex; media = dest/'media';media.mkdir(parents=True); rows=[]
            for info in sorted(z.infolist(),key=lambda i:i.filename):
                parts = PurePosixPath(info.filename).parts
                if len(parts)!=6 or parts[:3]!=(*PREFIX,folder) or parts[4]!='Train' or parts[3] not in {'Complete','Incomplete'}:continue
                match = re.fullmatch(r'(\d{2})_'+ex[1:]+r'_(00|01)_(\d{2})\.mp4',parts[5],flags=re.ASCII)
                if not match:continue
                who,code,_=match.groups()
                if who not in train | {'05'} or (code=='01')!=(parts[3]=='Complete'):continue
                if not 0<info.file_size<=64*1024*1024:raise ValueError('unexpected member size')
                # Member selected using exact directory, participant and concordant label before opening.
                item=media/parts[5]; h=hashlib.sha256()
                with z.open(info) as src,item.open('xb') as out:
                    for block in iter(lambda:src.read(1024*1024),b''):h.update(block);out.write(block)
                sha=h.hexdigest()
                if item.stat().st_size!=info.file_size or sha in seen:raise ValueError('size mismatch or duplicate development clip')
                seen.add(sha); rows.append(dict(path='media/'+parts[5],participant=who,label=int(code),split='validation' if who=='05' else 'train',sha256=sha))
            counts=Counter((r['label'],r['split']) for r in rows)
            if counts!=expected:raise ValueError('unexpected development groups/counts: '+ex)
            schema='qolda-public-e02-clean-development-v1' if ex=='E02' else 'qolda-additional-development-v1'
            manifest={'schema':schema,'exercise':ex,'source':'10.17632/49h9dcwx5v.1','rights':'CC BY 4.0','rows':rows}
            data=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode();(dest/'manifest.json').write_bytes(data)
            reports[ex]={'counts':{f'{s}_{c}':n for (c,s),n in sorted(counts.items())},'manifest_sha256':hashlib.sha256(data).hexdigest(),'clips':len(rows)}
        work.rename(target)
    return reports

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--archive',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);ap.add_argument('--selection',choices=('extras','E02'),default='extras');a=ap.parse_args();print(json.dumps(stage(a.archive,a.out,a.selection),indent=2))
