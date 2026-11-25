"""Private-replay delivery for frozen E01/E03. No publication, no training, no service changes."""
from pathlib import Path
import argparse, hashlib, json, subprocess, tempfile
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
REF='d0e2991173f1c72e085ffd892c1c682d51a505b2'
PINS={'E01':('8582ba137631d8ab2fee79a083fbffe102dae0b359e31765b7957799c00ec33a','287285d9386f26f7d147a9821fdd46b38f32673f801ea7c7df336ddefe7015eb','engine.json'),
      'E03':('95abbe0a8c77b36d9f319648a6f49490d77600c42479b18d14500cfcaac6f822','125abc289062c15a26a490bbaf7306d73767fcaddbea6c62fff81282a0d2b7ea','model.json')}
POSE='2e0a9c0c5287bfa8897440975c32e98e26be90e88ad9ddd9e72fba8644b68b78'

def sha(data):return hashlib.sha256(data).hexdigest()
def blob(path):return subprocess.check_output(['git','show',REF+':'+path],cwd=ROOT)

def pack(source,output):
    source=source.resolve(strict=True);output=output.resolve()
    if source.is_relative_to(ROOT) or output.is_relative_to(ROOT) or output.is_relative_to(source) or output.exists():raise ValueError('select separate private inputs and new output outside Git')
    members={};summary={}
    for ex,(weight,pin,config) in PINS.items():
        run=source/ex/'run'; report=json.loads((run/'receipt.json').read_bytes()); meta=(run/config).read_bytes();metadata=json.loads(meta); module='ml/extra_'+ex.lower()+'.py';code=blob(module);weights=(run/'checkpoint.safetensors').read_bytes()
        if not (report['exercise']==metadata['exercise']==ex and sha(weights)==weight and report['manifest_sha256']==pin and sha(code)==report['source_sha256']==metadata['source_sha256']):raise ValueError('unrecognized research release: '+ex)
        if ex=='E01':
            graph=(run/'end2end.b1.onnx').read_bytes()
            if sha(graph)!=POSE or metadata['resources']!={'checkpoint.safetensors':weight,'end2end.b1.onnx':POSE}:raise ValueError('E01 graph or classifier mismatch')
            members[ex+'/run/end2end.b1.onnx']=graph
        elif metadata['weights_sha256']!=weight:raise ValueError('E03 weight metadata mismatch')
        members[module]=code;members[ex+'/run/checkpoint.safetensors']=weights;members[ex+'/run/'+config]=meta
        name='reports/ml/extra-'+ex.lower()+'.md';members[name]=blob(name)
        summary[ex]={'checkpoint_sha256':weight,'passed_local_promotion_gate':report['promotion_allowed'],'product_approved':False}
    members['artifacts/seriq/ATTRIBUTION.md']=blob('artifacts/seriq/ATTRIBUTION.md')
    members['START-HERE.txt']=('Qolda — приватные исследовательские модели E01 и E03. Не устанавливать в качестве финальной поставки.\nКод моделей: '+REF+'. Требуется уже поставленный Windows CUDA engine и совместимый NVIDIA-драйвер; автономного Python или APK здесь нет.\nВ архиве НЕТ клипов и разрешений promotion.json. Использовать только локально на одобренном клипе выбранного упражнения.\nЧисла относятся к одному участнику validation; E01 не проверяет наличие предмета, E03 не измеряет угол запястья. См. reports/ml.\n').encode('utf-8-sig')
    manifest={'schema':'qolda-private-additional-research-v1','source_commit':REF,'exercises':summary,'files':{n:{'bytes':len(b),'sha256':sha(b)} for n,b in sorted(members.items())}}
    members['MANIFEST.json']=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        draft=Path(tmp)/'release.zip'
        with ZipFile(draft,'x',compression=ZIP_DEFLATED,compresslevel=3) as z:
            for k in sorted(members):z.writestr(k,members[k])
        with ZipFile(draft) as z:
            if len(z.infolist())!=len(members) or set(z.namelist())!=set(members) or not all(z.read(k)==b for k,b in members.items()):raise ValueError('archive readback differs')
        draft.rename(output)
    with output.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    print(json.dumps({'zip':output.name,'bytes':output.stat().st_size,'sha256':digest,'verified_members':len(members),'models':summary,'published':False}))

if __name__=='__main__':
    par=argparse.ArgumentParser(description=__doc__);par.add_argument('--source',type=Path,required=True);par.add_argument('--out',type=Path,required=True);a=par.parse_args();pack(a.source,a.out)
