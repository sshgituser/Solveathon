"""Build a private CUDA reference for this workstation's Android export; compare rewritten ONNX on CUDA."""
import argparse
from pathlib import Path

import math
import numpy as np
from tools.android_from_source import CHANNELS, CONVERTER, GRAPHS, disk, private_dir, read_bundle, read_json, require, write_json
from tools.prepare_android_gpu_reference import classifier_inputs, classifier_reference, pose_inputs, pose_reference
from ml.api import load_artifact
from ml.data import verify_training_inputs
from ml.gpu import open_session, assert_cuda_only


def dev_pose_comparison(manifest, source, rewritten):
    """Only aggregate scores from six fixed frames; never export real images or pose tensors."""
    import cv2
    pin='8c73b2a35584639f3a2ea057878d39cf105c52ffa4d77a4a41639c2606134931'
    import hashlib, json
    raw=manifest.read_bytes();require(hashlib.sha256(raw).hexdigest()==pin, 'Not the approved clean E02 development manifest')
    rows=json.loads(raw)['rows']; base=open_session(source); name=base.get_inputs()[0].name; other=rewritten.get_inputs()[0].name
    stats={'frames':0,'axis_visible_keypoints':0,'max_visible_bin_delta':0,'max_abs_score_delta':0.0}
    for label in (0,1):
        row=next(r for r in rows if r['label']==label and r['split']=='validation' and r['participant']=='05')
        path=manifest.parent/row['path'];require(path.resolve()==path.absolute() and disk(path)['sha256']==row['sha256'],'Changed or redirected approved clip')
        cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG)
        try:
            length=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));require(cap.isOpened() and length>=4,'Invalid development clip')
            for index in (length//4,length//2,3*length//4):
                require(cap.set(cv2.CAP_PROP_POS_FRAMES,index),'Cannot seek');ok,frame=cap.read();require(ok,'Cannot decode fixed frame')
                h,w=frame.shape[:2];s=min(192/w,256/h)/1.25
                transform=np.array([[s,0,(192-s*w)/2],[0,s,(256-s*h)/2]],np.float32)
                rgb=cv2.warpAffine(frame,transform,(192,256))[:,:,::-1].astype(np.float32)
                rgb=(rgb-np.array([123.675,116.28,103.53],np.float32))/np.array([58.395,57.12,57.375],np.float32)
                x=np.ascontiguousarray(rgb.transpose(2,0,1)[None]);a=[np.asarray(v) for v in base.run(['simcc_x','simcc_y'],{name:x})];b=[np.asarray(v) for v in rewritten.run(['simcc_x','simcc_y'],{other:x})]
                require(all(u.shape==v.shape and np.isfinite(u).all() and np.isfinite(v).all() for u,v in zip(a,b)), 'Invalid pose output')
                mask=np.minimum(a[0].max(-1),a[1].max(-1))>=.5
                require(np.array_equal(mask,np.minimum(b[0].max(-1),b[1].max(-1))>=.5),'Axis visibility differs')
                stats['frames']+=1;stats['axis_visible_keypoints']+=int(mask.sum())
                for u,v in zip(a,b):
                    stats['max_visible_bin_delta']=max(stats['max_visible_bin_delta'],int(np.abs(u.argmax(-1)-v.argmax(-1))[mask].max(initial=0)))
                    stats['max_abs_score_delta']=max(stats['max_abs_score_delta'],float(np.abs(u-v).max()))
        finally:
            cap.release()
    # Consume and remove the ORT profile (no lingering per-run profile alongside the model).
    assert_cuda_only(base,{name:x},label='E02 dev pose reference')
    require(stats['axis_visible_keypoints']>0 and stats['max_visible_bin_delta']<=1,'Visible-keypoint parity not established')
    return stats


def build(bundle, weights, pose, classifier, conversion, out, dev_manifest=None):
    descriptor = read_bundle(bundle)
    model = descriptor['model']; graphs = read_json(conversion/'conversion.json')
    require(graphs['schema']=='qolda-android-local-conversion-v1' and graphs['converter']==CONVERTER
            and graphs['bundle']==disk(bundle), 'Conversion belongs to another bundle or recipe')
    export = read_json(classifier.with_suffix('.onnx.json'))
    require(export['schema']=='qolda-android-local-a-onnx-v1' and export['bundle']==graphs['bundle']
            and export['channels']==CHANNELS and export['device']=='cuda:0' and export['graph']==disk(classifier)
            and export['artifact']==disk(weights)=={'bytes':descriptor['artifact_bytes'],'sha256':model['sha256']},
            'Classifier export or checkpoint mismatch')
    pred = load_artifact(weights, device='cuda:0', expected_sha256=model['sha256'])
    require(pred.metadata['threshold']==model['threshold'] and pred.metadata['variant']=='A', 'Classifier policy mismatch')
    verify_training_inputs(pred.metadata,manifest=descriptor['training_manifest_sha256'],original=descriptor['pose']['original_sha256'],graph=descriptor['pose']['export_sha256'])
    inputs={'pose':pose,'classifier':classifier}; rewritten={}; tflite={}
    for name,(stem,_,_,_) in GRAPHS.items():
        rec=graphs['graphs'][name];rewritten[name]=conversion/(stem+'-android.onnx');target=conversion/(stem+'-fp32.tflite')
        require(rec['input_onnx']==disk(inputs[name]) and rec['rewritten_onnx']==disk(rewritten[name]) and
                rec['file']==target.name and disk(target)=={k:rec[k] for k in ('bytes','sha256')},'Mismatched '+name+' conversion input or output')
        tflite[name]=rec['sha256']
    require(disk(pose)['sha256']==descriptor['pose']['export_sha256'],'Pose differs from trained checkpoint')
    out=private_dir(out)
    poses=pose_inputs(); features=classifier_inputs()
    base_profile,refs=pose_reference(pose,out,poses); decisions=classifier_reference(pred,out,features)
    ps=open_session(rewritten['pose']); cs=open_session(rewritten['classifier']); pi=ps.get_inputs()[0].name
    def pose_feed(x): return {pi:np.ascontiguousarray(x.transpose(0,3,1,2))}
    pp=assert_cuda_only(ps,pose_feed(next(iter(poses.values())))); cp=assert_cuda_only(cs,{'features':next(iter(features.values()))})
    axes={}; logits={}
    for name,x in poses.items():
        results=[np.asarray(v) for v in ps.run(['simcc_x','simcc_y'],pose_feed(x))];old=[]
        for k in ('simcc_x','simcc_y'):
            f=refs[name][k];old.append(np.fromfile(out/f['file'],dtype='<f4').reshape(f['shape']))
        require(all(a.shape==b.shape and np.isfinite(b).all() for a,b in zip(old,results,strict=True)), 'Invalid SimCC output')
        mask=np.minimum(old[0].max(-1),old[1].max(-1))>=.5
        newmask=np.minimum(results[0].max(-1),results[1].max(-1))>=.5
        diff=max(int(np.abs(a.argmax(-1)-b.argmax(-1))[mask].max(initial=0)) for a,b in zip(old,results))
        require(np.array_equal(mask,newmask) and diff<=1,'Pose rewrite changed visibility or visible bins')
        axes[name]={'visible':int(mask.sum()),'max_visible_bin_delta':diff,'max_abs_score_delta':max(float(np.abs(a-b).max()) for a,b in zip(old,results))}
    for name,x in features.items():
        raw=np.asarray(cs.run(['logit'],{'features':x})[0]);require(raw.shape==(1,1) and np.isfinite(raw).all(),'Invalid logit output')
        prob=1/(1+math.exp(-float(raw.item()))) if raw.item()>=0 else math.exp(float(raw.item()))/(1+math.exp(float(raw.item())))
        ref=decisions[name];delta=abs(prob-ref['p_complete']);require(delta<=.001 and (prob>=model['threshold'])==ref['threshold_side'],'Classifier rewrite changed decision or probability')
        logits[name]=delta
    return write_json(out/'reference.json',{'schema':'qolda-local-android-cuda-reference-v1','bundle':disk(bundle),'conversion':disk(conversion/'conversion.json'),
        'checkpoint_sha256':model['sha256'],'threshold':model['threshold'],'tflite_sha256':tflite,'input_channels':CHANNELS,'pose_cases':refs,'classifier_cases':decisions,
        'cuda_profiles':{'source_pose':base_profile,'rewritten_pose':pp,'rewritten_classifier':cp},'pose_comparison':axes,'probability_deltas':logits,
        'generated_reference_files_only':True,'dev_frame_comparison':dev_pose_comparison(dev_manifest.resolve(),pose,ps) if dev_manifest else None,
        'onnx_rewrite_verified_on_cuda':True,'tflite_execution_verified':False,'phone_parity_verified':False})


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','weights','pose','classifier','conversion','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--dev-manifest',type=Path,help='optional pinned clean E02 development manifest; no real frames are exported')
    a=ap.parse_args();print(build(**vars(a)))
