"""Wire invariants shared by the local HTTP interface and inference adapters."""
from math import isfinite
VERSION = 'seriq.v1'
MODELS = ('scratch', 'transfer')
LABELS = ('complete','incomplete','uncertain')

class ClipRejected(Exception):
    """Client-safe rejection from the local decoder."""
    def __init__(self, code='invalid_video'):
        self.code = code if code in ('invalid_video', 'duration_limit') else 'invalid_video'
        super().__init__(self.code)

def unavailable():
    return {'version':VERSION,'error':{'code':'model_unavailable'}}

def summarize(attempts):
    if not isinstance(attempts,list): raise ValueError('attempts must be a list')
    counts={'total':len(attempts),**dict.fromkeys(LABELS,0)}
    for i,a in enumerate(attempts,1):
        if not isinstance(a,dict) or set(a)!={'index','label','confidence','reason'} or type(a['index']) is not int or a['index']!=i: raise ValueError('invalid attempt')
        label=a['label']; confidence=a['confidence']; reason=a['reason']
        if label not in LABELS: raise ValueError('unknown label')
        if confidence is not None and (type(confidence) not in (float,int) or not isfinite(confidence) or not 0<=confidence<=1): raise ValueError('invalid confidence')
        if label=='uncertain':
            if reason not in ('low_visibility','low_confidence','unscorable'): raise ValueError('uncertain needs an explicit reason')
        elif reason is not None or confidence is None: raise ValueError('scored cycle needs confidence and no refusal reason')
        counts[label]+=1
    return counts

def validate_result(data):
    fields={'version','model','model_version','attempts','counts','warnings'}
    if not isinstance(data,dict) or set(data)!=fields or data['version']!=VERSION or data['model'] not in MODELS or not isinstance(data['model_version'],str) or not data['model_version'].strip(): raise ValueError('invalid result envelope')
    counts=summarize(data['attempts'])
    if not isinstance(data['counts'],dict) or any(type(v) is not int for v in data['counts'].values()) or data['counts']!=counts: raise ValueError('counts do not agree with attempts')
    warnings=data['warnings']
    if not isinstance(warnings,list) or any(w not in ('no_pose','low_visibility','partial_coverage') for w in warnings): raise ValueError('unexpected warning')
    if not data['attempts'] and 'no_pose' not in warnings: raise ValueError('empty detection must explain missing pose')
    if data['attempts'] and 'no_pose' in warnings: raise ValueError('no_pose cannot contain attempts')
    return data
