"""Exercise the CLI, frozen A/B and decoder locally, using only training and generated clips."""
from __future__ import annotations

import argparse
import http.client
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic


def request(port, route, data=None, media='video/mp4'):
    with HTTP(port) as conn:
        method = 'GET' if data is None else 'POST'
        conn.request(method, route, body=data, headers={} if data is None else {'Content-Type': media})
        response = conn.getresponse()
        return response.status, json.loads(response.read())


class HTTP(http.client.HTTPConnection):
    def __init__(self, port):
        super().__init__('127.0.0.1', port, timeout=60)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'artifacts', 'pose-model', 'slim-base', 'scratch'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--dll-dir', type=Path)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--ffprobe', default='ffprobe')
    a = parser.parse_args()
    # Deliberately not a manifest, fit, or sealed test command.
    clips = []
    for folder, code in (('Complete','01'),('Incomplete','00')):
        clip = next(iter(sorted((a.data_root/folder/'Train').glob(f'01_02_{code}_*.mp4'))), None)
        if clip is None:
            parser.error('one valid development clip per class is required')
        clips.append(clip)
    a.scratch.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='seriq-runtime-smoke-', dir=a.scratch) as temp:
        p = Path(temp)
        generated = {}
        for name, seconds, enc in (('blank.mp4', 1, 'libx264'), ('blank.webm', 1, 'libvpx-vp9'), ('long.mp4', 31, 'libx264')):
            target = p/name
            subprocess.run([a.ffmpeg,'-v','error','-nostdin','-f','lavfi','-i',f'color=teal:s=160x120:r=4:d={seconds}',
                            '-an','-c:v',enc,'-pix_fmt','yuv420p',str(target)],check=True,timeout=15)
            generated[name] = target.read_bytes()
        scratch = p/'uploads'
        args = [sys.executable, '-Bu','-m','backend.api','--port','0','--artifacts',str(a.artifacts),
                '--pose-model',str(a.pose_model),'--slim-base',str(a.slim_base),'--private-scratch',str(scratch),
                '--ffprobe',a.ffprobe]
        if a.dll_dir:
            args.extend(['--dll-dir',str(a.dll_dir)])
        server = subprocess.Popen(args,stdout=subprocess.PIPE,text=True,encoding='utf-8')
        try:
            assert server.stdout is not None
            start = server.stdout.readline()
            match = re.search(r'127\.0\.0\.1:(\d+)/', start)
            if not match:
                raise RuntimeError('local SerIQ CLI did not start')
            port = int(match.group(1))
            status, health = request(port,'/api/v1/health')
            assert status == 200 and health['models'] == {'scratch':True,'transfer':True}
            counts = {}
            times = {}
            for model in health['models']:
                route = f'/api/v1/analyze?model={model}&mode=single'
                counts[model] = []
                times[model] = []
                for clip in clips:
                    start = monotonic()
                    status, outcome = request(port,route,clip.read_bytes())
                    times[model].append(round(monotonic()-start, 2))
                    assert status == 200 and outcome['model'] == model, (status, outcome)
                    assert outcome['model_version'].startswith('seriq-e02-forearm-1.0.0+')
                    assert outcome['counts']['total'] == 1 and outcome['attempts'][0]['label'] in ('complete','incomplete'),outcome
                    counts[model].append(outcome['attempts'][0]['label'])
            route = '/api/v1/analyze?model=scratch&mode=single'
            for name, media in (('blank.mp4','video/mp4'),('blank.webm','video/webm')):
                status, fail = request(port,route,generated[name],media)
                assert status == 200 and fail['attempts'] == [] and fail['warnings'] == ['no_pose'],(name,status,fail)
            status, fail = request(port,route,generated['long.mp4'])
            assert status == 422 and fail['error']['code'] == 'duration_limit',(status,fail)
            status, fail = request(port,route,b'not a video')
            assert status == 422 and fail['error']['code'] == 'invalid_video',(status,fail)
            assert not any(scratch.iterdir()),'temporary uploads were not removed'
            print(json.dumps({'native_cli':True,'health':health['models'],'development_requests_per_model':2,
                              'development_class_counts':{m:{c:ls.count(c) for c in ('complete','incomplete')} for m,ls in counts.items()},
                              'max_development_request_seconds':{m:max(v) for m,v in times.items()},
                              'no_pose':['mp4','webm'],'duration_limit':422,'invalid_video':422,
                              'temporary_uploads_remaining':0,'sealed_test_reopened':False}))
        finally:
            server.terminate()
            server.wait(timeout=8)
            if server.stdout is not None:
                server.stdout.close()


if __name__ == '__main__':
    main()
