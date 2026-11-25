"""SerIQ loopback API. No raw-video logs, telemetry, permanent uploads or fixture predictions."""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import socket
from threading import BoundedSemaphore
from time import monotonic
from typing import cast
from urllib.parse import parse_qs
from backend.contract import ClipRejected, MODELS, VERSION, validate_result
from backend.live import LiveError, MAX_SIDE
from backend.extra_exercises import ExtraEngines, catalog, VERSION as EXTRA_VERSION

MAX_BYTES = 64 * 1024 * 1024
ACCEPTED_TYPES = {'video/mp4', 'video/webm'}
ROOT = Path(__file__).resolve().parents[1]
HEALTH = '/api/v1/health'
ANALYZE = '/api/v1/analyze'
EXERCISES = '/api/v1/exercises'
EXTRA_ROUTES = {f'{EXERCISES}/{exercise}/analyze': exercise for exercise in ('E01', 'E03')}
LIVE_ROUTES = {f'/api/v1/live/{op}': op for op in ('start', 'frame', 'finish', 'cancel')}
CLIENT_ORIGINS = {'http://127.0.0.1:5173', 'http://localhost:5173',
                  'http://127.0.0.1:8080', 'http://localhost:8080',
                  'https://localhost', 'capacitor://localhost', 'http://localhost'}
WEB = dict.fromkeys(('index.html', 'brand.html', 'lab.html'), 'text/html; charset=utf-8')
WEB_TREES = {'src': (re.compile(r'(?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_-]+\.js'), 'text/javascript; charset=utf-8'),
             'styles': (re.compile(r'(?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_-]+\.css'), 'text/css; charset=utf-8')}
ASSETS = {'assets/brand/qolda-mark.svg': 'image/svg+xml', 'assets/brand/qolda-mark-dark.svg': 'image/svg+xml',
          'assets/fonts/IBMPlexSans.ttf': 'font/ttf', 'assets/fonts/SourceSerif4.ttf': 'font/ttf',
          'assets/models/mannequin.glb': 'model/gltf-binary', 'assets/instruction/instruction.mp4': 'video/mp4',
          'assets/instruction/instruction.ru.vtt': 'text/vtt; charset=utf-8',
          'assets/instruction/instruction.kk.vtt': 'text/vtt; charset=utf-8'}
ASSETS.update({'assets/vendor/three/' + file: 'text/javascript; charset=utf-8' for file in (
    'three.module.min.js', 'GLTFLoader.js', 'RoomEnvironment.js', 'BufferGeometryUtils.js')})


def web_asset(root, route):
    if root is None:
        return None
    rel=route.removeprefix('/')
    if rel in ('','frontend','frontend/'):
        rel='index.html'
    rel=rel.removeprefix('frontend/')
    base=root
    if rel in WEB:
        rel, mime='frontend/'+rel, WEB[rel]
    elif rel in ASSETS:
        mime=ASSETS[rel]
    else:
        section, _, rel=rel.partition('/')
        types=WEB_TREES.get(section)
        if not types or not types[0].fullmatch(rel):
            return None
        base, mime=root/'frontend'/section, types[1]
    try:
        base=base.resolve(strict=True)
        resolved=(base/rel).resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    return (resolved, mime) if base.is_relative_to(root) and resolved.is_relative_to(base) and resolved.is_file() else None

def make_handler(engines, web_root: Path | None = ROOT, extra=None):
    """An engine is loaded and warmed before registration; (body, mime) -> v1 dict."""
    guard=BoundedSemaphore(1)
    if extra is None: extra=ExtraEngines()
    web_root=Path(web_root).resolve() if web_root is not None else None
    class Handler(BaseHTTPRequestHandler):
        server_version='SerIQ/1'; sys_version=''
        def setup(self):
            super().setup(); self.connection.settimeout(15)
        def log_message(self, format, *args):
            pass  # Never log input bytes, local paths, query strings or subject IDs.
        def origin_ok(self):
            port=cast(ThreadingHTTPServer, self.server).server_port
            hosts={f'127.0.0.1:{port}', f'localhost:{port}'}
            if port == 80:
                hosts.update(('127.0.0.1','localhost'))
            host=self.headers.get_all('Host',[])
            if len(host)!=1 or host[0].lower() not in hosts: return False
            origins=self.headers.get_all('Origin',[])
            return not origins or (len(origins)==1 and origins[0] in (CLIENT_ORIGINS | {f'http://{h}' for h in hosts}))
        def head(self,status,length,mime=None,*,cors=False,extra=()):
            self.send_response(status)
            self.send_header('Connection','close')
            if mime:
                self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(length))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            if cors:
                self.send_header('Vary','Origin')
                if self.origin_ok() and self.headers.get('Origin'):
                    self.send_header('Access-Control-Allow-Origin',self.headers['Origin'])
            for k,v in extra:
                self.send_header(k,v)
            self.end_headers()
        def send_json(self, status, obj, *, cors=True):
            body=json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')
            self.head(status,len(body),'application/json; charset=utf-8',cors=cors)
            if self.command!='HEAD':
                self.wfile.write(body)
        def err(self,status,code,*,cors=True):
            route=self.path.partition('?')[0]
            version=EXTRA_VERSION if route==EXERCISES or route in EXTRA_ROUTES else VERSION
            self.send_json(status,{'version':version,'error':{'code':code}},cors=cors)
        def reject(self,status,code,*,cors=True):
            self.err(status,code,cors=cors)
            # On Windows a close with unread body can reset the connection before JSON reaches the client.
            # Half-close after the reply, then discard a bounded quantity without storing it.
            values=self.headers.get_all('Content-Length',[])
            if len(values)!=1 or not values[0].isascii() or not values[0].isdecimal() or len(values[0])>12:
                return
            remaining=min(int(values[0]),MAX_BYTES)
            deadline=monotonic()+0.8
            try:
                self.connection.shutdown(socket.SHUT_WR)
                while remaining and (budget:=deadline-monotonic())>0:
                    self.connection.settimeout(budget)
                    chunk=self.rfile.read1(min(remaining,64*1024))
                    if not chunk: break
                    remaining-=len(chunk)
            except OSError:
                pass
        def do_OPTIONS(self):
            if not self.origin_ok(): return self.err(403,'forbidden',cors=False)
            route=self.path.partition('?')[0]
            if route not in (ANALYZE,HEALTH,EXERCISES) and route not in EXTRA_ROUTES: return self.err(404,'not_found')
            method='POST' if route==ANALYZE or route in EXTRA_ROUTES else 'GET'
            requested=self.headers.get_all('Access-Control-Request-Method',[])
            headers={part.strip().lower() for line in self.headers.get_all('Access-Control-Request-Headers',[])
                     for part in line.split(',') if part.strip()}
            if not self.headers.get('Origin') or requested!=[method] or not headers<={'content-type'}:
                return self.err(403,'forbidden',cors=False)
            self.head(204,0,cors=True,extra=(('Access-Control-Allow-Methods',method),
                                            ('Access-Control-Allow-Headers','Content-Type')))
        def do_GET(self):
            if not self.origin_ok(): return self.err(403,'forbidden',cors=False)
            route=self.path.partition('?')[0]
            if route==HEALTH:
                return self.send_json(200,{'version':VERSION,'models':{name:callable(engines.get(name)) for name in MODELS}})
            if route==EXERCISES:
                if '?' in self.path: return self.err(400,'invalid_request')
                return self.send_json(200,catalog(engines,extra))
            asset=web_asset(web_root,route)
            if asset is None: return self.err(404,'not_found')
            target,mime=asset
            try:
                body=target.read_bytes()
            except OSError:
                return self.err(404,'not_found')
            self.head(200,len(body),mime)
            if self.command!='HEAD': self.wfile.write(body)
        do_HEAD = do_GET
        def live_request(self, op, query):
            # Camera frames may only come from the page served by this very engine.
            origin = self.headers.get('Origin')
            if origin is not None and origin != 'http://' + self.headers['Host'].lower():
                return self.reject(403,'forbidden',cors=False)
            if self.headers.get_all('Transfer-Encoding'):
                return self.err(400,'invalid_request')
            seq = w = h = 0
            try:
                q = parse_qs(query, keep_blank_values=True, strict_parsing=True, max_num_fields=4)
                required = set() if op=='start' else {'session','seq','w','h'} if op=='frame' else {'session'}
                if set(q)!=required or any(len(values)!=1 for values in q.values()):
                    raise ValueError
                token = q['session'][0] if 'session' in q else ''
                if token and not re.fullmatch(r'[A-Za-z0-9_-]{32}', token): raise ValueError
                if op=='frame':
                    if not all(re.fullmatch(r'[1-9][0-9]{0,3}',q[k][0]) for k in ('seq','w','h')): raise ValueError
                    seq,w,h = (int(q[k][0]) for k in ('seq','w','h'))
                    if max(w,h)>MAX_SIDE: raise ValueError
            except ValueError: return self.reject(400,'invalid_request')
            mime = 'application/octet-stream' if op=='frame' else 'application/json'
            if len(self.headers.get_all('Content-Type',[]))!=1 or self.headers.get_content_type()!=mime:
                return self.reject(415,'unsupported_media')
            lengths = self.headers.get_all('Content-Length',[])
            if not lengths: return self.err(411,'length_required')
            if len(lengths)!=1 or not re.fullmatch(r'[0-9]{1,8}',lengths[0]): return self.reject(400,'invalid_request')
            n = int(lengths[0])
            if n > (MAX_SIDE*MAX_SIDE*3 if op=='frame' else 32): return self.reject(413,'too_large')
            if op=='frame' and n!=w*h*3: return self.reject(422,'invalid_frame')
            live = getattr(engines,'live',None)
            if live is None or (op!='cancel' and not callable(engines.get('scratch'))):
                return self.reject(503,'model_unavailable')
            if not guard.acquire(blocking=False): return self.reject(503,'engine_busy')
            try:
                body=self.rfile.read(n)
                if len(body)!=n: return self.err(400,'invalid_request')
                if op!='frame' and body.strip()!=b'{}': return self.err(400,'invalid_request')
                if op=='start': result=live.start()
                elif op=='frame': result=live.frame(token,seq,w,h,body)
                elif op=='finish': result=live.finish(token)
                else: result=live.cancel(token)
            except LiveError as exc: return self.err(exc.status,exc.code)
            except (TimeoutError,ConnectionError): return self.err(400,'invalid_request')
            finally: guard.release()
            return self.send_json(200,result)

        def do_POST(self):
            if not self.origin_ok(): return self.reject(403,'forbidden',cors=False)
            route,_,query=self.path.partition('?')
            if route in LIVE_ROUTES: return self.live_request(LIVE_ROUTES[route],query)
            if route in EXTRA_ROUTES: return self.extra_request(EXTRA_ROUTES[route])
            if route!=ANALYZE: return self.reject(404,'not_found')
            try: q=parse_qs(query,keep_blank_values=True,max_num_fields=2,strict_parsing=True)
            except ValueError: return self.reject(400,'invalid_request')
            if set(q)!={'model','mode'} or q['mode']!=['single'] or len(q['model'])!=1 or q['model'][0] not in MODELS: return self.reject(400,'invalid_request')
            name=q['model'][0]; mime=self.headers.get_content_type().lower()
            if len(self.headers.get_all('Content-Type',[]))!=1 or mime not in ACCEPTED_TYPES: return self.reject(415,'unsupported_media')
            if self.headers.get_all('Transfer-Encoding',[]): return self.err(400,'invalid_request')
            lengths=self.headers.get_all('Content-Length') or []
            if not lengths: return self.err(411,'length_required')
            if len(lengths)!=1 or not lengths[0].isascii() or not lengths[0].isdecimal() or len(lengths[0])>12: return self.err(400,'invalid_request')
            n=int(lengths[0])
            if n==0: return self.err(422,'invalid_video')
            if n>MAX_BYTES: return self.reject(413,'too_large')
            fn=engines.get(name)
            if not callable(fn): return self.reject(503,'model_unavailable')
            if not guard.acquire(blocking=False): return self.reject(503,'engine_busy')
            try:
                clip=self.rfile.read(n)
                if len(clip)!=n: return self.err(422,'invalid_video')
                result=validate_result(fn(clip,mime))
                if result['model']!=name or len(result['attempts'])>1: raise ValueError('adapter broke single-cycle contract')
            except ClipRejected as exc: return self.err(422,exc.code)
            except (TimeoutError,ConnectionError): return self.err(422,'invalid_video')
            except Exception: return self.err(503,'model_unavailable')
            finally: guard.release()
            return self.send_json(200,result)
        def extra_request(self, exercise):
            # No model selector, local path, or other query parameters are accepted.
            if '?' in self.path: return self.reject(400,'invalid_request')
            mime=self.headers.get_content_type().lower()
            if len(self.headers.get_all('Content-Type',[]))!=1 or mime not in ACCEPTED_TYPES: return self.reject(415,'unsupported_media')
            if self.headers.get_all('Transfer-Encoding',[]): return self.err(400,'invalid_request')
            lengths=self.headers.get_all('Content-Length') or []
            if not lengths: return self.err(411,'length_required')
            if len(lengths)!=1 or not lengths[0].isascii() or not lengths[0].isdecimal() or len(lengths[0])>12: return self.err(400,'invalid_request')
            n=int(lengths[0])
            if n==0: return self.err(422,'invalid_video')
            if n>MAX_BYTES: return self.reject(413,'too_large')
            if not extra.ready(exercise): return self.reject(503,'model_unavailable')
            if not guard.acquire(blocking=False): return self.reject(503,'engine_busy')
            try:
                if not extra.ready(exercise): return self.reject(503,'model_unavailable')
                try:
                    clip=self.rfile.read(n)
                except (TimeoutError,ConnectionError):
                    return self.err(422,'invalid_video')
                if len(clip)!=n: return self.err(422,'invalid_video')
                result=extra.analyze(exercise,clip,mime)
            except ClipRejected as exc: return self.err(422,exc.code)
            except Exception: return self.err(503,'model_unavailable')
            finally: guard.release()
            return self.send_json(200,result)
    return Handler

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port',type=int,default=8765)
    ap.add_argument('--api-only',action='store_true',help='do not serve the bundled Web client')
    ap.add_argument('--artifacts',type=Path,help='directory of local frozen A/B artifacts')
    ap.add_argument('--bundle',type=Path,help='private seriq-bundle-v1 runtime bundle (strict CUDA)')
    ap.add_argument('--freeze',type=Path,default=ROOT/'reports/ml/forearm-v1/freeze.json')
    ap.add_argument('--pose-model',type=Path)
    ap.add_argument('--slim-base',type=Path,help='enable transfer using the pinned SLiM base')
    ap.add_argument('--private-scratch',type=Path,help='ephemeral clips outside Git')
    ap.add_argument('--extra-model-root',type=Path,help='opt-in private E01/E03 model root outside Git')
    ap.add_argument('--ffprobe',default='ffprobe')
    ap.add_argument('--dll-dir',type=Path,help='optional process-local Windows Visual C++ runtime')
    args=ap.parse_args()
    if args.artifacts and (not args.pose_model or not args.private_scratch):
        ap.error('--artifacts requires --pose-model and --private-scratch')
    if args.extra_model_root and not args.private_scratch:
        ap.error('--extra-model-root requires --private-scratch')
    if (args.pose_model or args.slim_base) and not args.artifacts:
        ap.error('model options require --artifacts')
    dll=None
    try:
        if args.dll_dir:
            if os.name!='nt':
                ap.error('--dll-dir is Windows-only')
            dll=os.add_dll_directory(str(args.dll_dir.resolve(strict=True)))
        engines={}
        loader=None
        if args.bundle:
            if not (args.artifacts and args.pose_model and args.private_scratch):
                ap.error('--bundle requires --artifacts, --pose-model and --private-scratch')
            from backend.runtime import load_bundle_engines as loader_bundle
            loader=lambda: loader_bundle(args.bundle,args.artifacts,args.pose_model,args.private_scratch,
                                         args.ffprobe,args.slim_base)
        elif args.artifacts:
            from backend.runtime import load_frozen_engines as loader_frozen
            loader=lambda: loader_frozen(args.artifacts,args.freeze,args.pose_model,args.private_scratch,
                                         args.ffprobe,args.slim_base)
        if loader is not None:
            try:
                engines=loader()
            except Exception as exc:
                # The Web client stays available; analysis answers 503 model_unavailable.
                print(f'models unavailable: {type(exc).__name__}: {exc}',file=__import__('sys').stderr,flush=True)
                engines={}
        extra=ExtraEngines(args.private_scratch,args.ffprobe)
        if args.extra_model_root:
            try:
                extra.register(args.extra_model_root)
            except Exception:
                # Invalid private root/scratch leaves both extras unavailable.
                pass
        handler=make_handler(engines,web_root=None if args.api_only else ROOT,extra=extra)
        with ThreadingHTTPServer(('127.0.0.1',args.port),handler) as server:
            print(f'SerIQ http://127.0.0.1:{server.server_port}/; models: {", ".join(engines) or "none"}',flush=True)
            server.serve_forever()
    finally:
        if dll is not None:
            dll.close()

if __name__=='__main__': main()
