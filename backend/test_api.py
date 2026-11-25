import http.client
import json
from http.server import ThreadingHTTPServer
import threading
import unittest
from pathlib import Path
from backend.api import make_handler

ROOT = Path(__file__).resolve().parents[1]
HEALTH = '/api/v1/health'
ANALYZE = '/api/v1/analyze?model=scratch&mode=single'


class LoopbackAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler({}))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)

    def request_raw(self, method, path, body=None, headers=None):
        c = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)
        try:
            c.request(method,path,body=body,headers=headers or {})
            r=c.getresponse()
            return r.status,r.read(),dict(r.getheaders())
        finally:
            c.close()

    def request(self, method, path, body=None, headers=None):
        status,data,h=self.request_raw(method,path,body,headers)
        return status,json.loads(data),h.get('Cache-Control')

    def test_health_reflects_absent_models_and_post_never_fakes_success(self):
        status,data,cache=self.request('GET','/api/v1/health')
        self.assertEqual((status,data['models'],cache),(200,{'scratch':False,'transfer':False},'no-store'))
        status,data,_=self.request('POST','/api/v1/analyze?model=scratch&mode=single',b'not-a-clinical-record',{'Content-Type':'video/mp4'})
        self.assertEqual((status,data['error']['code']),(503,'model_unavailable'))

    def test_rejects_foreign_origin_and_bad_model(self):
        path='/api/v1/analyze?model=transfer&mode=single'
        status,data,_=self.request('POST',path,b'123',{'Content-Type':'video/mp4','Origin':'https://remote.invalid'})
        self.assertEqual((status,data['error']['code']),(403,'forbidden'))
        status,data,_=self.request('POST',path.replace('transfer','imaginary'),b'123',{'Content-Type':'video/mp4'})
        self.assertEqual((status,data['error']['code']),(400,'invalid_request'))

    def test_rejected_small_uploads_return_json_reliably_on_windows(self):
        for _ in range(30):
            status,data,_=self.request('POST',ANALYZE,b'0'*32768,{'Content-Type':'video/mp4'})
            self.assertEqual((status,data['error']['code']),(503,'model_unavailable'))

    def test_serves_real_web_client_and_never_serves_repo_files(self):
        paths = {'/':'frontend/index.html','/frontend/':'frontend/index.html',
                 '/src/main.js':'frontend/src/main.js','/frontend/src/core/api.js':'frontend/src/core/api.js',
                 '/styles/views/check.css':'frontend/styles/views/check.css',
                 '/assets/brand/qolda-mark.svg':'assets/brand/qolda-mark.svg',
                 '/assets/vendor/three/GLTFLoader.js':'assets/vendor/three/GLTFLoader.js',
                 '/assets/models/mannequin.glb':'assets/models/mannequin.glb'}
        for url,file in paths.items():
            with self.subTest(url=url):
                status,body,headers=self.request_raw('GET',url)
                self.assertEqual((status,body),(200,(ROOT/file).read_bytes()))
                self.assertEqual(headers['X-Content-Type-Options'],'nosniff')
                if file.endswith('.js'):
                    self.assertTrue(headers['Content-Type'].startswith('text/javascript'))
        for url in ('/.git/config','/ml/manifest.json','/assets/%2e%2e/.git/config',
                    '/frontend/src/../../.git/config','/frontend/src/%2e%2e/README.md'):
            self.assertEqual(self.request('GET',url)[0],404)

    def test_preflight_is_scoped_to_allowed_origin_method_and_header(self):
        headers={'Origin':'http://127.0.0.1:5173','Access-Control-Request-Method':'POST',
                 'Access-Control-Request-Headers':'content-type'}
        status,_,h=self.request_raw('OPTIONS',ANALYZE,headers=headers)
        self.assertIn(status,(200,204))
        self.assertEqual(h.get('Access-Control-Allow-Origin'),headers['Origin'])
        docs_origin='http://127.0.0.1:8080'
        status,_,h=self.request_raw('OPTIONS',ANALYZE,headers={**headers,'Origin':docs_origin})
        self.assertEqual((status,h.get('Access-Control-Allow-Origin')),(204,docs_origin))
        for bad in ({'Origin':'https://remote.invalid'},{'Access-Control-Request-Method':'PUT'},
                    {'Access-Control-Request-Headers':'content-type,x-external'}):
            status,_,h=self.request_raw('OPTIONS',ANALYZE,headers={**headers,**bad})
            self.assertEqual(status,403)
            self.assertNotIn('Access-Control-Allow-Origin',h)

    def test_rejects_rebound_host_and_more_than_one_query_value(self):
        self.assertEqual(self.request('GET',HEALTH,headers={'Host':'elsewhere.invalid'})[0],403)
        status,data,_=self.request('POST',ANALYZE+'&model=transfer',b'123',{'Content-Type':'video/mp4'})
        self.assertEqual((status,data['error']['code']),(400,'invalid_request'))
