"""Local browser smoke for the real SerIQ Web + backend refusal path (no model needed)."""
from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import tempfile
from threading import Thread
from time import monotonic, sleep
from urllib.request import urlopen

from backend.api import make_handler


def until(check, seconds=12):
    end = monotonic() + seconds
    while monotonic() < end:
        result = check()
        if result:
            return result
        sleep(.08)
    raise TimeoutError('browser smoke step did not become ready')


def socket_url(debug_file):
    if not debug_file.is_file():
        return None
    port = debug_file.read_text('utf-8').splitlines()[0]
    try:
        with urlopen(f'http://127.0.0.1:{port}/json/list', timeout=2) as response:
            tabs = json.load(response)
    except OSError:
        return None
    return next((t['webSocketDebuggerUrl'] for t in tabs if t['type'] == 'page'), None)


def exercise(chrome, ffmpeg, scratch):
    from websockets.sync.client import connect

    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='seriq-web-', dir=scratch, ignore_cleanup_errors=True) as tmp:
        temp = Path(tmp)
        clip = temp / 'synthetic-no-person.mp4'
        subprocess.run([ffmpeg, '-nostdin', '-v', 'error', '-f', 'lavfi', '-i', 'color=teal:s=160x120:r=24:d=1',
                        '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(clip)],
                       check=True, timeout=15)
        profile = temp / 'browser'
        with ThreadingHTTPServer(('127.0.0.1', 0), make_handler({})) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_port
            base = f'http://127.0.0.1:{port}'
            browser = subprocess.Popen([str(chrome), '--headless=new', '--disable-background-networking',
                                        '--disable-component-update', '--disable-default-apps', '--no-first-run',
                                        '--no-default-browser-check', '--remote-debugging-address=127.0.0.1',
                                        '--remote-debugging-port=0', f'--user-data-dir={profile}', 'about:blank'],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                with connect(until(lambda: socket_url(profile / 'DevToolsActivePort')), proxy=None,
                             open_timeout=5, close_timeout=1) as ws:
                    seq = 0
                    responses = []

                    def call(method, **params):
                        nonlocal seq
                        seq += 1
                        ws.send(json.dumps(dict(id=seq, method=method, params=params)))
                        while True:
                            message = json.loads(ws.recv(timeout=12))
                            if message.get('method') == 'Network.responseReceived':
                                r = message['params']['response']
                                responses.append((r['url'], r['status']))
                            if message.get('id') == seq:
                                if 'error' in message:
                                    raise RuntimeError(f'CDP {method}: {message["error"]["message"]}')
                                return message.get('result', {})

                    def evaluate(expression):
                        result = call('Runtime.evaluate', expression=expression, returnByValue=True, awaitPromise=True)
                        if 'exceptionDetails' in result:
                            raise RuntimeError('unexpected JS failure during local smoke')
                        return result['result'].get('value')

                    call('Page.enable')
                    call('Network.enable')
                    call('Page.addScriptToEvaluateOnNewDocument', source=(
                        'window.smokeErrors=[]; addEventListener("error",e=>{if(e.message)smokeErrors.push(e.message)});'
                        'addEventListener("unhandledrejection",e=>smokeErrors.push(String(e.reason)));'))
                    call('Page.navigate', url=base + '/')
                    until(lambda: evaluate('document.readyState === "complete" && !!document.querySelector(".w-setup button[type=submit]")'))
                    evaluate('document.querySelector(".w-setup button[type=submit]").click(); true')
                    until(lambda: evaluate('!!document.querySelector("nav a[href*=cabinet]")'))
                    evaluate('document.querySelector("nav a[href*=cabinet]").click(); true')
                    until(lambda: evaluate('!!document.querySelector("#view input[type=url]")'))
                    evaluate(f'document.querySelector("#view input[type=url]").value={json.dumps(base)};'
                             'document.querySelector("#view input[type=url]").dispatchEvent(new Event("change")); true')
                    until(lambda: evaluate('document.querySelector(".engine")?.dataset.state === "nomodel"'))
                    evaluate('document.querySelector("nav a[href*=check]").click(); true')
                    until(lambda: evaluate('!!document.querySelector("#clip")'))
                    evaluate('document.querySelector(".c-steps li:nth-child(2) button").click(); true')
                    node = call('DOM.querySelector', nodeId=call('DOM.getDocument')['root']['nodeId'], selector='#clip')
                    call('DOM.setFileInputFiles', nodeId=node['nodeId'], files=[str(clip)])
                    evaluate('document.querySelector(".c-steps li:nth-child(3) button").click(); true')
                    evaluate('document.querySelector(".c-panel:not([hidden]) .consent input[type=checkbox]").click(); true')
                    until(lambda: evaluate('!document.querySelector(".c-panel:not([hidden]) button.btn--cta").disabled'))
                    evaluate('document.querySelector(".c-panel:not([hidden]) button.btn--cta").click(); true')
                    until(lambda: evaluate('document.querySelector(".c-verdict").dataset.state === "error"'))
                    observed = evaluate('({text:document.querySelector(".c-verdict").textContent,'
                                        'state:document.querySelector(".c-verdict").dataset.state,errors:smokeErrors})')
                    count = evaluate('import("/src/core/store.js").then(m=>m.store.state.data.attempts.length)')
                    assert observed['state'] == 'error' and observed['text'] and not observed['errors'], observed
                    assert count == 0, 'a failed upload must not add an attempt'
                    doc_status = next(status for url, status in responses if url == base + '/')
                    module_status = next(status for url, status in responses if url == base + '/src/main.js')
                    upload_status = next(status for url, status in responses if url.startswith(base + '/api/v1/analyze?'))
                    health_status = next(status for url, status in responses if url == base + '/api/v1/health')
                    stylesheet_status = next(status for url, status in responses if url == base + '/styles/tokens.css')
                    external = [url for url, _ in responses if url.startswith(('http:', 'https:'))
                                and not url.startswith(base + '/')]
                    assert (doc_status, module_status, stylesheet_status, health_status, upload_status) == (200, 200, 200, 200, 503)
                    assert not external, 'page received an off-origin HTTP response'
                    print(json.dumps({'browser': 'Chromium', 'synthetic_clip_seconds': 1,
                                      'web_document': doc_status, 'es_module': module_status, 'health': health_status, 'upload': upload_status,
                                      'stylesheet': stylesheet_status,
                                      'ui_state': observed['state'], 'ui_message': observed['text'],
                                      'external_http_responses': len(external), 'js_exceptions': len(observed['errors']),
                                      'saved_attempts': count, 'classification_tested': False}, ensure_ascii=False))
                    ws.send(json.dumps({'id': seq + 1, 'method': 'Browser.close'}))
                browser.wait(timeout=6)
            finally:
                if browser.poll() is None:
                    browser.terminate()
                    browser.wait(timeout=6)
                server.shutdown()
                thread.join(timeout=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chrome', type=Path, required=True)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--scratch', type=Path, required=True)
    args = parser.parse_args()
    exercise(args.chrome, args.ffmpeg, args.scratch)


if __name__ == '__main__':
    main()
