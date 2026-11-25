"""Inspect a new private output of desktop/Build-Windows.ps1 without launching its binaries."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
from zipfile import ZipFile

DLL = 'WebView2Loader.dll'
NATIVE_DLL = 'runtimes/win-x64/native/' + DLL
SUMS = 'SHA256SUMS.txt'
ZIP = 'qolda-windows-win-x64.zip'
PAYLOAD = {'Qolda.exe', DLL, NATIVE_DLL, 'README.md', 'qolda.example.json',
           'licenses/dotnet-LICENSE.txt', 'licenses/dotnet-THIRD-PARTY-NOTICES.txt',
           'licenses/windows-desktop-LICENSE.txt', 'licenses/webview2-LICENSE.txt', 'licenses/webview2-NOTICE.txt'}


def require(value, message):
    if not value:
        raise ValueError(message)


def disk(path):
    with path.open('rb') as f:
        return {'bytes':path.stat().st_size, 'sha256':hashlib.file_digest(f,'sha256').hexdigest()}


def check_pe(path):
    with path.open('rb') as f:
        head = f.read(64)
        require(len(head)==64 and head[:2]==b'MZ','invalid DOS header')
        pos, = struct.unpack_from('<I',head,60)
        require(64 <= pos <= path.stat().st_size-26, 'invalid PE offset')
        f.seek(pos); pe=f.read(26)
        require(pe[:4]==b'PE\0\0' and struct.unpack_from('<H',pe,4)[0]==0x8664
                and struct.unpack_from('<H',pe,24)[0]==0x20b, 'not AMD64 PE32+')


def verify(directory):
    directory=directory.resolve(strict=True)
    allowed=PAYLOAD|{SUMS, ZIP}
    # Only examine the private release directory; reject unknown files before reading their contents.
    actual=set()
    for p in directory.rglob('*'):
        require(p.resolve()==p and not p.is_symlink(), 'redirected release member')
        if not p.is_dir():actual.add(p.relative_to(directory).as_posix())
    require(actual==allowed,'unexpected or missing release files; verify a fresh output, not a user profile')
    checks={}
    for line in (directory/SUMS).read_text(encoding='ascii').splitlines():
        match=re.fullmatch(r'([0-9a-f]{64})  (.+)',line)
        require(match is not None,'invalid SHA256SUMS line')
        digest,name=match.groups();require(name in PAYLOAD and name not in checks,'unknown or duplicate checksum path');checks[name]=digest
    require(set(checks)==PAYLOAD,'incomplete checksum file')
    records={name:disk(directory/name) for name in sorted(PAYLOAD|{SUMS})}
    require(all(records[n]['sha256']==v for n,v in checks.items()),'release file differs from SHA256SUMS')
    for name in ('Qolda.exe',DLL,NATIVE_DLL):check_pe(directory/name)
    require(records[DLL]==records[NATIVE_DLL],'WebView2 loader copies differ')
    with ZipFile(directory/ZIP) as z:
        require(len(z.infolist())==len(records) and set(z.namelist())=={'windows/'+n for n in records},'ZIP inventory differs')
        for n,r in records.items():
            entry=z.getinfo('windows/'+n);require(entry.file_size==r['bytes'],'ZIP size differs')
            with z.open(entry) as src:require(hashlib.file_digest(src,'sha256').hexdigest()==r['sha256'],'ZIP member differs')
    return {'schema':'qolda-windows-shell-readback-v1','exe':records['Qolda.exe'],'architecture':'AMD64 PE32+',
            'webview2_loader':records[DLL],'manifest':records[SUMS],'zip':disk(directory/ZIP),'verified_zip_members':len(records),
            'launched_by_this_check':False,'engine_verified_by_this_check':False,'clip_verified_by_this_check':False}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--windows',required=True,type=Path)
    print(json.dumps(verify(ap.parse_args().windows),indent=2))
