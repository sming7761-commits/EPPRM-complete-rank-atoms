"""Hash and frozen-count checks; does not rerun detection experiments."""
from pathlib import Path
import hashlib,json,csv,collections

ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'FILE_SHA256.json').read_text(encoding='utf-8'))
for name,expected in manifest.items():
    p=ROOT/name
    if not p.is_file():raise RuntimeError('missing '+name)
    if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise RuntimeError('hash mismatch '+name)
with (ROOT/'formal/07_tidy_results/method_records.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
assert len(rows)==48000
assert len({(r['trajectory_id'],r['method']) for r in rows})==48000
assert not any(r['numeric_status'] in ['UNKNOWN','ERROR'] for r in rows)
with (ROOT/'industrial/REALDATA_GATE_F3/case_method_results.csv').open(newline='',encoding='utf-8') as f:industrial=list(csv.DictReader(f))
assert len(industrial)==72
assert not any(r['numeric_status'].startswith(('UNKNOWN','ERROR')) for r in industrial)
print('REPOSITORY_CHECK_PASS',len(manifest),'files;',len(rows),'formal method records;',len(industrial),'industrial method records')
print('Full production traces are a separate release asset; this check does not replace their deep verification.')
