"""Generate immutable content hashes after any batch. Excludes generated hashes themselves."""
from pathlib import Path
import hashlib,csv,os
root=Path(__file__).resolve().parents[1]
# Compiled Python/Numba caches are machine-specific regenerable artifacts, not research data.
def research_file(p):
    return (p.is_file() and p.name not in ('SHA256SUMS.txt','MANIFEST.csv','integrity_report.md')
            and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.nbi','.nbc')
            and p.name != '.lock' and '.tmp.' not in p.name)
paths=sorted(p for p in root.rglob('*') if research_file(p))
rows=[]
for p in paths:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    rows.append((str(p.relative_to(root)).replace('\\','/'),p.stat().st_size,h.hexdigest()))
with (root/'MANIFEST.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['relative_path','size_bytes','sha256']);w.writerows(rows)
(root/'SHA256SUMS.txt').write_text(''.join(f'{h}  {rel}\n' for rel,size,h in rows))
print('MANIFEST',len(rows),'files','bytes',sum(size for _,size,_ in rows))
