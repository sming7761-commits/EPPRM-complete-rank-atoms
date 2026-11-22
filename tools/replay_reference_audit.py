"""Replay original small-input checks in a temporary output directory."""
from pathlib import Path
import sys,tempfile,shutil,importlib

ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'reference'
sys.path.insert(0,str(REF))
audit=importlib.import_module('audit_supplement')
with tempfile.TemporaryDirectory(prefix='epprm_reference_') as tmp:
    dest=Path(tmp)
    for name in ['legacy_cases.json','historical_random_48_input.json','source_witness.csv']:
        shutil.copy2(REF/name,dest/name)
    audit.ROOT=dest
    audit.main()
