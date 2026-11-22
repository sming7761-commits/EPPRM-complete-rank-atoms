"""Create an isolated runtime; never overwrite the frozen formal snapshot."""
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'industrial/runtime_package'
DEST.mkdir(parents=True,exist_ok=True)
shutil.copytree(ROOT/'formal/01_source',DEST/'01_source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbc','*.nbi'))
for name in ['full_domain_cctm_atoms_fast.py','hybrid_exact_key.py','tie_safe_rank_geometry.py']:
    shutil.copy2(ROOT/'industrial/backends'/name,DEST/'01_source/cctm_profile'/name)
shutil.copy2(ROOT/'industrial/backends/full_domain_prm_atoms_fast.py',DEST/'01_source/epprm/full_domain_prm_atoms_fast.py')
print('INDUSTRIAL_RUNTIME_READY',DEST)
