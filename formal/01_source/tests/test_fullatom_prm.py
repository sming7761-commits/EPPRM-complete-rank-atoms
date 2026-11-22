"""Deterministic smoke/regression of complete root-atom PRM vs independent rational backend."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
for x in ['epprm','cctm_profile']:
 sys.path.insert(0,str(ROOT/'01_source'/x))
from full_domain_prm_atoms_fast import stream_fullatom
from generic_profile import stream,prm_backend

def main():
 rng=np.random.default_rng(20261109)
 for j in range(50):
  n=5+j%3;T=6+j%4
  cy=rng.normal(size=n);cx=rng.normal(size=n)
  oy=rng.normal(size=T);ox=rng.normal(size=T)
  a=stream_fullatom(cy,cx,oy,ox)
  b=stream(cy,cx,oy,ox,prm_backend(),max_atoms=10000)
  error=float(np.max(np.abs(a['profile_path']-np.asarray(b['profile'],dtype=float))))
  assert error<=2e-12,(j,error)
  assert np.array_equal(a['atom_counts'],b['atoms']),(j,a['atom_counts'],b['atoms'])
 print('FULLATOM_PRM_RATIONAL_REGRESSION 50/50 PASS')
if __name__=='__main__':main()
