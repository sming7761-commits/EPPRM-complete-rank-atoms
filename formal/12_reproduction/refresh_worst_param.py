"""Repair argmin diagnostic only after changing interval representative selection; alarm/wealth unchanged."""
from reproduce_main import ensure_raw,arrays_to_xy,save_trace
from full_domain_prm_atoms_fast import stream_fullatom
import sys,numpy as np
for d,sc in [('gaussian','health'),('gaussian','fault'),('t5','health'),('t5','fault'),('exp_centered','health'),('exp_centered','fault'),('skew_mixture','health'),('skew_mixture','fault')]:
 if len(sys.argv)>1 and d!=sys.argv[1]:continue
 data=ensure_raw(d,sc,0);cy,cx,oy,ox=arrays_to_xy(data)
 old=np.load(__import__('pathlib').Path(__file__).resolve().parents[1]/'06_method_traces'/d/sc/'0000_EP_PRM_736.npz')
 z=stream_fullatom(cy,cx,oy,ox)
 assert np.array_equal(z['profile_path'],old['value']) or np.allclose(z['profile_path'],old['value'],rtol=1e-14,atol=1e-14)
 save_trace(d,sc,0,'EP_PRM_736',z['profile_path'],20.,z['status'],z['atom_counts'],z['worst_parameter'])
 print('REFRESHED',d,sc,'same path',np.max(np.abs(z['profile_path']-old['value'])),flush=True)
