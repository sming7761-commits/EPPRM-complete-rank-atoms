"""I/O and validation only. Frozen scientific implementations are imported unmodified.

A shard owns one 25-trajectory partition. Each method is committed as an atomic
NPZ followed by an atomic JSON journal entry; no worker writes the global CSV.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'12_reproduction'))
import reproduce_main as core
DISTS=tuple(core.DISTS)
METHODS=tuple(core.METHODS)
SCENES=('health','fault')
FIELDS=('trajectory_id','method','distribution','scenario','replicate','seed',
        'threshold','threshold_source','first_alarm','prefault_alarm','detected_after_change',
        'delay','censored','max_statistic','runtime_seconds','peak_memory_mb',
        'atom_count_final','exact_fallback_count','numeric_status','error_message')
VALID_STATUSES=('COMPLETE','GENERAL_POSITION_FLOAT_NOT_FORMAL_CERTIFIED',
                'GENERAL_POSITION_FLOAT_NOT_FORMALLY_CERTIFIED',
                'GENERAL_POSITION_FLOAT_VERIFIED_BY_SMALL_REFERENCE',
                'SMALL_INPUT_EXACT_BINARY_RATIONAL_FALLBACK',
                'EXACT_RANK_MULTIPRECISION_POINT_WITNESS_NONALARM')
THRESHOLD_SOURCE={
    'EP_PRM_736':'MARGINAL_ANYTIME_PFA',
    'Profiled_CCTM_fair':'CCTM_ALPHA_0.04_DELTA_0.01',
    'Old_K8':'LEGACY_K8_SYMMETRIC_NOISE_ONLY',
    'Oracle_PRM':'MARGINAL_ANYTIME_PFA',
}


def shard_id(dist,scene,start,stop):return f'{dist}_{scene}_{start:04d}_{stop-1:04d}'
def shard_dir(sid):return ROOT/'07_tidy_results'/'shards'/sid
def journal_path(sid,rep,m):return shard_dir(sid)/'rows'/f'{rep:04d}_{m}.json'
def trace_path(dist,scene,rep,m):return core.trace_path(dist,scene,rep,m)
def trajectory_id(dist,scene,rep):return f'FORMAL_{dist}_{scene}_{rep:04d}'


def freeze_thresholds():
    q=json.loads((ROOT/'03_configs/frozen_thresholds.json').read_text())
    if set(q)!=set(DISTS):raise AssertionError('threshold distributions missing')
    for d,values in q.items():
        if set(values)!=set(METHODS[2:6]):raise AssertionError('threshold methods mismatch '+d)
        if not all(np.isfinite(float(v)) for v in values.values()):raise AssertionError('threshold nonfinite')
    return q


def method_threshold(dist,m,thresholds):
    return float(20 if m in ('EP_PRM_736','Oracle_PRM') else 25 if m in ('Profiled_CCTM_fair','Old_K8') else thresholds[dist][m])


def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,allow_nan=True,indent=2)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def atomic_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,extrasaction='ignore');w.writeheader();w.writerows(rows)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def filehash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def trace_arrays(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def validate_trace(path,thr,require_time=True):
    if not Path(path).exists():return None
    a=trace_arrays(path)
    needed=('value','running_max','crossing','computed','atom_count','worst_parameter','threshold','status')
    if any(k not in a for k in needed):raise ValueError('missing trace arrays '+str(path))
    v=a['value'];cross=a['crossing'];mask=a['computed'];run=a['running_max']
    for k in ('value','running_max','crossing','computed','atom_count','worst_parameter'):
        if a[k].shape!=(601,):raise ValueError(f'{k} not 601: {path}')
    if require_time:
        if 'time_index' not in a or not np.array_equal(a['time_index'],np.arange(601,dtype=np.int32)):
            raise ValueError('missing or incorrect explicit 0..600 time index '+str(path))
    if float(a['threshold'])!=float(thr):raise ValueError('threshold mismatch '+str(path))
    if not np.array_equal((np.isfinite(v)).astype(np.uint8),mask.astype(np.uint8)):
        raise ValueError('computed mask does not match finite values '+str(path))
    status=str(a['status'])
    if status in VALID_STATUSES:
        if not np.isfinite(v).all():raise ValueError('nominally successful trace is not fully finite '+str(path))
        if not np.allclose(run,np.maximum.accumulate(v),rtol=1e-13,atol=1e-13):
            raise ValueError('running maxima mismatch '+str(path))
        if not np.array_equal(cross,(v>=thr).astype(np.int8)):
            raise ValueError('crossing mismatch '+str(path))
        hits=np.flatnonzero(v[1:]>=thr);first=int(hits[0]+1) if len(hits) else 0
    elif status in ('UNKNOWN','ERROR'):
        first=-1
    else:raise ValueError('unknown numerical status '+status+' '+str(path))
    return a,first


def row_from_trace(a,first,dist,scene,rep,m,thr,thresholds):
    vals=a['value'];valid=np.isfinite(vals)
    return dict(trajectory_id=trajectory_id(dist,scene,rep),method=m,distribution=dist,scenario=scene,
      replicate=rep,seed=core.seed_for(dist,scene,rep),threshold=thr,
      threshold_source=THRESHOLD_SOURCE.get(m,'INDEPENDENT_EMPIRICAL_PFA'),
      first_alarm=first,prefault_alarm=int(0<first<core.TAU),
      detected_after_change=int(scene=='fault' and first>=core.TAU),
      delay=first-core.TAU if scene=='fault' and first>=core.TAU else 244,
      censored=int(not(scene=='fault' and first>=core.TAU)),
      max_statistic=float(np.max(vals)) if valid.all() else float(np.nanmax(vals)) if valid.any() else float('nan'),
      runtime_seconds=float(a['runtime_seconds']) if 'runtime_seconds' in a else float('nan'),
      peak_memory_mb=float(a['peak_memory_mb']) if 'peak_memory_mb' in a else float('nan'),
      atom_count_final=int(a['atom_count'][-1]) if int(a['atom_count'][-1])>=0 else '',
      exact_fallback_count=int(a['exact_fallback_count']) if 'exact_fallback_count' in a else 0,
      numeric_status=str(a['status']),error_message=str(a['error_message']) if 'error_message' in a else '')


def validate_row(row,trace_path,thresholds,require_time=True):
    d=row['distribution'];sc=row['scenario'];rep=int(row['replicate']);m=row['method']
    if (d not in DISTS or sc not in SCENES or m not in METHODS):raise ValueError('unknown method or group')
    if str(row['trajectory_id'])!=trajectory_id(d,sc,rep):raise ValueError('trajectory_id wrong')
    if int(row['seed'])!=core.seed_for(d,sc,rep):raise ValueError('seed wrong')
    thr=method_threshold(d,m,thresholds)
    if float(row['threshold'])!=thr:raise ValueError('threshold row mismatch')
    ans=validate_trace(trace_path,thr,require_time)
    if ans is None:raise ValueError('missing trace')
    a,first=ans
    if int(row['first_alarm'])!=first:raise ValueError('alarm time mismatch')
    if int(row['prefault_alarm'])!=int(0<first<core.TAU):raise ValueError('prefault mismatch')
    if int(row['detected_after_change'])!=int(sc=='fault' and first>=core.TAU):raise ValueError('detection mismatch')
    if int(row['delay'])!=(first-core.TAU if sc=='fault' and first>=core.TAU else 244):raise ValueError('delay mismatch')
    if str(row['numeric_status'])!=str(a['status']):raise ValueError('numeric status mismatch')
    return a,first


def semantic_hash(arrs):
    h=hashlib.sha256()
    for key in ('value','running_max','crossing','computed','atom_count','worst_parameter'):
        arr=np.asarray(arrs[key]);h.update(key.encode()+str(arr.shape).encode()+str(arr.dtype).encode());h.update(arr.tobytes())
    h.update(np.asarray(arrs['threshold']).tobytes());return h.hexdigest()


def all_shards():
    for d in DISTS:
        for sc in SCENES:
            maximum=1000 if sc=='health' else 500
            for start in range(0,maximum,25):yield d,sc,start,min(start+25,maximum)


def frozen_raw_hashes():
    p=ROOT/'09_algorithm_audit/FROZEN_RAW_SHA256SUMS.txt'
    if not p.exists():raise FileNotFoundError('Frozen original raw hash snapshot missing')
    vals={}
    for line in p.read_text().splitlines():
        if not line.strip():continue
        sha,rel=line.split('  ',1);vals[rel]=sha
    if len(vals)!=6000:raise ValueError(f'expected 6000 frozen SHA, got {len(vals)}')
    return vals


def verify_scientific_frozen():
    fp=ROOT/'09_algorithm_audit/FROZEN_SCIENTIFIC_SHA256SUMS.txt'
    if not fp.exists():raise FileNotFoundError('Missing frozen scientific code/threshold SHA snapshot')
    for line in fp.read_text().splitlines():
        if not line.strip():continue
        h,rel=line.split('  ',1)
        p=ROOT/rel
        if not p.is_file() or filehash(p)!=h:
            raise AssertionError('FROZEN SCIENTIFIC CODE / THRESHOLD MODIFIED: '+rel)
    return True
