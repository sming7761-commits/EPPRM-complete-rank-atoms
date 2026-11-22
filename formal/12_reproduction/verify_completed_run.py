"""Strict final 6000/48000 verification with frozen raw SHA and per-step replay.

For an unfinished study use --partial to inspect integrity without claiming PASS_COMPLETE.
"""
from __future__ import annotations
import argparse,collections,csv,hashlib,json,sys
import numpy as np
from shard_common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');ap.add_argument('--deep',action='store_true')
    args=ap.parse_args()
    errors=[];counts=collections.Counter();th=freeze_thresholds()
    verify_scientific_frozen()
    hashbaseline=frozen_raw_hashes();counts['frozen_raw_hashes']=len(hashbaseline)
    # Each frozen source NPZ is compared to the original archive's hashes, not newly generated hashes.
    for rel,h in hashbaseline.items():
        fp=ROOT/rel
        if not fp.exists() or filehash(fp)!=h:errors.append('RAW_SHA_MISMATCH '+rel)
        else:counts['verified_raw_hashes']+=1
    seeds={}
    for name in ('pilot','calibration','formal_test'):
        with (ROOT/'04_seed_manifests'/f'{name}_seeds.csv').open(newline='') as f:
            a=list(csv.DictReader(f))
        seeds[name]={int(q['seed']) for q in a}
        if len(seeds[name])!=len(a):errors.append('DUPLICATE_'+name+'_SEEDS')
    if any(seeds[a]&seeds[b] for a,b in [('pilot','calibration'),('pilot','formal_test'),('calibration','formal_test')]):errors.append('SEED_OVERLAP')
    gl=ROOT/'07_tidy_results/method_records.csv'
    rs=[]
    if gl.exists():
        with gl.open(newline='') as f:rs=list(csv.DictReader(f))
    # Shard journals are authoritative during resume, not shared global CSV.
    keys={}
    for r in rs:keys[(r['trajectory_id'],r['method'])]=r
    for p in (ROOT/'07_tidy_results/shards').glob('*/rows/*.json'):
        r=json.loads(p.read_text());keys[(r['trajectory_id'],r['method'])]=r
    seen=collections.defaultdict(set)
    for key,r in keys.items():
        d=r['distribution'];sc=r['scenario'];rep=int(r['replicate']);m=r['method']
        seen[r['trajectory_id']].add(m)
        if int(r['seed']) not in seeds['formal_test']:errors.append('SEED_NOT_FORMAL '+str(key))
        try:
            a,first=validate_row(r,trace_path(d,sc,rep,m),th)
            counts['verified_method_traces']+=1
            if r['numeric_status'] in ('UNKNOWN','ERROR'):counts[r['numeric_status']]+=1
        except Exception as e:
            errors.append('TRACE_INVALID '+str(key)+' '+repr(e))
    counts['method_records']=len(keys)
    counts['paired_trajectories']=sum(ms==set(METHODS) for ms in seen.values())
    counts['total_distinct_trajectories_seen']=len(seen)
    for d in DISTS:
        for sc in SCENES:
            cnt=sum(trajectory_id(d,sc,i) in seen and seen[trajectory_id(d,sc,i)]==set(METHODS) for i in range(1000 if sc=='health' else 500))
            counts[f'paired_{d}_{sc}']=cnt
    for d in DISTS:
        fp=ROOT/f'05_raw_data/calibration/max_{d}.csv'
        with fp.open(newline='') as f:q=list(csv.DictReader(f))
        if len(q)!=20000:errors.append('CALIBRATION_COUNT '+d)
        counts['calibration_method_maxima']+=len(q)
        if any(int(x['seed']) not in seeds['calibration'] for x in q):errors.append('CALIBRATION_LEAK '+d)
    if not args.partial:
        if counts['method_records']!=48000:errors.append('EXPECTED_48000_METHOD_ROWS')
        if counts['paired_trajectories']!=6000:errors.append('EXPECTED_6000_TRAJECTORIES')
        if counts['verified_method_traces']!=48000:errors.append('EXPECTED_48000_TRACES')
        if counts['UNKNOWN'] or counts['ERROR']:errors.append('UNRESOLVED_NUMERIC_FAILURES')
    status='PASS_COMPLETE' if not errors and not args.partial else 'PASS_PARTIAL_STRUCTURE_ONLY' if not errors else 'FAIL'
    report=dict(status=status,counts=dict(counts),errors=errors[:100],error_count=len(errors),
        note='PASS_PARTIAL_STRUCTURE_ONLY explicitly does NOT mean formal experiment complete')
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if not args.partial and not errors:
        # Companion legacy verifier checks SHA256SUMS/MANIFEST coverage independently.
        print('NOTE: run analyze_main.py, build_manifest.py, verify_package.py --deep after freezing all writers')
    return 0 if not errors else 1
if __name__=='__main__':sys.exit(main())
