"""Deterministic single-writer shard merge. No live workers permitted.

--partial-checkpoint writes method_records_merged_partial.csv ONLY.
--final requires exactly 48,000 distinct valid method records and 6000 pairs;
        only then atomically replaces method_records.csv.
"""
from __future__ import annotations
import argparse,collections,csv,json
from pathlib import Path
from shard_common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial-checkpoint',action='store_true');ap.add_argument('--final',action='store_true')
    args=ap.parse_args()
    if args.final==args.partial_checkpoint:ap.error('select exactly one of --partial-checkpoint or --final')
    if any((ROOT/'07_tidy_results/shards').rglob('.lock')):raise SystemExit('REFUSE MERGE: active or stale shard locks found')
    thresholds=freeze_thresholds();merged={};sources={}
    gl=ROOT/'07_tidy_results/method_records.csv'
    if gl.exists():
        with gl.open(newline='') as f:
            for r in csv.DictReader(f):
                key=(r['trajectory_id'],r['method'])
                if key in merged:raise ValueError('duplicate pre-merge global '+str(key))
                merged[key]=r;sources[key]='previous_global'
    journals=list((ROOT/'07_tidy_results/shards').glob('*/rows/*.json'))
    for p in journals:
        r=json.loads(p.read_text());key=(r['trajectory_id'],r['method'])
        if key in merged:
            old=merged[key]
            for field in ('trajectory_id','method','distribution','scenario','replicate','seed','threshold','first_alarm','prefault_alarm','detected_after_change','delay','numeric_status'):
                if str(old[field])!=str(r[field]):
                    # numeric lexical distinctions: string '0' vs '0.0' are avoidable, compare values if appropriate.
                    try:
                        if float(old[field])==float(r[field]):continue
                    except (ValueError,TypeError):pass
                    raise ValueError(f'CONFLICT field {field}: {key}: {old[field]} vs {r[field]}')
        merged[key]=r;sources[key]=str(p.relative_to(ROOT))
    for key,r in merged.items():
        d=r['distribution'];sc=r['scenario'];rep=int(r['replicate']);m=r['method']
        validate_row(r,trace_path(d,sc,rep,m),thresholds)
    ordered=[]
    for d in DISTS:
        for sc in SCENES:
            for rep in range(1000 if sc=='health' else 500):
                for m in METHODS:
                    key=(trajectory_id(d,sc,rep),m)
                    if key in merged:ordered.append(merged[key])
    if len(ordered)!=len(merged):raise ValueError('unrecognized keys outside fixed protocol')
    valid=sum(r['numeric_status'] in VALID_STATUSES for r in ordered)
    completed=sum(all((trajectory_id(d,sc,rep),m) in merged for m in METHODS)
        for d in DISTS for sc in SCENES for rep in range(1000 if sc=='health' else 500))
    if args.final and not (len(ordered)==48000 and valid==48000 and completed==6000):
        raise SystemExit(f'REFUSE FINAL: method rows={len(ordered)}, valid={valid}, paired={completed}, expected=48000,48000,6000')
    target=ROOT/'07_tidy_results'/('method_records.csv' if args.final else 'method_records_merged_partial.csv')
    atomic_csv(target,ordered)
    atomic_json(ROOT/'10_logs/merge_last.json',dict(output=str(target.relative_to(ROOT)),method_rows=len(ordered),valid_rows=valid,complete_trajectories=completed,journals=len(journals),final=args.final))
    print('MERGE',('FINAL' if args.final else 'PARTIAL'),len(ordered),'VALID',valid,'COMPLETE_PAIRED',completed,'JOURNALS',len(journals),'OUTPUT',str(target))
if __name__=='__main__':main()
