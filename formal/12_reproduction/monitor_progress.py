"""Read-only counts for safe sharded execution. No scientific interim analysis."""
from __future__ import annotations
import csv,collections,json,os,time
from pathlib import Path
from shard_common import *

def main():
    rows={};base=ROOT/'07_tidy_results/method_records.csv'
    if base.exists():
        with base.open(newline='') as f:
            for r in csv.DictReader(f):rows[(r['trajectory_id'],r['method'])]=r
    journal_dir=ROOT/'07_tidy_results/shards'
    journals=list(journal_dir.glob('*/rows/*.json'))
    for fp in journals:
        r=json.loads(fp.read_text());rows[(r['trajectory_id'],r['method'])]=r
    count=collections.Counter(r['numeric_status'] for r in rows.values())
    groups=collections.defaultdict(set)
    for (id,m) in rows:groups[id].add(m)
    done=sum(ms==set(METHODS) for ms in groups.values())
    pending=6000-done;locks=list(journal_dir.rglob('.lock'))
    completed_shards=sum(1 for p in journal_dir.glob('*/shard_status.json') if (lambda q:q.get('journal_records')==q.get('expected'))(json.loads(p.read_text())))
    trace_count=sum(1 for _ in (ROOT/'06_method_traces').rglob('*.npz'))
    # Archive extraction rewrites filesystem mtimes, which cannot be used for a
    # scientifically honest throughput metric. Only real executor wall timing counts.
    recent_inv=ROOT/'10_logs/last_sharded_invocation.json'
    throughput_recent=None
    if recent_inv.exists():
        inv=json.loads(recent_inv.read_text())
        wall=float(inv.get('elapsed_wall_seconds',0))
        n_methods=int(inv.get('new_method_records',0))
        if wall>0 and n_methods>=8:
            throughput_recent=round(3600*n_methods/(8*wall),3)
    known_sec=sum(float(r['runtime_seconds']) for r in rows.values() if str(r.get('runtime_seconds','')).lower() not in ('','nan','none'))
    mean_sec=known_sec/max(1,sum(str(r.get('runtime_seconds','')).lower() not in ('','nan','none') for r in rows.values()))
    disk_mib=sum(p.stat().st_size for p in (ROOT/'06_method_traces').rglob('*.npz'))/1048576
    report=dict(trajectories_total=6000,trajectories_complete=done,trajectories_remaining=pending,
        method_records=len(rows),method_records_remaining=48000-len(rows),
        COMPLETE=count['COMPLETE'],GENERAL_POSITION_FLOAT_NOT_FORMAL_CERTIFIED=count['GENERAL_POSITION_FLOAT_NOT_FORMAL_CERTIFIED']+count['GENERAL_POSITION_FLOAT_NOT_FORMALLY_CERTIFIED'],
        UNKNOWN=count['UNKNOWN'],ERROR=count['ERROR'],all_other_statuses={k:v for k,v in count.items() if k not in ('COMPLETE','GENERAL_POSITION_FLOAT_NOT_FORMAL_CERTIFIED','GENERAL_POSITION_FLOAT_NOT_FORMALLY_CERTIFIED','UNKNOWN','ERROR')},
        traces_on_disk=trace_count,completed_shards=completed_shards,active_shard_locks=[str(p.parent.name) for p in locks],
        recent_trajectories_per_hour=throughput_recent,remaining_method_computations=48000-len(rows),
        observed_mean_method_cpu_seconds=round(mean_sec,4),method_traces_disk_mib=round(disk_mib,2),
        note='Computational accounting only; no scientific comparison or promise of background execution.')
    print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
