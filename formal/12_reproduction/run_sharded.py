"""Crash-resumable, no-shared-writer scheduler. Research algorithms are imported verbatim.

Examples:
  python 12_reproduction/run_sharded.py --plan
  python 12_reproduction/run_sharded.py --run --workers 1 --limit 1
  python 12_reproduction/run_sharded.py --run --workers 2 --limit 2 --start-shard 1
  python 12_reproduction/run_sharded.py --run --workers 2
  python 12_reproduction/run_sharded.py --replay-parity gaussian health 0000

--limit limits NEW trajectories, not shards; workers own distinct shard locks.
"""
from __future__ import annotations
import argparse, collections, concurrent.futures, csv, hashlib, json, os, sys, time, traceback, threading
from pathlib import Path
import numpy as np
from shard_common import *


def planned_shards():return list(all_shards())

def manifest_shards():
    fp=ROOT/'04_seed_manifests/shards.csv'
    if fp.exists():return
    fp.parent.mkdir(parents=True,exist_ok=True)
    with fp.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(('shard_id','distribution','scenario','rep_start','rep_stop_exclusive','n_trajectories'))
        for d,sc,start,end in all_shards():w.writerow((shard_id(d,sc,start,end),d,sc,start,end,end-start))


def historic_rows():
    fp=ROOT/'07_tidy_results/method_records.csv'
    if not fp.exists():return {}
    out={}
    with fp.open(newline='') as f:
        for r in csv.DictReader(f):
            key=(r['distribution'],r['scenario'],int(r['replicate']),r['method'])
            if key in out:raise ValueError('duplicates in existing global CSV')
            out[key]=r
    return out


def get_rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss/1048576
    except Exception:
        try:
            import resource  # POSIX-only fallback; Windows relies on psutil.
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        except Exception:return float('nan')


def compute_method(m,d,sc,rep,data,xy,p,thresholds):
    cy,cx,oy,ox=xy
    thr=method_threshold(d,m,thresholds)
    tic=time.perf_counter();status='COMPLETE';fallback=0;atoms=None;params=None;err=''
    # RSS high-watermark sampled during actual method execution (not just after).
    peak=[get_rss_mb()];done=threading.Event()
    def memory_sample():
        while not done.wait(.025):
            try:peak[0]=max(peak[0],get_rss_mb())
            except Exception:pass
    monitor=threading.Thread(target=memory_sample,daemon=True);monitor.start()
    try:
        if m in p:trace=np.asarray(p[m],np.float64)
        elif m=='Old_K8':trace=np.asarray(core.old_k8_path(data['reference_A'],data['reference_B'],data['online_states']),np.float64)
        elif m=='Profiled_CCTM_fair':
            out=core.cctm_fullatom_stream(cy,cx,oy,ox,delta=.01,D=.5,C=.1,smooth=1e-6,stop_at_alarm=False,full_path=True)
            trace=np.asarray(out['profile_path'],np.float64)
            atoms=np.asarray(out['atom_counts'],np.int64)
            params=np.asarray(out['worst_parameters'],np.float64)
            fallback=int(out['status'].startswith('SMALL_INPUT'))
            status='GENERAL_POSITION_FLOAT_NOT_FORMAL_CERTIFIED' if out['status'].startswith('GENERAL_POSITION') else out['status']
        elif m=='EP_PRM_736':
            from full_domain_prm_atoms_fast import stream_fullatom
            out=stream_fullatom(cy,cx,oy,ox,threshold=20.)
            trace=np.asarray(out['profile_path'],np.float64)
            atoms=np.asarray(out['atom_counts'],np.int64)
            params=np.asarray(out['worst_parameter'],np.float64)
            status=out['status']
        else:raise RuntimeError('Unknown frozen method '+m)
        if trace.shape!=(601,) or not np.isfinite(trace).all():raise ArithmeticError('nonfinite or incomplete path')
    except Exception as e:
        trace=np.full(601,np.nan)
        status='UNKNOWN' if isinstance(e,core.NumericUnknown) else 'ERROR'
        err=repr(e)+'\n'+traceback.format_exc()
    runtime=time.perf_counter()-tic
    done.set();monitor.join(timeout=1)
    rss=max(peak[0],get_rss_mb())
    if atoms is None:atoms=np.full(601,-1,dtype=np.int64)
    if params is None:params=np.full(601,np.nan)
    trace=np.asarray(trace,dtype=np.float64)
    mask=np.isfinite(trace)
    running=np.maximum.accumulate(np.where(mask,trace,-np.inf))
    running[~mask]=np.nan
    crossings=np.where(mask,(trace>=thr).astype(np.int8),-1).astype(np.int8)
    arrays=dict(value=trace,running_max=running,crossing=crossings,threshold=np.float64(thr),
        computed=mask.astype(np.uint8),time_index=np.arange(601,dtype=np.int32),
        atom_count=atoms,worst_parameter=params,exact_fallback_count=np.int64(fallback),
        status=np.array(status),runtime_seconds=np.float64(runtime),peak_memory_mb=np.float64(rss),
        error_message=np.array(err))
    return arrays,err


def check_raw_input(d,sc,rep,raw_hashes):
    path=core.raw_file(d,sc,rep)
    rel=str(path.relative_to(ROOT)).replace('\\','/')
    if not path.is_file():raise FileNotFoundError(str(path))
    if filehash(path)!=raw_hashes[rel]:raise ValueError('FROZEN RAW SHA MISMATCH: '+rel)
    with np.load(path,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    if int(data['seed'])!=core.seed_for(d,sc,rep):raise ValueError('frozen raw seed mismatch')
    if data['online_states'].shape!=(601,) or len(data['reference_A'])!=113 or len(data['reference_B'])!=625:raise ValueError('raw shape mismatch')
    if str(data['distribution'])!=d or str(data['scenario'])!=sc:raise ValueError('raw group mismatch')
    return data


def shard_worker(task):
    d,sc,start,end,reps=task
    sid=shard_id(d,sc,start,end)
    base=shard_dir(sid);base.mkdir(parents=True,exist_ok=True)
    lock=base/'.lock'
    try:
        fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError:
        # Recover only stale worker locks; never override a confirmed live process.
        try:
            import psutil
            info=json.loads(lock.read_text());pid=int(info['pid']);born=float(info['created_at'])
            try:
                alive=psutil.pid_exists(pid) and abs(psutil.Process(pid).create_time()-born)<5.0
            except psutil.NoSuchProcess:alive=False
            if alive:raise RuntimeError(f'shard already owned by live pid {pid}: {lock}')
            lock.unlink()
            fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        except (KeyError,ValueError,json.JSONDecodeError):
            raise RuntimeError(f'unrecognized lock; fail closed {lock}')
    try:
        import psutil
        born=psutil.Process().create_time()
    except Exception:born=time.time()
    with os.fdopen(fd,'w') as f:
        json.dump(dict(pid=os.getpid(),created_at=born,shard=sid),f)
    completed=0;repaired=0;skipped=0;failures=0;t_start=time.monotonic()
    try:
        verify_scientific_frozen()
        thresholds=freeze_thresholds();legacy=historic_rows();hashes=frozen_raw_hashes()
        for rep in reps:
            data=None;xy=None;p=None
            for m in METHODS:
                tp=trace_path(d,sc,rep,m);jp=journal_path(sid,rep,m)
                thr=method_threshold(d,m,thresholds)
                old=legacy.get((d,sc,rep,m))
                jr=json.loads(jp.read_text()) if jp.exists() else None
                if tp.exists():
                    try:
                        arr,first=validate_trace(tp,thr,require_time=True)
                        candidate=jr or old or row_from_trace(arr,first,d,sc,rep,m,thr,thresholds)
                        validate_row(candidate,tp,thresholds)
                        if jr is None:atomic_json(jp,candidate);repaired+=1
                        skipped+=1
                        continue
                    except (ValueError,KeyError,AssertionError) as exc:
                        # Existing traces must never be silently overwritten.
                        # Fail closed and leave evidence for manual review.
                        audit=ROOT/'09_algorithm_audit/formal_failures'/f'{sid}_{rep:04d}_{m}_invalid_existing.json'
                        atomic_json(audit,dict(shard=sid,rep=rep,method=m,error=repr(exc),trace_sha256=filehash(tp)))
                        raise RuntimeError('INVALID EXISTING TRACE: '+str(tp)) from exc
                if jr is not None or old is not None:
                    # Summary without trace: run original trajectory/method again.
                    audit=ROOT/'09_algorithm_audit/formal_failures'/f'{sid}_{rep:04d}_{m}_orphan_row.json'
                    atomic_json(audit,dict(shard=sid,rep=rep,method=m,old_row=jr or old,action='RECOMPUTE_SAME_FROZEN_INPUT'))
                if data is None:
                    data=check_raw_input(d,sc,rep,hashes)
                    xy=core.arrays_to_xy(data)
                    p=core.basic_paths(*xy,int(data['seed']))
                print('METHOD_BEGIN',sid,rep,m,flush=True)
                arr,err=compute_method(m,d,sc,rep,data,xy,p,thresholds)
                core.write_npz_atomic(tp,**arr)
                # A trace-only completed transaction is recoverable, even if killed here.
                saved,first=validate_trace(tp,thr,require_time=True)
                row=row_from_trace(saved,first,d,sc,rep,m,thr,thresholds)
                validate_row(row,tp,thresholds)
                atomic_json(jp,row)
                completed+=1
                if first<0:
                    failures+=1
                    audit=ROOT/'09_algorithm_audit/formal_failures'/f'{sid}_{rep:04d}_{m}_failure.json'
                    atomic_json(audit,dict(shard=sid,trajectory_id=row['trajectory_id'],method=m,raw_sha256=filehash(core.raw_file(d,sc,rep)),numeric_status=row['numeric_status'],error_message=err))
                print('SHARD',sid,rep,m,row['numeric_status'],'alarm',first,'sec',round(float(row['runtime_seconds']),3),flush=True)
        # Adopt already completed legacy records into the shard's local journal.
        # No scientific computation is repeated; every source trace is validated.
        allrows=[]
        for rep in range(start,end):
            for m in METHODS:
                jp=journal_path(sid,rep,m)
                if not jp.exists() and (d,sc,rep,m) in legacy:
                    row=legacy[(d,sc,rep,m)]
                    validate_row(row,trace_path(d,sc,rep,m),thresholds)
                    atomic_json(jp,row)
                    repaired+=1
                if jp.exists():allrows.append(json.loads(jp.read_text()))
        atomic_csv(base/f'method_records_shard_{sid}.csv',allrows)
        atomic_json(base/'shard_status.json',dict(shard=sid,expected=8*(end-start),journal_records=len(allrows),new_method_records=completed,skipped=skipped,recovered_rows=repaired,failures=failures,finished_at=time.time(),elapsed_seconds=time.monotonic()-t_start))
        return dict(shard=sid,completed=completed,skipped=skipped,repaired=repaired,failures=failures,elapsed_seconds=round(time.monotonic()-t_start,2))
    finally:
        if lock.exists():
            try:
                owner=json.loads(lock.read_text())
                if int(owner.get('pid',-1))==os.getpid():lock.unlink()
            except (ValueError,json.JSONDecodeError):pass


def replay_parity(d,sc,rep):
    thresholds=freeze_thresholds();hashes=frozen_raw_hashes();data=check_raw_input(d,sc,rep,hashes)
    xy=core.arrays_to_xy(data);p=core.basic_paths(*xy,int(data['seed']))
    out=[]
    for m in METHODS:
        fp=trace_path(d,sc,rep,m)
        if not fp.exists():raise RuntimeError('cannot replay missing trace '+str(fp))
        a0,first0=validate_trace(fp,method_threshold(d,m,thresholds),require_time=True)
        a,err=compute_method(m,d,sc,rep,data,xy,p,thresholds)
        if err:raise RuntimeError('REPLAY FAILED '+m+' '+err)
        first=int(np.flatnonzero(a['crossing'][1:]==1)[0]+1) if np.any(a['crossing'][1:]==1) else 0
        maxdiff=float(np.max(np.abs(a['value']-a0['value'])))
        # Historical Windows CCTM checkpoints did not store the optional
        # worst-parameter diagnostic and contain all-NaN placeholders.
        # Compare all scientific paths exactly while normalizing only that
        # explicitly unavailable legacy auxiliary field.
        legacy_worst_unavailable = bool(np.all(np.isnan(a0['worst_parameter'])))
        if legacy_worst_unavailable:
            comparable = dict(a)
            comparable['worst_parameter'] = a0['worst_parameter']
            hash_equal = semantic_hash(comparable) == semantic_hash(a0)
        else:
            hash_equal = semantic_hash(a) == semantic_hash(a0)
        if not hash_equal or first!=first0:raise AssertionError(f'PARITY FAILED {d}/{sc}/{rep}/{m}: diff={maxdiff}, {first}, {first0}')
        out.append(dict(distribution=d,scenario=sc,rep=rep,method=m,first_alarm=first,hash_equal=True,max_wealth_abs_error=maxdiff,semantic_sha256=semantic_hash(a)))
    audit=ROOT/'09_algorithm_audit'/f'replay_parity_{d}_{sc}_{rep:04d}.csv'
    audit.parent.mkdir(parents=True,exist_ok=True)
    tmp=audit.with_name(audit.name+f'.tmp.{os.getpid()}')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out);f.flush();os.fsync(f.fileno())
    os.replace(tmp,audit)
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',action='store_true');ap.add_argument('--run',action='store_true');ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--limit',type=int,default=None,help='max NEW trajectories for this command')
    ap.add_argument('--per-shard-limit',type=int,default=25,help='engineering parity runs: pick 1 missing trajectory from each shard')
    ap.add_argument('--start-shard',type=int,default=0);ap.add_argument('--replay-parity',nargs=3,metavar=('DIST','SCENE','REP'))
    ap.add_argument('--group',default=None,help='optional group e.g. gaussian:health')
    args=ap.parse_args();manifest_shards()
    if args.plan:
        print('SHARDS',len(planned_shards()),'EXPECTED_TRAJECTORIES',6000,'EXPECTED_METHOD_ROWS',48000)
        return
    if args.replay_parity:
        d,sc,rep=args.replay_parity;r=replay_parity(d,sc,int(rep))
        print('REPLAY_PASS',len(r),'backend records');return
    if not args.run:ap.error('choose --plan, --run or --replay-parity')
    if args.workers<1 or args.workers>8:ap.error('workers must be 1..8')
    try:
        import psutil
        available_mb=psutil.virtual_memory().available/1048576
    except Exception:available_mb=0
    # Refuse dangerous no-swap concurrency (override is intentionally not available).
    reserve=1150;estimated_mb=750
    if available_mb and args.workers*estimated_mb+reserve>available_mb:
        raise SystemExit(f'MEMORY SAFETY REFUSAL: {args.workers} workers need ~{args.workers*estimated_mb+reserve} MB, only {available_mb:.0f} MB available')
    pending=[];limit=args.limit
    legacy=historic_rows();th=freeze_thresholds()
    for ix,(d,sc,start,end) in enumerate(planned_shards()):
        if ix<args.start_shard:continue
        if args.group and args.group!=d+':'+sc:continue
        reps=[]
        for rep in range(start,end):
            complete=True
            for m in METHODS:
                tp=trace_path(d,sc,rep,m)
                jp=journal_path(shard_id(d,sc,start,end),rep,m)
                if not tp.exists() or not jp.exists() and (d,sc,rep,m) not in legacy:
                    complete=False;break
            if complete:continue
            if limit is not None and limit<=0:break
            reps.append(rep)
            if limit is not None:limit-=1
            if len(reps)>=args.per_shard_limit:break
        if reps:pending.append((d,sc,start,end,reps))
        if limit is not None and limit<=0:break
    if not pending:
        print('NO_MISSING_TRAJECTORIES_SELECTED');return
    print('SCHEDULE',len(pending),'shards',sum(len(x[-1]) for x in pending),'trajectories','workers',args.workers,'available_ram_mb',round(available_mb),flush=True)
    results=[];invoke_started=time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        future={ex.submit(shard_worker,t):t for t in pending}
        for f in concurrent.futures.as_completed(future):
            result=f.result();results.append(result);print('SHARD_DONE',json.dumps(result),flush=True)
    invoke_ended=time.time()
    atomic_json(ROOT/'10_logs/last_sharded_invocation.json',dict(results=results,started_at=invoke_started,completed_at=invoke_ended,elapsed_wall_seconds=invoke_ended-invoke_started,new_method_records=sum(r['completed'] for r in results),workers=args.workers))
    print('ALL_SCHEDULED_SHARDS_DONE',len(results),flush=True)
if __name__=='__main__':main()
