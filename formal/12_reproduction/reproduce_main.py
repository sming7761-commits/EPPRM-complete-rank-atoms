"""Frozen EPPRM main study. NO use of fault trajectories in threshold calibration.
The CLI supports resume, atomic per-partition save and exactly preserved binary64 arrays.
Run from package root: python 12_reproduction/reproduce_main.py calibrate gaussian 0 5000
                     python 12_reproduction/reproduce_main.py raw gaussian health 0 1000
                     python 12_reproduction/reproduce_main.py run gaussian health 0 1000
Do not mix pilot and formal samples. No approximated complete profile is written.
"""
from __future__ import annotations
import sys,csv,math,time,hashlib,json,os,traceback,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
for name in ('simulation','epprm','baselines','cctm_profile'):
    sys.path.insert(0,str(ROOT/'01_source'/name))
from simulate_core import ar_path
from prm_core import oracle_prm
from modern_baselines import cctm_wealth
from full_domain_cctm_atoms_fast import cctm_fullatom_stream,NumericUnknown
from legacy_baselines import old_k_tail_monitor,_shared_max_tail_count
from scipy.stats import binom,beta
from scipy.stats import binomtest

DISTS=['gaussian','t5','exp_centered','skew_mixture']
NOISE=['gaussian','t5','exp','skew_mix']
METHODS=['EP_PRM_736','Profiled_CCTM_fair','OLS_PlugIn_PRM','Gaussian_GLR','Gaussian_CUSUM','Conformal_CUSUM','Old_K8','Oracle_PRM']
N=736;T=600;TAU=357;A0=.9
SEEDPREFIX_CAL=20261110000000
SEEDPREFIX_FORMAL=20261109000000


def seed_for(dist,scenario,rep,cal=False):
    j=DISTS.index(dist)
    if cal:return SEEDPREFIX_CAL+j*100000000+rep
    return SEEDPREFIX_FORMAL+j*100000000+(10000000 if scenario=='fault' else 0)+rep


def create_trajectory(dist,scenario,rep,cal=False):
    seed=seed_for(dist,scenario,rep,cal)
    rng=np.random.default_rng(seed)
    noise=NOISE[DISTS.index(dist)]
    A,eps_A=ar_path(rng,A0,112,noise,burn=200)
    B,eps_B=ar_path(rng,A0,624,noise,burn=200)
    on,eps_on=ar_path(rng,A0,600,noise,scale=1.5 if scenario=='fault' else 1.,fault_at=(TAU-1) if scenario=='fault' else None,burn=200)
    return dict(reference_A=A,reference_B=B,reference_innovations_A=eps_A,
                reference_innovations_B=eps_B,online_states=on,online_innovations=eps_on,
                seed=np.int64(seed),scenario=np.array(scenario),distribution=np.array(dist),
                change_point=np.int32(TAU),a0=np.float64(A0),scale_factor=np.float64(1.5 if scenario=='fault' else 1.))


def arrays_to_xy(data):
    A=data['reference_A'];B=data['reference_B'];on=data['online_states']
    return np.r_[A[1:],B[1:]],np.r_[A[:-1],B[:-1]],on[1:],on[:-1]


def write_npz_atomic(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.stem+'.temp.npz')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def raw_file(dist,scenario,rep):
    return ROOT/'05_raw_data'/('health' if scenario=='health' else 'fault')/dist/f'{rep:04d}.npz'


def ensure_raw(dist,scenario,rep,overwrite=False):
    path=raw_file(dist,scenario,rep)
    if overwrite or not path.exists():write_npz_atomic(path,**create_trajectory(dist,scenario,rep))
    with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    assert int(d['seed'])==seed_for(dist,scenario,rep)
    assert len(d['reference_A'])==113 and len(d['reference_B'])==625
    assert len(d['online_states'])==601 and len(d['online_innovations'])==600
    np.testing.assert_allclose(np.diff(d['online_states'])+(1-A0)*d['online_states'][:-1],d['online_innovations'],rtol=0,atol=2e-14)
    return d


def ols(cy,cx):return float(np.dot(cy,cx)/np.dot(cx,cx))


from numba import njit
@njit(cache=True)
def _glr_numba(z):
    n=len(z); pref=np.zeros(n+1);out=np.zeros(n+1)
    for j in range(n):pref[j+1]=pref[j]+z[j]
    for t in range(1,n+1):
        best=0.
        for i in range(t):
            q=(pref[t]-pref[i])/(t-i)
            if q>1.:
                val=.5*(t-i)*(q-1.-math.log(q))
                if val>best:best=val
        out[t]=best
    return out

def glr(z2):
    return _glr_numba(np.asarray(z2,dtype=np.float64))


def cusum(z2,s=1.5):
    inc=-np.log(s)+.5*(1-1/(s*s))*z2
    arr=np.zeros(T+1)
    for i,x in enumerate(inc,1):arr[i]=max(0.,arr[i-1]+x)
    return arr


def conf_cusum(cal,on,seed):
    import bisect
    data=sorted(map(float,cal));rng=np.random.default_rng(seed%(2**32));out=np.zeros(T+1)
    for t,x in enumerate(on,1):
        i=bisect.bisect_right(data,float(x));greater=len(data)-i
        p=max((greater+rng.uniform())/(len(data)+1),1e-308)
        out[t]=max(0.,out[t-1]+math.log(.5)-.5*math.log(p))
        bisect.insort_right(data,float(x))
    return out


def basic_paths(cy,cx,oy,ox,seed):
    a=ols(cy,cx);cal=np.abs(cy-a*cx);on=np.abs(oy-a*ox)
    s=np.sqrt(np.mean((cy-a*cx)**2));z2=((oy-a*ox)/max(float(s),1e-12))**2
    return {
      'OLS_PlugIn_PRM':oracle_prm(cal,on,'order_abs'),
      'Gaussian_GLR':glr(z2),
      'Gaussian_CUSUM':cusum(z2),
      'Conformal_CUSUM':conf_cusum(cal,on,seed),
      'Oracle_PRM':oracle_prm(np.abs(cy-A0*cx),np.abs(oy-A0*ox),'order_abs')}


def calibration_path(dist,rep):
    data=create_trajectory(dist,'health',rep,cal=True)
    cy,cx,oy,ox=arrays_to_xy(data)
    p=basic_paths(cy,cx,oy,ox,int(data['seed']))
    return {k:float(np.max(p[k])) for k in ['OLS_PlugIn_PRM','Gaussian_GLR','Gaussian_CUSUM','Conformal_CUSUM']}


def append_csv(path,fields,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        if f.tell()==0:w.writeheader()
        w.writerows(rows);f.flush();os.fsync(f.fileno())


def calibrate(dist,start,count):
    fn=ROOT/'05_raw_data/calibration'/f'max_{dist}.csv'
    prior={}
    if fn.exists():
        with fn.open(newline='') as f:
            for r in csv.DictReader(f):prior[int(r['rep'])]=r
    fields=['trajectory_id','distribution','rep','seed','method','max_statistic','runtime_seconds']
    # each calibration trajectory has four method entries, do not double count
    done={}
    for r in prior.values():
        pass
    existing=set()
    if fn.exists():
        with fn.open(newline='') as f:
            for r in csv.DictReader(f):existing.add((int(r['rep']),r['method']))
    t0=time.perf_counter()
    pending=[]
    for rep in range(start,start+count):
        needed=[m for m in ['OLS_PlugIn_PRM','Gaussian_GLR','Gaussian_CUSUM','Conformal_CUSUM'] if (rep,m) not in existing]
        if not needed:continue
        t=time.perf_counter();p=calibration_path(dist,rep);elapsed=time.perf_counter()-t
        rows=[dict(trajectory_id=f'CAL_{dist}_{rep:05d}',distribution=dist,rep=rep,seed=seed_for(dist,'health',rep,cal=True),method=m,max_statistic=p[m],runtime_seconds=elapsed) for m in needed]
        pending.extend(rows)
        if len(pending)>=200:
            append_csv(fn,fields,pending);pending=[]
        if rep%250==249:print('CAL',dist,rep+1,'elapsed',round(time.perf_counter()-t0,2),flush=True)
    if pending:append_csv(fn,fields,pending)
    print('CAL_FINISHED',dist,start,count,flush=True)


def thresholds():
    r={};reports=[]
    for dist in DISTS:
        fn=ROOT/'05_raw_data/calibration'/f'max_{dist}.csv'
        if not fn.exists():continue
        vals={m:{} for m in ['OLS_PlugIn_PRM','Gaussian_GLR','Gaussian_CUSUM','Conformal_CUSUM']}
        for x in csv.DictReader(fn.open()):
            m=x['method'];i=int(x['rep']);assert i not in vals[m],(dist,m,i)
            vals[m][i]=float(x['max_statistic'])
        if not all(len(v)==5000 for v in vals.values()):continue
        r[dist]={}
        for m,series in vals.items():
            scores=np.sort(np.array([series[i] for i in range(5000)]))
            rank=math.ceil(5001*.95)
            thr=float(np.nextafter(scores[rank-1],np.inf))
            r[dist][m]=thr
            count=int(np.sum(scores>=thr));ci=binomtest(count,5000).proportion_ci(.95,method='exact')
            reports.append(dict(distribution=dist,method=m,n_calibration=5000,quantile_rank=rank,threshold=thr,calibration_exceedances=count,empirical_pfa=count/5000,calibration_pfa_ci_low=ci.low,calibration_pfa_ci_high=ci.high,threshold_source='INDEPENDENT_EMPIRICAL_PFA'))
    (ROOT/'03_configs/frozen_thresholds.json').write_text(json.dumps(r,indent=2,ensure_ascii=False))
    if reports:
        p=ROOT/'08_statistical_analysis/calibration_summary.csv'
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(reports[0]));w.writeheader();w.writerows(reports)
    print('THRESHOLDS_FROZEN_FOR',list(r),flush=True)


def old_k8_path(A,B,on):
    # frozen old K8 code: uses B prefix for 112 training and suffix 512 calibration
    from scipy.stats import beta,binom
    x=np.r_[A[:-1],B[:112]];y=np.r_[A[1:],B[1:113]]
    ratios=np.sort(y/x);m=len(ratios)
    kl=int(binom.ppf(.004/2,m,.5));ku=int(binom.isf(.004/2,m,.5))
    lo=float(ratios[m-ku-1]);hi=float(ratios[m-kl])
    cal=B[112:];n=512;THRESH=(1.5,2.,2.5,3.)
    from legacy_baselines import _shared_max_tail_count,old_k_tail_monitor
    probs=[]
    for c in THRESH:
        J=_shared_max_tail_count(cal[:-1],cal[1:],c,lo,hi)
        probs.append(1. if J==n else float(beta.ppf(1-.006/len(THRESH),J+1,n-J)))
    return old_k_tail_monitor(on, (lo,hi,np.array(probs)),K=8,thresholds=THRESH,alpha_monitor=.04)


def trace_path(dist,scene,rep,m):
    return ROOT/'06_method_traces'/dist/scene/f'{rep:04d}_{m}.npz'


def save_trace(dist,scene,rep,m,values,thr,status,atoms=None,params=None,fallback=0):
    values=np.asarray(values,dtype=np.float64)
    assert values.shape==(601,)
    computed=np.isfinite(values)
    running=np.maximum.accumulate(np.where(computed,values,-np.inf))
    running[~computed]=np.nan
    cross=np.where(computed,(values>=thr).astype(np.int8),-1).astype(np.int8)
    arrays=dict(value=values,running_max=running,crossing=cross,
                threshold=np.float64(thr),computed=computed.astype(np.uint8),
                status=np.array(status),atom_count=np.asarray(atoms if atoms is not None else np.full(601,-1),dtype=np.int64),
                worst_parameter=np.asarray(params if params is not None else np.full(601,np.nan),dtype=float),
                exact_fallback_count=np.int64(fallback))
    write_npz_atomic(trace_path(dist,scene,rep,m),**arrays)
    return cross


def run_one(dist,scene,rep,thrmap):
    data=ensure_raw(dist,scene,rep)
    cy,cx,oy,ox=arrays_to_xy(data)
    rows=[]
    seed=int(data['seed']);id=f'FORMAL_{dist}_{scene}_{rep:04d}'
    tm=ROOT/'07_tidy_results/method_records.csv'
    present=set()
    if tm.exists():
        with tm.open(newline='') as f:
            present={(r['trajectory_id'],r['method']) for r in csv.DictReader(f)}
    p=basic_paths(cy,cx,oy,ox,seed)
    A=data['reference_A']; B=data['reference_B'];on=data['online_states']
    for m in METHODS:
        if (id,m) in present:continue
        t0=time.perf_counter();status='COMPLETE';err='';trace=np.full(601,np.nan)
        count=None;mem=np.nan;fallback=0;atoms=None;params=None
        thr=20. if m in ('EP_PRM_736','Oracle_PRM') else 25. if m in ('Profiled_CCTM_fair','Old_K8') else float(thrmap[m])
        src='MARGINAL_ANYTIME_PFA' if m in ('EP_PRM_736','Oracle_PRM') else 'CCTM_ALPHA_0.04_DELTA_0.01' if m=='Profiled_CCTM_fair' else 'LEGACY_K8_SYMMETRIC_NOISE_ONLY' if m=='Old_K8' else 'INDEPENDENT_EMPIRICAL_PFA'
        try:
            if m in p: trace=np.asarray(p[m],float)
            elif m=='Old_K8':trace=np.asarray(old_k8_path(A,B,on),float)
            elif m=='Profiled_CCTM_fair':
                q=cctm_fullatom_stream(cy,cx,oy,ox,delta=.01,D=.5,C=.1,smooth=1e-6,stop_at_alarm=False,full_path=True)
                trace=q['profile_path'];atoms=np.array(q['atom_counts'],int);params=q['worst_parameters'];count=int(atoms[-1]);fallback=1 if q['status'].startswith('SMALL_INPUT') else 0
                if q['status'].startswith('GENERAL_POSITION'):status='GENERAL_POSITION_FLOAT_NOT_FORMAL_CERTIFIED'
            elif m=='EP_PRM_736':
                from full_domain_prm_atoms_fast import stream_fullatom
                q=stream_fullatom(cy,cx,oy,ox,threshold=20.)
                trace=np.asarray(q['profile_path'],float)
                atoms=np.asarray(q['atom_counts'],int)
                params=np.asarray(q['worst_parameter'],float)
                count=int(atoms[-1])
                status=q['status']
            else:raise RuntimeError('Unknown method')
            if np.any(~np.isfinite(trace)):raise ArithmeticError('nonfinite trace')
            cross=save_trace(dist,scene,rep,m,trace,thr,status,atoms,params,fallback)
            hits=np.flatnonzero(cross[1:]==1);first=int(hits[0]+1) if len(hits) else 0
        except (NumericUnknown,Exception) as e:
            first=-1; status='UNKNOWN' if isinstance(e,NumericUnknown) else 'ERROR';err=repr(e)
            save_trace(dist,scene,rep,m,trace,thr,status,atoms,params,fallback)
        secs=time.perf_counter()-t0
        rows.append(dict(trajectory_id=id,method=m,distribution=dist,scenario=scene,replicate=rep,seed=seed,
          threshold=thr,threshold_source=src,first_alarm=first,prefault_alarm=int(0<first<TAU),
          detected_after_change=int(scene=='fault' and first>=TAU),
          delay=(first-TAU if scene=='fault' and first>=TAU else 244),censored=int(not(scene=='fault' and first>=TAU)),
          max_statistic=float(np.nanmax(trace)) if np.any(np.isfinite(trace)) else np.nan,
          runtime_seconds=secs,peak_memory_mb=mem,atom_count_final=count,
          exact_fallback_count=fallback,numeric_status=status,error_message=err))
        print('METHOD',id,m,status,'alarm',first,'time',round(secs,2),flush=True)
    if rows:append_csv(tm,list(rows[0]),rows)


def main():
    if len(sys.argv)<2:raise SystemExit('calibrate|thresholds|raw|run')
    cmd=sys.argv[1]
    if cmd=='thresholds':thresholds();return
    if len(sys.argv)!=(5 if cmd=='calibrate' else 6):raise SystemExit('Usage: calibrate dist start count | raw/run dist health|fault start count')
    if cmd=='calibrate':
        _,_,d,s,c=sys.argv;calibrate(d,int(s),int(c));return
    _,_,dist,scene,start,count=sys.argv; start=int(start);count=int(count)
    assert dist in DISTS and scene in ('health','fault') and start>=0
    assert start+count <= (1000 if scene=='health' else 500)
    if cmd in ('raw','repair_fault'):
        if cmd=='repair_fault' and scene!='fault':raise ValueError('repair_fault only for fault scene')
        for rep in range(start,start+count):ensure_raw(dist,scene,rep,overwrite=(cmd=='repair_fault'))
        print('RAW_SAVED',dist,scene,start,count);return
    if cmd=='run':
        thr=json.loads((ROOT/'03_configs/frozen_thresholds.json').read_text())
        if dist not in thr:raise RuntimeError('No full 5000-calibration frozen thresholds for '+dist)
        for rep in range(start,start+count):run_one(dist,scene,rep,thr[dist])
        return
    raise ValueError(cmd)
if __name__=='__main__':main()
