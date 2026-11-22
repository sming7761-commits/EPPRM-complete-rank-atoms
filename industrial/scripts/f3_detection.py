from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import beta

METHODS=('EP_PRM','Profiled_CCTM','OLS_PlugIn_PRM','Gaussian_GLR','Gaussian_CUSUM','Conformal_CUSUM')
EMPIRICAL=METHODS[2:]


def seed_for(*parts):
    return int(hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()[:16],16)%(2**32)


def ols(y,x):
    d=float(x@x); return float(x@y/d) if d>1e-15 else 0.0


def glr(z2):
    n=len(z2); pref=np.r_[0.,np.cumsum(z2)]; out=np.zeros(n+1)
    for t in range(1,n+1):
        dur=np.arange(t,0,-1); q=(pref[t]-pref[:t])/dur; mask=q>1
        if np.any(mask): out[t]=np.max(.5*dur[mask]*(q[mask]-1-np.log(q[mask])))
    return out


def cusum(z2,s=1.5):
    out=np.zeros(len(z2)+1); inc=-np.log(s)+.5*(1-1/s**2)*z2
    for i,v in enumerate(inc,1): out[i]=max(0.,out[i-1]+v)
    return out


def conformal(cal,on,seed):
    d=sorted(map(float,cal)); rng=np.random.default_rng(seed); out=np.zeros(len(on)+1)
    for t,x in enumerate(on,1):
        i=bisect.bisect_right(d,float(x)); p=max((len(d)-i+rng.uniform())/(len(d)+1),1e-300)
        out[t]=max(0.,out[t-1]+math.log(.5)-.5*math.log(p)); bisect.insort_right(d,float(x))
    return out


def baseline_paths(ref,on,oracle_prm,seed):
    cy,cx=ref[1:],ref[:-1]; oy,ox=on[1:],on[:-1]
    a=ols(cy,cx); cal=np.abs(cy-a*cx); mon=np.abs(oy-a*ox)
    sd=max(float(np.sqrt(np.mean((cy-a*cx)**2))),1e-12); z2=((oy-a*ox)/sd)**2
    return {
        'OLS_PlugIn_PRM':np.asarray(oracle_prm(cal,mon,'order_abs')),
        'Gaussian_GLR':glr(z2),
        'Gaussian_CUSUM':cusum(z2),
        'Conformal_CUSUM':conformal(cal,mon,seed),
    },a


def calibrate(ref,T,oracle_prm,unit,B):
    cy,cx=ref[1:],ref[:-1]; a=ols(cy,cx); e=cy-a*cx; e=e-np.mean(e)
    rng=np.random.default_rng(seed_for('bootstrap',unit)); maxima={m:np.empty(B) for m in EMPIRICAL}
    for b in range(B):
        eb=rng.choice(e,size=T,replace=True); x=np.empty(T+1); x[0]=ref[-1]
        for i in range(T): x[i+1]=a*x[i]+eb[i]
        paths,_=baseline_paths(ref,x,oracle_prm,seed_for('calpath',unit,b))
        for m in EMPIRICAL: maxima[m][b]=float(np.max(paths[m]))
        if (b+1)%1000==0: print('CALIBRATION_PROGRESS',unit,b+1,B,flush=True)
    thresholds={m:float(np.nextafter(np.quantile(maxima[m],.95,method='higher'),np.inf)) for m in EMPIRICAL}
    return thresholds,maxima,a


def atomic_npz(path,**kw):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.stem+'.tmp.npz')
    with tmp.open('wb') as f: np.savez_compressed(f,**kw); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


def run_unit(job):
    unit,cases,features,out,pkg,B=job
    for sub in ('epprm','baselines','cctm_profile'): sys.path.insert(0,str(pkg/'01_source'/sub))
    from prm_core import oracle_prm
    from full_domain_prm_atoms_fast import stream_fullatom
    from full_domain_cctm_atoms_fast import cctm_fullatom_stream

    health=next(c for c in cases if c['label']=='health')
    with np.load(features/f"{health['case_id']}.npz",allow_pickle=False) as z: ref=z['reference'].copy()
    thresholds,maxima,a_cal=calibrate(ref,600,oracle_prm,unit,B)
    threshold_rows=[dict(statistical_unit=unit,method=m,threshold=thresholds[m],bootstrap_reps=B,target_window_fpr=.05,a_hat=a_cal) for m in EMPIRICAL]
    maximum_rows=[]
    for m in EMPIRICAL:
        maximum_rows.extend(dict(statistical_unit=unit,method=m,replicate=i,max_statistic=float(v)) for i,v in enumerate(maxima[m]))

    fault_cases=sorted((c for c in cases if c['label']=='fault'),key=lambda c:int(c['monitor_start']))
    primary=fault_cases[0]['case_id']
    rows=[]; trace_dir=out/'traces'
    for c in cases:
        cid=c['case_id']; tau=int(c['tau']); label=c['label']
        with np.load(features/f'{cid}.npz',allow_pickle=False) as z:
            reference=z['reference'].copy(); online=z['online'].copy()
        cy,cx=reference[1:],reference[:-1]; oy,ox=online[1:],online[:-1]
        paths={}; status={}; runtimes={}
        tic=time.perf_counter()
        try:
            q=stream_fullatom(cy,cx,oy,ox,threshold=20.); paths['EP_PRM']=np.asarray(q['profile_path']); status['EP_PRM']=q['status']
        except Exception as exc:
            paths['EP_PRM']=np.full(len(online),np.nan); status['EP_PRM']='UNKNOWN:'+repr(exc)
        runtimes['EP_PRM']=time.perf_counter()-tic
        tic=time.perf_counter()
        try:
            q=cctm_fullatom_stream(cy,cx,oy,ox,delta=.01,D=.5,C=.1,smooth=1e-6,stop_at_alarm=False,full_path=True)
            paths['Profiled_CCTM']=np.asarray(q['profile_path']); status['Profiled_CCTM']=q['status']
        except Exception as exc:
            paths['Profiled_CCTM']=np.full(len(online),np.nan); status['Profiled_CCTM']='UNKNOWN:'+repr(exc)
        runtimes['Profiled_CCTM']=time.perf_counter()-tic
        tic=time.perf_counter(); bp,_=baseline_paths(reference,online,oracle_prm,seed_for('testpath',unit,cid)); base_sec=time.perf_counter()-tic
        for m in EMPIRICAL: paths[m]=bp[m];status[m]='COMPLETE';runtimes[m]=base_sec/len(EMPIRICAL)
        th={'EP_PRM':20.,'Profiled_CCTM':25.,**thresholds}
        for m in METHODS:
            v=np.asarray(paths[m],float); hits=np.flatnonzero(np.isfinite(v[1:])&(v[1:]>=th[m]))+1
            first=int(hits[0]) if len(hits) else 0
            post=hits[hits>=tau] if label=='fault' else np.array([],dtype=int)
            first_post=int(post[0]) if len(post) else 0
            pre=int(label=='fault' and first>0 and first<tau)
            valid=int(label=='fault' and first>=tau)
            false_alarm=int(label=='health' and first>0)
            row=dict(case_id=cid,statistical_unit=unit,label=label,fault_types=c['fault_types'],primary_fault_case=int(cid==primary),method=m,threshold=th[m],first_alarm=first,first_post_tau_crossing=first_post,pre_fault_alarm=pre,valid_detection=valid,false_alarm=false_alarm,detection_delay=(first-tau if valid else ''),max_statistic=(float(np.nanmax(v)) if np.any(np.isfinite(v)) else ''),runtime_seconds=runtimes[m],numeric_status=status[m])
            rows.append(row)
            atomic_npz(trace_dir/f'{cid}_{m}.npz',value=v,threshold=np.float64(th[m]),numeric_status=np.array(status[m]))
        print('F3_CASE_COMPLETE',cid,flush=True)
    return rows,threshold_rows,maximum_rows


def ci(k,n,alpha=.05):
    if n==0:return ('','')
    lo=0. if k==0 else float(beta.ppf(alpha/2,k,n-k+1))
    hi=1. if k==n else float(beta.ppf(1-alpha/2,k+1,n-k))
    return lo,hi


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);ap.add_argument('--f2',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--bootstrap-reps',type=int,default=5000);ap.add_argument('--workers',type=int,default=5)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    registry=list(csv.DictReader((a.f2/'case_registry.csv').open()))
    eligible=[c for c in registry if int(c['eligible'])==1]
    groups={}
    for c in eligible:groups.setdefault(c['statistical_unit'],[]).append(c)
    if len(groups)!=5 or len(eligible)!=12:raise RuntimeError(f'frozen F2 mismatch units={len(groups)} cases={len(eligible)}')
    jobs=[(u,cs,a.f2/'features',a.output,a.package,a.bootstrap_reps) for u,cs in sorted(groups.items())]
    rows=[];thr=[];mx=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,len(jobs))) as ex:
        fut=[ex.submit(run_unit,j) for j in jobs]
        for f in as_completed(fut):
            r,t,m=f.result();rows.extend(r);thr.extend(t);mx.extend(m)
    rows=sorted(rows,key=lambda r:(r['statistical_unit'],r['case_id'],r['method']))
    thr=sorted(thr,key=lambda r:(r['statistical_unit'],r['method']))
    mx=sorted(mx,key=lambda r:(r['statistical_unit'],r['method'],r['replicate']))
    write_csv(a.output/'case_method_results.csv',rows);write_csv(a.output/'calibration_thresholds.csv',thr);write_csv(a.output/'calibration_maxima.csv',mx)
    summary=[]
    for m in METHODS:
        z=[r for r in rows if r['method']==m]; health=[r for r in z if r['label']=='health']; primary=[r for r in z if r['primary_fault_case']==1]; allfault=[r for r in z if r['label']=='fault']
        hf=sum(r['false_alarm'] for r in health); pd=sum(r['valid_detection'] for r in primary); pp=sum(r['pre_fault_alarm'] for r in primary); ad=sum(r['valid_detection'] for r in allfault)
        hlo,hhi=ci(hf,len(health));plo,phi=ci(pd,len(primary)); delays=[float(r['detection_delay']) for r in primary if r['valid_detection']]
        summary.append(dict(method=m,health_units=len(health),health_false_alarms=hf,health_fpr=hf/len(health),health_fpr_ci_low=hlo,health_fpr_ci_high=hhi,primary_fault_units=len(primary),primary_valid_detections=pd,primary_detection_rate=pd/len(primary),primary_detection_ci_low=plo,primary_detection_ci_high=phi,primary_pre_fault_alarms=pp,primary_median_delay=(float(np.median(delays)) if delays else ''),all_fault_clusters=len(allfault),all_cluster_valid_detections=ad,numeric_unknown=sum(str(r['numeric_status']).startswith('UNKNOWN') for r in z),median_runtime_seconds=float(np.median([float(r['runtime_seconds']) for r in z]))))
    write_csv(a.output/'method_summary.csv',summary)
    protocol=dict(frozen_units=sorted(groups),primary_endpoint='earliest fault cluster per statistical unit',secondary_endpoint='all seven fault clusters',theoretical_thresholds={'EP_PRM':20,'Profiled_CCTM':25},empirical_baselines=list(EMPIRICAL),bootstrap_reps=a.bootstrap_reps,target_full_window_fpr=.05,bootstrap_source='fixed 737-point calibration segment only',fault_results_used_for_thresholds=False)
    (a.output/'GATE_F3_PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print('GATE_F3_COMPLETE',json.dumps({'units':len(groups),'cases':len(eligible),'method_records':len(rows),'calibration_maxima':len(mx),'unknown':sum(str(r['numeric_status']).startswith('UNKNOWN') for r in rows)}),flush=True)

if __name__=='__main__':main()
