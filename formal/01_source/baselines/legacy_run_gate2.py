"""Gate2 paired-trial experiment, fixed 736 healthy transitions = 112A + 624B.
No synthetic performance estimates; rows are generated from actual independent samples.
A and B independent paths; B prefix paired with A; B suffix 512 used in PRM.
Always emits raw per-trajectory records, including CS failure, first alarm and profiling status.
Run: python run_gate2.py gaussian healthy 0 1000
"""
from __future__ import annotations
import sys,os,time,csv,math
from pathlib import Path
import numpy as np
from scipy.stats import binom,beta
from simulate_core import ar_path
from crosspair_cs import crosspair_roots,lookup_beta,invert,in_interval
from prm_core import oracle_prm,first_alarm
from baselines import _shared_max_tail_count,old_k_tail_monitor
from exact_sweep import exact_profile,prove_no_alarm_by_witness

ROOT=Path(__file__).resolve().parent
RAW=Path(os.environ.get('SNPRM_RAW',str(ROOT/'health_raw.csv')))
FIELDS=['dist','scenario','rep','seed','a0','M_A','M_B_id','N_cal','T','tau','method','alarm','early','detected','cs_covered','cs_width','cs_segments','first_param','median_root','profile_exact','screened_non_alarm','num_cells','argmin_at_end','argmin_dist','log_oracle_gap_end','seconds','error']
DIST=['gaussian','t5','t10','exp','skew_mix']
SCENARIOS={'healthy':1.,'weak':1.5,'strong':2.}
M=112;N=512;T=600;TAU=357
ETA=.01;ALPHA=.04
THRESH=(1.5,2.0,2.5,3.0)
TAB=lookup_beta(M)

def old_cal(A,B,cal):
    """Old K1/K8: historical symmetric-innovation ratio-sign CI, using 112+112
    *transitions* and same B suffix as new method. Only valid if innovations
    are symmetric (Gaussian, t5, t10); skew/exp labelled empirical only.
    """
    x=np.r_[A[:-1],B[:M]];y=np.r_[A[1:],B[1:M+1]]
    ratios=np.sort(y/x);m=len(ratios)
    delta_a=.004;delta_tail=.006
    kl=int(binom.ppf(delta_a/2,m,.5));ku=int(binom.isf(delta_a/2,m,.5))
    lo=float(ratios[m-ku-1]);hi=float(ratios[m-kl]);
    p=[]
    for c in THRESH:
        J=_shared_max_tail_count(cal[:-1],cal[1:],c,lo,hi)
        v=1. if J==N else float(beta.ppf(1-delta_tail/len(THRESH),J+1,N-J))
        p.append(v)
    return (lo,hi,np.array(p))

def assess(alarm,sigma):
    alarm=int(alarm)
    return dict(alarm=alarm,early=int(alarm>0 and alarm<=TAU),
      detected=int(alarm>TAU) if sigma!=1. else int(alarm>0))

def do_trial(dist,scenario,rep,methods=('oracle','plugin','k8','unweighted','weighted'),force_exact=False):
    ix=DIST.index(dist);jj=list(SCENARIOS).index(scenario)
    seed=int(20261107000000+ix*100000000+jj*10000000+rep)
    rng=np.random.default_rng(seed)
    a0=.9;s=SCENARIOS[scenario]
    A,_=ar_path(rng,a0,M,dist,burn=200)
    B,_=ar_path(rng,a0,M+N,dist,burn=200)
    on,_=ar_path(rng,a0,T,dist,scale=s,fault_at=TAU,burn=200)
    cal=B[M:];caly=cal[1:];calx=cal[:-1]
    ony=on[1:];onx=on[:-1]
    rr,q,_=crosspair_roots(A,B,M)
    pm=float(np.median(rr))
    base=dict(dist=dist,scenario=scenario,rep=rep,seed=seed,a0=a0,M_A=M,M_B_id=M,N_cal=N,T=T,tau=TAU,median_root=pm)
    rows=[]
    def put(method,alarm,tt,**extra):
        row=dict(base,method=method,**assess(alarm,s),seconds=round(tt,6),**extra)
        rows.append(row)
    t=time.perf_counter()
    w_oracle=oracle_prm(np.abs(caly-a0*calx),np.abs(ony-a0*onx),'order_abs')
    oracle_alarm=first_alarm(w_oracle,ALPHA)
    if 'oracle' in methods:put('oracle',oracle_alarm,time.perf_counter()-t)
    if 'plugin' in methods:
        t=time.perf_counter()
        w=oracle_prm(np.abs(caly-pm*calx),np.abs(ony-pm*onx),'order_abs')
        put('plugin',first_alarm(w,ALPHA),time.perf_counter()-t)
    if any(x in methods for x in ['k1','k8']):
        t=time.perf_counter()
        ci=old_cal(A,B,cal)
        for label,K in [('k1',1),('k8',8)]:
            if label not in methods:continue
            w=old_k_tail_monitor(on,ci,K=K,thresholds=THRESH,alpha_monitor=ALPHA)
            put(label,first_alarm(w,ALPHA),time.perf_counter()-t)
    for wt,label in [(False,'unweighted'),(True,'weighted')]:
        if label not in methods:continue
        t=time.perf_counter()
        C=invert(rr,q,TAB,eta=ETA,weighted=wt)
        cover=in_interval(C,a0)
        width=sum(b-a for a,b in C)
        extras=dict(cs_covered=int(cover),cs_width=width,cs_segments=len(C),first_param=C[0][0] if C else None)
        if not C:
            put(label,0,time.perf_counter()-t,profile_exact=0,screened_non_alarm=0,error='EMPTY_CS',**extras)
            continue
        screened=False
        if not force_exact:
            screened,_=prove_no_alarm_by_witness(caly,calx,ony,onx,C,alpha=ALPHA,num=10)
        if screened:
            put(label,0,time.perf_counter()-t,profile_exact=0,screened_non_alarm=1,**extras)
        else:
            try:
                res=exact_profile(caly,calx,ony,onx,C,threshold=1/ALPHA)
                at=res['alarm'] if res['alarm'] else T
                arg=float(res['argmin'][at])
                oracle_at=w_oracle[at]
                profile_at=res['min_path'][at]
                gap=float(np.log(max(oracle_at,1e-300))-np.log(max(profile_at,1e-300)))
                put(label,res['alarm'],time.perf_counter()-t,profile_exact=1,screened_non_alarm=0,
                    num_cells=res['num_cells'],argmin_at_end=arg,argmin_dist=abs(arg-a0),log_oracle_gap_end=gap,**extras)
            except Exception as e:
                put(label,-1,time.perf_counter()-t,profile_exact=0,screened_non_alarm=0,error=type(e).__name__+': '+str(e),**extras)
    return rows

def main():
    if len(sys.argv)<5:raise SystemExit('Usage: python run_gate2.py dist healthy|weak|strong start count [methods]')
    dist,scenario=sys.argv[1:3];start=int(sys.argv[3]);count=int(sys.argv[4]);
    methods=sys.argv[5].split(',') if len(sys.argv)>5 else ['oracle','plugin','k8','unweighted','weighted']
    if dist not in DIST or scenario not in SCENARIOS:raise ValueError('dist/scenario')
    with RAW.open('a',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        if f.tell()==0:writer.writeheader()
        t0=time.time()
        for i in range(start,start+count):
            rows=do_trial(dist,scenario,i,methods)
            writer.writerows(rows)
            if (i-start+1)%10==0:
                f.flush()
                print(dist,scenario,'DONE',i-start+1,'/',count,'last',[(r['method'],r['alarm'],round(r['seconds'],2)) for r in rows], 'elapsed',round(time.time()-t0,1),flush=True)
    print('COMPLETE',dist,scenario,start,count,'sec',round(time.time()-t0,2),flush=True)

if __name__=='__main__':main()
