"""Pre-registered 100/100 paired preview. Uses exact Gate2 trajectory seeds and 
reference paths. No hyperparameter choice uses faults. CCTM default is audited 
for possible nonpositive factors. Fixed-witness shortcut is EXACT as a nonalarm
implication, but is not a fully floating-point certified profile.
"""
from __future__ import annotations
import csv, json, math, time, sys, hashlib, bisect
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parent
SRC=ROOT/'engine'
sys.path.insert(0,str(SRC))
from simulate_core import ar_path
from prm_core import oracle_prm, first_alarm
from modern_baselines import cctm_wealth
from full_domain_cctm_atoms_fast import cctm_fullatom_stream, NumericUnknown

DIST=['gaussian','t5','t10','exp','skew_mix']
RUN=['gaussian','t5','exp','skew_mix']
TAU=357;T=600;N=736;ALPHA=.05
# Computational *nonalarm certificates*, never approximations used for alarms.
# Pre-fixed symmetric points; any chosen point is a valid nonalarm witness.
WITNESS=np.linspace(-.999,.999,17)
WITNESS_EXTRA=np.setdiff1d(np.linspace(-.999,.999,65),WITNESS)
CSET=[('official_default',1.3,0.,0.),('fair_budget',.5,.1,1e-6)]


def make_data(dist,scene,rep):
    idx=DIST.index(dist);js={'healthy':0,'weak':1}[scene]
    seed=int(20261107000000+idx*100000000+js*10000000+rep)
    rng=np.random.default_rng(seed)
    A,_=ar_path(rng,.9,112,dist,burn=200)
    B,_=ar_path(rng,.9,624,dist,burn=200)
    on,_=ar_path(rng,.9,600,dist,scale=1. if scene=='healthy' else 1.5,fault_at=TAU,burn=200)
    cy=np.r_[A[1:],B[1:]];cx=np.r_[A[:-1],B[:-1]]
    oy=on[1:];ox=on[:-1]
    assert len(cy)==736 and len(oy)==600
    return seed,cy,cx,oy,ox


def a_ols(cy,cx):
    return float(np.dot(cy,cx)/np.dot(cx,cx))


def prm_plugin(cy,cx,oy,ox,ahat):
    return oracle_prm(np.abs(cy-ahat*cx),np.abs(oy-ahat*ox),'order_abs')


def pfa_quantile(scores,alpha=.05):
    x=np.sort(np.asarray(scores,dtype=float));n=len(x)
    rank=int(np.ceil((n+1)*(1-alpha)))
    if rank>n:return float('inf')
    return float(np.nextafter(x[rank-1],np.inf))


def gaussian_fixed_cusum(z2,s=1.5):
    inc=-math.log(s)+.5*(1-1/(s*s))*z2
    out=np.zeros(len(inc)+1)
    for i,v in enumerate(inc,1):out[i]=max(0,out[i-1]+v)
    return out


def glr_gaussian_scale(z2):
    """GLR with known normalized null variance; supremum over scale s>=1
    and all previous change-point positions. The model is Gaussian and is
    misspecified on the three non-Gaussian distributions.
    """
    z=np.asarray(z2,dtype=float)
    pref=np.r_[0.,np.cumsum(z)]
    out=np.zeros(len(z)+1)
    for t in range(1,len(z)+1):
        length=np.arange(t,0,-1)
        means=(pref[t]-pref[:t])/length
        q=np.maximum(means,1.0)
        vals=.5*length*(q-1-np.log(q))
        out[t]=np.max(vals)
    return out


def cusum_conformal(past, on, seed, c=20., power=.5):
    """Conformal-style plug-in residual-score variant; original paper's ARL
    claims do not automatically transfer after estimating the AR slope.
    """
    data=sorted(map(float,past));rng=np.random.default_rng(seed%(2**32))
    out=np.zeros(len(on)+1)
    for t,x in enumerate(on,1):
        i=bisect.bisect_right(data,float(x))
        greater=len(data)-i
        p=(greater+rng.uniform())/(len(data)+1)
        p=max(p,1e-308)
        out[t]=max(0.,out[t-1]+math.log(power)+(power-1)*math.log(p))
        bisect.insort_right(data,float(x))
    return out


def noncctm_methods(cy,cx,oy,ox,seed,thresholds):
    ahat=a_ols(cy,cx)
    resid=oy-ahat*ox
    calres=cy-ahat*cx
    sigma=float(np.sqrt(np.mean(calres**2)))
    z2=np.square(resid/max(sigma,1e-12))
    pp=prm_plugin(cy,cx,oy,ox,ahat)
    glr=glr_gaussian_scale(z2)
    gc=gaussian_fixed_cusum(z2)
    cc=cusum_conformal(np.abs(calres),np.abs(resid),seed)
    return {
        'OLS_PlugIn_capital20':(pp,20.,'NOMINAL_CAPITAL_NOT_PFA_GUARANTEED',ahat),
        'OLS_PlugIn_empiricalPFA':(pp,thresholds['OLS_PlugIn_empiricalPFA'],'EMPIRICAL_PFA_600',ahat),
        'Gaussian_GLR_empiricalPFA':(glr,thresholds['Gaussian_GLR_empiricalPFA'],'EMPIRICAL_PFA_600_GAUSSIAN_SCORE',ahat),
        'Gaussian_CUSUM_empiricalPFA':(gc,thresholds['Gaussian_CUSUM_empiricalPFA'],'EMPIRICAL_PFA_600_GAUSSIAN_SCORE',ahat),
        'Conformal_CUSUM_ARL_c20':(cc,math.log(20),'ARL_ORIGINAL_ONLY_NOT_PLUGIN_GUARANTEE',ahat),
        'Conformal_CUSUM_empiricalPFA':(cc,thresholds['Conformal_CUSUM_empiricalPFA'],'EMPIRICAL_PFA_600',ahat),
    }


def calibrate(dist,n=500):
    t0=time.perf_counter();d={'dist':dist,'n_calibration':n,'seed_prefix':20261001000000,'seed_range':'20261001000000+distindex*1e8+rep','alpha':.05}
    scores={x:[] for x in ['OLS_PlugIn_empiricalPFA','Gaussian_GLR_empiricalPFA','Gaussian_CUSUM_empiricalPFA','Conformal_CUSUM_empiricalPFA']}
    for rep in range(n):
        # Predefined independent calibration seed block, disjoint from all Gate2 tested seeds.
        seed=int(20261001000000+DIST.index(dist)*100000000+rep)
        rng=np.random.default_rng(seed)
        A,_=ar_path(rng,.9,112,dist,burn=200)
        B,_=ar_path(rng,.9,624,dist,burn=200)
        on,_=ar_path(rng,.9,600,dist,burn=200)
        cy=np.r_[A[1:],B[1:]];cx=np.r_[A[:-1],B[:-1]]
        oy=on[1:];ox=on[:-1]
        ahat=a_ols(cy,cx);calres=cy-ahat*cx;resid=oy-ahat*ox
        sigma=np.sqrt(np.mean(calres**2))
        z2=np.square(resid/max(sigma,1e-12))
        scores['OLS_PlugIn_empiricalPFA'].append(float(np.max(prm_plugin(cy,cx,oy,ox,ahat))))
        scores['Gaussian_GLR_empiricalPFA'].append(float(np.max(glr_gaussian_scale(z2))))
        scores['Gaussian_CUSUM_empiricalPFA'].append(float(np.max(gaussian_fixed_cusum(z2))))
        scores['Conformal_CUSUM_empiricalPFA'].append(float(np.max(cusum_conformal(np.abs(calres),np.abs(resid),seed))))
        if (rep+1)%100==0:print('CAL',dist,rep+1,'/',n,'seconds',round(time.perf_counter()-t0,1),flush=True)
    for m,v in scores.items():d[m]=pfa_quantile(v)
    d['elapsed_seconds']=round(time.perf_counter()-t0,3)
    return d


def cctm_witness(cy,cx,oy,ox,D,C,sm,threshold=25):
    # Each *fixed* candidate path is computed in full, not refreshed or point-estimated.
    # If for every t some witnessed path has wealth < threshold, the inf must be below.
    # This is a logically sufficient NON-ALARM certificate, not an alarm certificate.
    minpath=np.ones(T+1)*np.inf; failures=[];candidates=0
    for a in WITNESS:
        try:
            curve=cctm_wealth(np.abs(cy-a*cx),np.abs(oy-a*ox),delta=.01,D=D,C=C,smooth=sm)
            if not np.all(np.isfinite(curve)) or np.any(curve<0):
                raise ArithmeticError('negative or nonfinite wealth')
            minpath=np.minimum(minpath,curve)
            candidates+=1
        except Exception as e:
            failures.append((float(a),type(e).__name__,str(e)))
    # Additional pre-fixed witnesses can avoid an expensive all-atom scan; they
    # cannot create a false ALARM because they are used only to certify NO ALARM.
    if np.any(minpath >= threshold-1e-7):
        for a in WITNESS_EXTRA:
            try:
                curve=cctm_wealth(np.abs(cy-a*cx),np.abs(oy-a*ox),delta=.01,D=D,C=C,smooth=sm)
                if not np.all(np.isfinite(curve)) or np.any(curve<0):raise ArithmeticError('negative or nonfinite wealth')
                minpath=np.minimum(minpath,curve);candidates+=1
                if np.all(minpath < threshold-1e-7):break
            except Exception as e:
                failures.append((float(a),type(e).__name__,str(e)))
    # Conservative float margin: do not certify if near threshold.
    is_noalarm=bool(np.all(minpath < threshold-1e-7))
    return is_noalarm,failures,candidates,float(np.max(minpath))


def cctm_profile(cy,cx,oy,ox,which):
    D,C,sm=next((D,C,s) for label,D,C,s in CSET if label==which)
    t0=time.perf_counter()
    witness,failures,nok,maxmin=cctm_witness(cy,cx,oy,ox,D,C,sm)
    if which=='official_default' and failures:
        return {'alarm':-1,'status':'INVALID_POINT_FACTOR_IN_DEFAULT_DOMAIN','unknown_time':None,
        'witness_count':nok,'invalid_candidate_count':len(failures),'invalid_sample':repr(failures[0]),
        'atoms':None,'processed':None,'seconds':round(time.perf_counter()-t0,5)}
    if witness:
        return {'alarm':0,'status':'WITNESS_NOALARM','unknown_time':None,
            'witness_count':nok,'invalid_candidate_count':len(failures),'invalid_sample':'',
            'atoms':None,'processed':600,'seconds':round(time.perf_counter()-t0,5)}
    try:
        z=cctm_fullatom_stream(cy,cx,oy,ox,delta=.01,D=D,C=C,smooth=sm,alpha_det=.04,stop_at_alarm=True,full_path=False)
        return {'alarm':int(z['alarm']),'status':z['status'],'unknown_time':None,
            'witness_count':nok,'invalid_candidate_count':len(failures),'invalid_sample':repr(failures[:1]),
            'atoms':z['atoms'],'processed':z['processed'],'seconds':round(time.perf_counter()-t0,5)}
    except NumericUnknown as e:
        msg=str(e)
        return {'alarm':-1,'status':'UNKNOWN','unknown_time':int(msg.split('time ')[1].split(':')[0]) if 'time ' in msg else None,
            'witness_count':nok,'invalid_candidate_count':len(failures),'invalid_sample':repr(failures[:1]),
            'atoms':None,'processed':None,'seconds':round(time.perf_counter()-t0,5),'error':msg}
    except Exception as e:
        return {'alarm':-1,'status':'ERROR','unknown_time':None,
            'witness_count':nok,'invalid_candidate_count':len(failures),'invalid_sample':repr(failures[:1]),
            'atoms':None,'processed':None,'seconds':round(time.perf_counter()-t0,5),'error':repr(e)}


def alarm_row(dist,scene,rep,seed,method,alarm,guarantee,threshold,secs=0.,**kw):
    a=int(alarm)
    return dict(dist=dist,scenario=scene,rep=rep,seed=seed,method=method,alarm=a,
         early=int(0<a<=TAU),detected=int(a>TAU) if scene=='weak' else int(a>0),
         cap_delay=(a-TAU if scene=='weak' and a>TAU else 244),
         threshold=str(threshold),guarantee=guarantee,seconds=round(secs,5),
         status=kw.pop('status','COMPLETED'),unknown_time=kw.pop('unknown_time',None),**kw)


def save_rows(rows,f):
    import pandas as pd
    d=pd.DataFrame(rows);d.to_csv(f,index=False)


def main():
    if len(sys.argv)<2:raise SystemExit('preaccept.py calibration|run dist scene start count')
    cmd=sys.argv[1]
    if cmd=='calibration':
        dists=RUN if len(sys.argv)<3 else [sys.argv[2]]
        pp=ROOT/'independent_calibration.json';old=json.loads(pp.read_text()) if pp.exists() else {}
        for d in dists:
            old[d]=calibrate(d,500)
            pp.write_text(json.dumps(old,indent=2,ensure_ascii=False))
        return
    if cmd!='run' or len(sys.argv)!=6:raise SystemExit('preaccept.py run gaussian healthy 0 100')
    dist,scene,start,count=sys.argv[2],sys.argv[3],int(sys.argv[4]),int(sys.argv[5])
    if dist not in RUN or scene not in ('healthy','weak'):raise ValueError
    caljson=json.loads((ROOT/'independent_calibration.json').read_text())
    thresholds=caljson[dist];
    # Existing historical rows are regenerated to test seed alignment, not recounted as new experiments.
    hist=ROOT/'FullDomain';ep={};oracle={};old={}
    for r in csv.DictReader(open(hist/f'{scene}_{dist}.csv')):
        if r['method']=='full736':ep[int(r['rep'])]=r
    for r in csv.DictReader(open(hist/f'oracle736_{scene}_{dist}.csv')):oracle[int(r['rep'])]=r
    src=hist/'legacy_results'/f'{scene}_{dist}.csv' if scene=='weak' else ROOT/'legacy_health_raw.csv'
    for r in csv.DictReader(open(src)):
        if r['method']=='k8' and r['dist']==dist and r['scenario']==scene:old[int(r['rep'])]=r
    out=ROOT/f'{scene}_{dist}_{start}_{start+count}.csv';rows=[];t0=time.perf_counter()
    for rep in range(start,start+count):
        seed,cy,cx,oy,ox=make_data(dist,scene,rep)
        assert int(ep[rep]['seed'])==seed and int(oracle[rep]['seed'])==seed and int(old[rep]['seed'])==seed
        rows.append(alarm_row(dist,scene,rep,seed,'EP_PRM_736',int(ep[rep]['alarm']),'MARGINAL_ANYTIME_PFA_5PCT',20,
                              status='HISTORICAL_SEED_MATCHED',atoms=ep[rep]['num_cells'],legacy_runtime=ep[rep]['runtime_s']))
        rows.append(alarm_row(dist,scene,rep,seed,'Oracle_PRM_736',int(oracle[rep]['alarm']),'ORACLE_KNOWS_A0',20,
                              status='HISTORICAL_SEED_MATCHED',legacy_runtime=oracle[rep]['seconds']))
        rows.append(alarm_row(dist,scene,rep,seed,'Old_K8',int(old[rep]['alarm']),'SYMMETRIC_NOISE_ONLY;EMPIRICAL_ON_SKEW',25,
                              status='HISTORICAL_SEED_MATCHED',legacy_runtime=old[rep]['seconds']))
        t=time.perf_counter()
        methods=noncctm_methods(cy,cx,oy,ox,seed,thresholds)
        dt=(time.perf_counter()-t)/len(methods)
        for name,(path,thr,guarantee,ahat) in methods.items():
            alarm=next((i for i,x in enumerate(path) if x>=thr),0)
            rows.append(alarm_row(dist,scene,rep,seed,name,alarm,guarantee,thr,dt,ahat=ahat))
        for which in ['fair_budget','official_default']:
            r=cctm_profile(cy,cx,oy,ox,which)
            runtime=r.pop('seconds')
            name='Profiled_CCTM_'+which
            rows.append(alarm_row(dist,scene,rep,seed,name,r.pop('alarm'),
                'CCTM_DKW_delta.01_and_alpha.04' if which=='fair_budget' else 'OFFICIAL_DEFAULT_POINT_BET_UNTESTED_FULLDOMAIN',25,runtime,
                **r))
        if (rep-start+1)%10==0:
            save_rows(rows,out)
            print('RUN',dist,scene,rep-start+1,'/',count,'elapsed',round(time.perf_counter()-t0,1),
                  'cctm statuses',[(x['method'],x['status']) for x in rows[-2:]],flush=True)
    save_rows(rows,out)
    print('DONE',dist,scene,start,count,round(time.perf_counter()-t0,1),flush=True)

if __name__=='__main__':main()
