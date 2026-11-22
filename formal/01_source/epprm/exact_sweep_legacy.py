"""Full rank-cell 1D hard profile, without point-grid substitution.
Sweepline on all roots |y_on-a*x_on|=|y_cal-a*x_cal|.
ONS recalculated on every distinct open cell. Generic roots are simple;
at root ties ranks equal the higher of adjacent cell ranks, so the set of
rank paths on open cells includes all boundary-tie paths w.p. 1 under
continuous independent innovations. Independent identification boundaries
also have zero probability of coinciding with score equality roots.

Results report FLOATING-POINT exact cell enumeration, not formal interval
arithmetic verification. When roots coincide (rounded), this routine raises.
"""
import numpy as np
from numba import njit
from prm_core import oracle_prm

@njit(cache=True)
def _scan(R, t_events, deltas, h, N, threshold):
    T=len(R)
    H=N+1
    hsum=0.
    for i in range(H):hsum+=h[i]
    min_paths=np.full(T+1,np.inf)
    min_arg=np.zeros(T+1, np.int64)
    
    for k in range(len(t_events)+1):
        if k:
            j=t_events[k-1]
            R[j]+=deltas[k-1]
            if not(1<=R[j]<=N+1):raise ValueError('rank sweep out of range')
        lam=0.;A=1.; seen=0.;M=1.
        if M<min_paths[0]:min_paths[0]=M;min_arg[0]=k
        for t in range(T):
            center=(hsum+seen)/(N+t+1)
            v=h[R[t]-1]
            z=v-center
            bound=max(abs(h[0]-center),abs(h[-1]-center))
            b=min(1.,.75/max(bound,1e-12))
            if lam>b:lam=b
            if lam<-b:lam=-b
            fac=1.+lam*z
            M*=fac
            if M<min_paths[t+1]:min_paths[t+1]=M;min_arg[t+1]=k
            g=z/fac;A+=g*g; seen+=v
            cent2=(hsum+seen)/(N+t+2)
            bound2=max(abs(h[0]-cent2),abs(h[-1]-cent2))
            b2=min(1.,.75/max(bound2,1e-12))
            lam=max(-b2,min(b2,lam+4.5*g/A))
    return min_paths,min_arg

def _roots_with_events(caly,calx,ony,onx,lo,hi):
    n=len(caly);T=len(ony)
    Y=ony[:,None];X=onx[:,None];C=caly[None,:];Q=calx[None,:]
    roots=[];events=[]
    for num,den in [(Y-C,X-Q),(Y+C,X+Q)]:
        r=np.broadcast_to(np.divide(num,den,out=np.full((T,n),np.nan),where=np.abs(den)>1.e-14),(T,n))
        ok=(r>lo)&(r<hi)&np.isfinite(r)
        t,i=np.where(ok)
        a=r[t,i]
        res_on=ony[t]-a*onx[t]
        res_cal=caly[i]-a*calx[i]
        derivative=-2.*onx[t]*res_on+2.*calx[i]*res_cal
        if np.any(np.abs(derivative)<1.e-12):raise ArithmeticError('Tangent score tie root')
        roots.append(a)
        events.append(np.stack((t,np.sign(derivative).astype(np.int32)),axis=1))
    if not roots:return np.empty(0),np.empty((0,2),np.int32)
    roots=np.concatenate(roots)
    events=np.vstack(events)
    order=np.argsort(roots)
    roots=roots[order];events=events[order]
    # Near-coincidence is a distinct case; avoid pretending arbitrary tie-order is exact.
    if len(roots)>1 and np.any(np.diff(roots)==0.0):
        raise ArithmeticError('near-coincident score roots; requires separate grouping')
    return roots,events

def exact_profile(caly,calx,ony,onx,intervals,threshold=25., max_roots=1500000):
    caly=np.asarray(caly,float); calx=np.asarray(calx,float)
    ony=np.asarray(ony,float); onx=np.asarray(onx,float)
    N=len(caly);T=len(ony)
    h=np.arange(N+1,dtype=float)/N - 0.5
    best=np.full(T+1,np.inf); best_param=np.full(T+1,np.nan)
    total_roots=0
    for lo,hi in intervals:
        root,events=_roots_with_events(caly,calx,ony,onx,lo,hi)
        total_roots+=len(root)
        if total_roots>max_roots:raise RuntimeError('exact enumeration cap')
        aa=np.r_[lo,root,hi]
        pos=(aa[:-1]+aa[1:])*.5
        # At left endpoint, probability of tie is zero under continuous innovation;
        # initial midpoint before first root defines first open cell.
        a_init=pos[0]
        cal=np.sort(np.abs(caly-a_init*calx))
        R=(np.searchsorted(cal,np.abs(ony-a_init*onx),side='right')+1).astype(np.int32)
        w, idx=_scan(R,events[:,0].astype(np.int32),events[:,1].astype(np.int32),h,N,threshold)
        keep=w<best
        best[keep]=w[keep]
        best_param[keep]=pos[idx[keep]]
    hit=np.flatnonzero(best>=threshold)
    return dict(alarm=int(hit[0]) if len(hit) else 0, min_path=best,argmin=best_param,
                num_roots=total_roots,num_cells=total_roots+len(intervals))

def prove_no_alarm_by_witness(caly,calx,ony,onx,intervals,alpha=.04, num=12):
    """CERTIFIED no alarm if ANY candidate in C stays below threshold for all t.
    Does not approximate infimum. Returns (is_no_alarm_proved, witness_param)
    """
    for a,b in intervals:
        for w in np.linspace(.05,.95,num):
            x=a+(b-a)*w
            p=oracle_prm(np.abs(caly-x*calx),np.abs(ony-x*onx),'order_abs')
            if np.max(p)<1/alpha: return True,float(x)
    return False,np.nan

def selftest():
    from simulate_core import ar_path
    rng=np.random.default_rng(12)
    for repeat in range(8):
        cal,_=ar_path(rng,.9,16,'gaussian')
        on,_=ar_path(rng,.9,18,'gaussian',1.5,9)
        C=[(.73,.96)]
        res=exact_profile(cal[1:],cal[:-1],on[1:],on[:-1],C)
        from rank_cells import profile_1d
        old=profile_1d(cal[1:],cal[:-1],on[1:],on[:-1],C,'order_abs',max_candidates=100000)
        assert np.allclose(res['min_path'],old['lower'],atol=1.e-10,rtol=1.e-9),(repeat,abs(res['min_path']-old['lower']).max())
        assert res['num_cells']>0
    print('exact root sweep vs all breakpoints+equality points: 8/8 PASS')

if __name__=='__main__':selftest()

