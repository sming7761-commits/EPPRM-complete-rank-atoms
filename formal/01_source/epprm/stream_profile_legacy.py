"""Exact one-dimensional open rank-cell streaming profile for absolute-residual Order PRM.
Numerically exact for generic continuous data; no parameter grid and no after-the-fact
batch reconstruction. Supports arbitrary finite offline calibration concatenated from
independent healthy innovations. It represents open cells and inherits previous ONS
state at every split. Equalities at roots must be separately audited if non-generic.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from numba import njit

@njit(cache=True)
def _advance(p, r, h, hsum, n, t, prev_M, prev_lam, prev_A, prev_seen):
    k=len(p)
    M=np.empty(k,np.float64);lam=np.empty(k,np.float64)
    A=np.empty(k,np.float64);seen=np.empty(k,np.float64)
    for i in range(k):
        j=p[i]
        l=prev_lam[j]; aa=prev_A[j]; ss=prev_seen[j];w=prev_M[j]
        center=(hsum+ss)/(n+t+1)
        bd=max(abs(h[0]-center),abs(h[n]-center))
        b=min(1.,.75/max(bd,1e-12))
        l=max(-b,min(b,l))
        z=h[r[i]-1]-center
        fac=1.+l*z
        if fac<.25-1e-9: raise ValueError('Betting bound violated')
        w*=fac
        g=z/fac;aa+=g*g;ss+=h[r[i]-1]
        center2=(hsum+ss)/(n+t+2)
        bd2=max(abs(h[0]-center2),abs(h[n]-center2))
        b2=min(1.,.75/max(bd2,1e-12))
        l=max(-b2,min(b2,l+4.5*g/aa))
        M[i]=w;lam[i]=l;A[i]=aa;seen[i]=ss
    return M,lam,A,seen


def root_events_step(caly,calx,yt,xt,lo,hi):
    """Pairwise equality roots and rank increments of current online score."""
    numer=np.r_[yt-caly,yt+caly]
    denom=np.r_[xt-calx,xt+calx]
    r=np.divide(numer,denom,out=np.full(len(numer),np.nan),where=np.abs(denom)>1e-14)
    mask=np.isfinite(r)&(r>lo)&(r<hi)
    roots=r[mask]
    if len(roots)==0:return roots,np.empty(0,np.int32)
    calindex=np.arange(len(numer))%len(caly)
    inds=calindex[mask]
    delta=-2.*xt*(yt-roots*xt)+2.*calx[inds]*(caly[inds]-roots*calx[inds])
    if np.any(np.abs(delta)<1e-13):raise ArithmeticError('Tangent root; use explicit tie handling')
    ix=np.argsort(roots,kind='stable'); roots=roots[ix]; ds=np.sign(delta[ix]).astype(np.int32)
    if np.any(np.diff(roots)==0):raise ArithmeticError('Exact duplicate roots; tie group not implemented')
    return roots,ds


def streaming_profile(caly,calx,ony,onx,lo=-.999,hi=.999,alpha=.05,
                      stop_on_alarm=True,save_path=False):
    """Updates each current parameter cell once at each online time; returns first alarm.
    Open-cell rank paths exact a.s. for continuous noise, plus endpoint/tie audit required
    for pathological rational/algebraic input. Values are double precision, not interval
    arithmetic proof certificates.
    """
    caly=np.asarray(caly,dtype=np.float64);calx=np.asarray(calx,dtype=np.float64)
    ony=np.asarray(ony,dtype=np.float64);onx=np.asarray(onx,dtype=np.float64)
    n=len(caly);T=len(ony)
    if not np.all(np.isfinite(np.r_[caly,calx,ony,onx])):raise ValueError('Nonfinite')
    h=np.arange(n+1,dtype=np.float64)/n-.5
    hsum=float(h.sum());edges=np.array([lo,hi],np.float64)
    M=np.ones(1);lam=np.zeros(1);A=np.ones(1);seen=np.zeros(1)
    path=[1.]; cells=[1];newroots=[];t_stop=T;argmin=np.nan
    for t in range(T):
        roots,delta=root_events_step(caly,calx,ony[t],onx[t],lo,hi)
        newer=np.sort(np.r_[edges,roots])
        if np.any(np.diff(newer)<=0):raise ArithmeticError('Zero-size cell')
        mid=(newer[1:]+newer[:-1])*.5
        parent=np.searchsorted(edges,mid,side='right')-1
        if np.any((parent<0)|(parent>=len(M))): raise ArithmeticError('Parent mapping')
        atlo=np.abs(ony[t]-lo*onx[t]);cal=np.sort(np.abs(caly-lo*calx))
        left_rank=int(np.searchsorted(cal,atlo,side='right')+1)
        diff=np.r_[0,np.cumsum(delta,dtype=np.int32)]
        rank=left_rank+diff[np.searchsorted(roots,mid,side='left')]
        if np.any((rank<1)|(rank>n+1)):
            raise ArithmeticError('Invalid rank from root sweep')
        M,lam,A,seen=_advance(parent.astype(np.int32),rank.astype(np.int32),h,hsum,n,t,M,lam,A,seen)
        edges=newer
        k=int(np.argmin(M));v=float(M[k]);argmin=float(mid[k])
        path.append(v);cells.append(len(M));newroots.append(len(roots))
        if stop_on_alarm and v>=1/alpha:
            t_stop=t+1;break
    return dict(alarm=t_stop if path[-1]>=1/alpha else 0,
                path=np.array(path) if save_path else None,
                n_cells=len(M),cell_history=np.array(cells) if save_path else None,
                new_roots=np.array(newroots) if save_path else None,
                argmin=argmin,processed=t_stop)


def selftest():
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from simulate_core import ar_path
    from rank_cells import profile_1d
    from exact_sweep import exact_profile
    rng=np.random.default_rng(9918)
    errs=[]
    for n,T in [(12,17),(20,25),(36,40)]:
      for i in range(3):
        cal,_=ar_path(rng,.9,n,'gaussian');on,_=ar_path(rng,.9,T,'gaussian',1.5,T//2)
        res=streaming_profile(cal[1:],cal[:-1],on[1:],on[:-1],-.999,.999,stop_on_alarm=False,save_path=True)
        ex=exact_profile(cal[1:],cal[:-1],on[1:],on[:-1],[(-.999,.999)],threshold=20.)
        vec=profile_1d(cal[1:],cal[:-1],on[1:],on[:-1],[(-.999,.999)],'order_abs',max_candidates=50000)
        err=float(np.max(np.abs(res['path']-ex['min_path'])))
        err2=float(np.max(np.abs(res['path']-vec['lower'])))
        assert np.allclose(res['path'],ex['min_path'],atol=1e-9,rtol=1e-8),(n,T,i,err)
        assert np.allclose(res['path'],vec['lower'],atol=1e-9,rtol=1e-8),(n,T,i,err2)
        errs.append([n,T,i,err,err2,res['n_cells']])
    print('STREAM↔BATCH↔BRUTE PASS',errs,flush=True)

if __name__=='__main__':selftest()
