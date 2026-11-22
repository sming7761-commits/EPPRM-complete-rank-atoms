"""Fast exact generic continuous-case streaming rank-cell PRM.
Merges prior ordered boundaries with new sorted rank roots in linear work.
All existing ONS state is inherited by new subcells before current update.
No parameter grid, approximate optimization or future observations.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from numba import njit
from stream_profile import root_events_step

@njit(cache=True)
def _merge_update(edges, roots, increments, M0,L0,A0,S0,baserank, n,t):
    zlen=len(edges)-1+len(roots)
    ne=np.empty(zlen+1,dtype=np.float64);ne[0]=edges[0]
    NM=np.empty(zlen);NL=np.empty(zlen);NA=np.empty(zlen);NS=np.empty(zlen)
    oldpos=1; rtpos=0;rcum=0
    minval=np.inf;argidx=-1
    # For order feature h(r)=(r-1)/n - 0.5, hsum == 0 (n+1 equally-spaced points).
    for j in range(zlen):
        par=oldpos-1
        rank=baserank+rcum
        if rank<1 or rank>n+1: raise ValueError('Bad rank')
        # process current rank using state inherited from corresponding parent interval
        lam=L0[par]; a=A0[par];seen=S0[par]; wealth=M0[par]
        center=seen/(n+t+1)
        bd=max(abs(-.5-center),abs(.5-center))
        bound=min(1.,.75/max(bd,1e-12))
        lam=max(-bound,min(bound,lam))
        z=(rank-1)/n-.5-center
        factor=1.+lam*z
        if factor<.25-1e-10: raise ValueError('Betting factor')
        wealth*=factor
        g=z/factor;a+=g*g;seen+=(rank-1)/n-.5
        nextcent=seen/(n+t+2)
        nextbd=max(abs(-.5-nextcent),abs(.5-nextcent))
        nextbound=min(1.,.75/max(nextbd,1e-12))
        lam=max(-nextbound,min(nextbound,lam+4.5*g/a))
        NM[j]=wealth;NL[j]=lam;NA[j]=a;NS[j]=seen
        if wealth<minval:minval=wealth;argidx=j
        # extend current rank cell until next old boundary or new root
        takeold = oldpos<len(edges) and (rtpos>=len(roots) or edges[oldpos]<roots[rtpos])
        if takeold:
            v=edges[oldpos];oldpos+=1
        else:
            if rtpos>=len(roots): raise ArithmeticError('Root merge exhausted')
            v=roots[rtpos];rcum+=increments[rtpos];rtpos+=1
        if not v>ne[j]:raise ArithmeticError('Overlapping / duplicate boundaries')
        ne[j+1]=v
    if ne[zlen]!=edges[len(edges)-1]:raise ArithmeticError('Bad end')
    return ne,NM,NL,NA,NS,minval,argidx

def stream_exact(caly,calx,ony,onx,alpha=.05,lo=-.999,hi=.999,stop_at_alarm=True,full_path=False):
    caly=np.asarray(caly,dtype=np.float64);calx=np.asarray(calx,dtype=np.float64)
    ony=np.asarray(ony,dtype=np.float64);onx=np.asarray(onx,dtype=np.float64)
    if len(caly)!=len(calx) or len(ony)!=len(onx):raise ValueError('Length mismatch')
    n=len(caly);T=len(ony)
    edges=np.array([lo,hi],dtype=np.float64)
    M=np.ones(1);lam=np.zeros(1);A=np.ones(1);seen=np.zeros(1)
    path=np.ones(T+1) if full_path else None
    cells=[];increments=[];max_mem=0;alarm=0;argmin=np.nan
    for t in range(T):
        roots,ds=root_events_step(caly,calx,ony[t],onx[t],lo,hi)
        r0=1+int(np.count_nonzero(np.abs(caly-lo*calx)<=abs(ony[t]-lo*onx[t])))
        edges,M,lam,A,seen,v,k=_merge_update(edges,roots,ds,M,lam,A,seen,r0,n,t)
        ncel=len(M)
        max_mem=max(max_mem,edges.nbytes+M.nbytes+lam.nbytes+A.nbytes+seen.nbytes)
        cells.append(ncel);increments.append(len(roots))
        if full_path:path[t+1]=v
        if v>=1/alpha:
            if alarm==0: alarm=t+1
            argmin=(edges[k]+edges[k+1])*.5
            if stop_at_alarm:break
    return dict(alarm=alarm,profile_path=path,processed=len(cells),num_cells=len(M),new_roots=np.array(increments),n_cells=np.array(cells),max_mem_bytes=max_mem,argmin=argmin)

def selftest():
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from simulate_core import ar_path
    from exact_sweep import exact_profile
    from stream_profile import streaming_profile
    rng=np.random.default_rng(1225)
    for n,T in [(12,16),(25,30),(50,50)]:
      for r in range(3):
        ca,_=ar_path(rng,.9,n,'gaussian');on,_=ar_path(rng,.9,T,'gaussian',1.5,T//2)
        v=stream_exact(ca[1:],ca[:-1],on[1:],on[:-1],stop_at_alarm=False,full_path=True)
        w=streaming_profile(ca[1:],ca[:-1],on[1:],on[:-1],stop_on_alarm=False,save_path=True)
        ex=exact_profile(ca[1:],ca[:-1],on[1:],on[:-1],[(-.999,.999)],threshold=20.)
        assert np.allclose(v['profile_path'],w['path'],atol=1e-8,rtol=1e-8)
        assert np.allclose(v['profile_path'],ex['min_path'],atol=1e-8,rtol=1e-8)
        assert v['num_cells']==w['n_cells']
    print('FAST STREAMING vs BATCH vs STREAM: 9/9 PASS')
if __name__=='__main__':selftest()
