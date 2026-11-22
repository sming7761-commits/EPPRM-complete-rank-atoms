"""Full parameter rank-atom streaming PRM (open cells + root singletons + endpoints).
Binary64 general-position implementation, conservatively raises NumericUnknown
for near ties. Reuses same exact event geometry as full-domain CCTM.
"""
from __future__ import annotations
import time
import numpy as np
from numba import njit
from full_domain_cctm_atoms_fast import _newroots, NumericUnknown

@njit(cache=True)
def _step(M,L,A,S,r,n,t):
    center=S/(n+t)
    b=max(abs(-.5-center),abs(.5-center))
    limit=min(1.,.75/max(b,1e-12))
    lam=max(-limit,min(limit,L))
    h=(r-1.)/n-.5
    z=h-center
    fac=1.+lam*z
    if fac<.25-1e-10 or not np.isfinite(fac):raise ValueError('invalid PRM factor')
    M=M*fac
    g=z/fac
    A=A+g*g
    S=S+h
    cent2=S/(n+t+1)
    b2=max(abs(-.5-cent2),abs(.5-cent2))
    limit2=min(1.,.75/max(b2,1e-12))
    L=max(-limit2,min(limit2,lam+4.5*g/A))
    return M,L,A,S

@njit(cache=True)
def _merge(edges,pm,pl,pa,ps,om,ol,oa,os,roots,inc,rlo,n,t):
    L=len(edges)+len(roots)
    ed=np.empty(L);PM=np.empty(L);PL=np.empty(L);PA=np.empty(L);PS=np.empty(L)
    OM=np.empty(L-1);OL=np.empty(L-1);OA=np.empty(L-1);OS=np.empty(L-1)
    old=0;ri=0;rank=rlo;minimum=1.7976931348623157e308;best=np.nan
    for j in range(L):
        use_old=old<len(edges) and (ri>=len(roots) or edges[old]<roots[ri])
        if use_old:
            ed[j]=edges[old]
            if ri<len(roots) and roots[ri]==edges[old]:raise ValueError('root singleton collision')
            m,l,a,s=_step(pm[old],pl[old],pa[old],ps[old],rank,n,t)
            old+=1
        else:
            if old<=0 or old>=len(edges):raise ValueError('new root outside domain')
            ed[j]=roots[ri]
            point_rank=rank+(1 if inc[ri]>0 else 0)
            m,l,a,s=_step(om[old-1],ol[old-1],oa[old-1],os[old-1],point_rank,n,t)
            rank+=inc[ri];ri+=1
        PM[j]=m;PL[j]=l;PA[j]=a;PS[j]=s
        if m<minimum:minimum=m;best=ed[j]
        if j<L-1:
            if old<1 or old>len(om):raise ValueError('parent interval mismatch')
            m,l,a,s=_step(om[old-1],ol[old-1],oa[old-1],os[old-1],rank,n,t)
            OM[j]=m;OL[j]=l;OA[j]=a;OS[j]=s
            if m<minimum:
                minimum=m
                best=(ed[j]+(edges[old] if ri>=len(roots) or edges[old]<roots[ri] else roots[ri]))/2. if old<len(edges) else ed[j]
    if old!=len(edges) or ri!=len(roots):raise ValueError('unconsumed events')
    return ed,PM,PL,PA,PS,OM,OL,OA,OS,minimum,best


def stream_fullatom(caly,calx,ony,onx,lo=-.999,hi=.999,threshold=20.):
    cy=np.asarray(caly,np.float64);cx=np.asarray(calx,np.float64)
    oy=np.asarray(ony,np.float64);ox=np.asarray(onx,np.float64)
    n=len(cy);T=len(oy)
    edges=np.array([lo,hi],float)
    pm=np.ones(2);pl=np.zeros(2);pa=np.ones(2);ps=np.zeros(2)
    om=np.ones(1);ol=np.zeros(1);oa=np.ones(1);os=np.zeros(1)
    path=np.ones(T+1);atoms=np.full(T+1,3,dtype=np.int64); argmin=np.full(T+1,np.nan)
    new_roots=np.zeros(T+1,dtype=np.int32);alarm=0;t0=time.perf_counter()
    for t,(y,x) in enumerate(zip(oy,ox),1):
        try:
            r,inc=_newroots(cy,cx,y,x,lo,hi,edges)
            rlo=1+int(np.count_nonzero(np.abs(cy-lo*cx)<=abs(y-lo*x)))
            edges,pm,pl,pa,ps,om,ol,oa,os,v,aa=_merge(edges,pm,pl,pa,ps,om,ol,oa,os,r,inc,rlo,n,t)
            if not np.isfinite(v):raise ValueError('nonfinite PRM wealth')
        except Exception as exc:
            raise NumericUnknown(f'UNKNOWN at time {t}: {exc}') from exc
        path[t]=v;atoms[t]=len(pm)+len(om)
        ip=int(np.argmin(pm));io=int(np.argmin(om))
        argmin[t]=float(edges[ip]) if pm[ip]<=om[io] else float((edges[io]+edges[io+1])/2.)
        new_roots[t]=len(r)
        if alarm==0 and v>=threshold:alarm=t
    return dict(profile_path=path,atom_counts=atoms,worst_parameter=argmin,
                new_roots=new_roots,alarm=alarm,seconds=time.perf_counter()-t0,
                status='GENERAL_POSITION_FLOAT_NOT_FORMALLY_CERTIFIED')
