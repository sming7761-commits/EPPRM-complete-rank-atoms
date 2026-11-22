"""High-throughput FULL-rank-atom CCTM research profiler, general-position floats.

Unlike the old open-interval-only engine, carries OPEN and SINGLETON states,
including both domain endpoints. A transversal new root has its own state
and a tie rank chosen by calibration-first <= online. Uses a linear merge
of old parameter atoms and newly sorted rank-changing roots.

All-zero factors/degenerate or numerically ambiguous roots are explicitly
reported as UNKNOWN (exception). Small-input exact-rational fallback belongs
in generic_profile.py / cctm_atoms.py; this is NOT certified floating point.
It is an independent formula-level implementation, not authors' source.
"""
from __future__ import annotations
import math
import time
import numpy as np
from numba import njit

class NumericUnknown(ArithmeticError):pass

@njit(cache=True)
def _update_one(M, eta, A, rank, n, eps, D, C, sm):
    u=(rank-1.)/n
    norm=.5+(1.+sm)*eps
    ebet=eta if abs(eta)>=C else 0.
    if sm>0.:
        b=1.+(ebet*(u-.5)-math.sqrt(ebet*ebet+sm*sm)*eps)/norm
        deriv=((u-.5)-eta*eps/math.sqrt(eta*eta+sm*sm))/norm
        dot=(eta*(u-.5)-math.sqrt(eta*eta+sm*sm)*eps)/norm
        g=deriv/(1.+dot)
    else:
        ebsgn=1. if ebet>0. else (-1. if ebet<0. else 0.)
        esgn=1. if eta>0. else (-1. if eta<0. else 0.)
        b=1.+ebet*(u-.5-ebsgn*eps)/norm
        v=(u-.5-esgn*eps)/norm
        g=v/(1.+eta*v)
    if b<=0 or not math.isfinite(b) or not math.isfinite(g):
        raise ValueError('bad CCTM factor or ONS gradient')
    A=A+g*g
    e=eta+(2/(2-math.log(3.)))*g/A
    if e>D:e=D
    if e< -D:e=-D
    return M*b,e,A

@njit(cache=True)
def _merge_atoms(edges,PM,PE,PA,OM,OE,OA,roots,incs,rank_lo,n,eps,D,C,sm):
    B=len(edges)+len(roots)
    nb=np.empty(B,dtype=np.float64)
    npM=np.empty(B,dtype=np.float64); npE=np.empty(B,dtype=np.float64);npA=np.empty(B,dtype=np.float64)
    noM=np.empty(B-1,dtype=np.float64);noE=np.empty(B-1,dtype=np.float64);noA=np.empty(B-1,dtype=np.float64)
    oi=0;ri=0;rank=rank_lo;minv=1.7976931348623157e308
    for j in range(B):
        use_old=oi<len(edges) and (ri>=len(roots) or edges[oi]<roots[ri])
        if use_old:
            nb[j]=edges[oi]
            if ri<len(roots) and roots[ri]==edges[oi]:raise ValueError('root coincides with old singleton')
            v,e,a=_update_one(PM[oi],PE[oi],PA[oi],rank,n,eps,D,C,sm)
            oi+=1
        else:
            if oi<=0 or oi>=len(edges):raise ValueError('new root outside old intervals')
            nb[j]=roots[ri]
            point_rank=rank+(1 if incs[ri]>0 else 0)
            v,e,a=_update_one(OM[oi-1],OE[oi-1],OA[oi-1],point_rank,n,eps,D,C,sm)
            rank+=incs[ri]
            ri+=1
        npM[j]=v;npE[j]=e;npA[j]=a
        if v<minv:minv=v
        if j<B-1:
            if oi<1 or oi>len(OM):raise ValueError('old open parent out of range')
            if rank<1 or rank>n+1:raise ValueError('rank outside valid range')
            v,e,a=_update_one(OM[oi-1],OE[oi-1],OA[oi-1],rank,n,eps,D,C,sm)
            noM[j]=v;noE[j]=e;noA[j]=a
            if v<minv:minv=v
    if oi!=len(edges) or ri!=len(roots):raise ValueError('merge failed to consume all roots')
    if rank<1 or rank>n+1:raise ValueError('ending rank out of bounds')
    return nb,npM,npE,npA,noM,noE,noA,minv


def _newroots(caly,calx,y,x,lo,hi,old_edges):
    n=len(caly)
    numer=np.r_[y-caly,y+caly]
    denom=np.r_[x-calx,x+calx]
    # Exact zero/zero factor is an equality for ALL a, no root event.
    zeros=(denom==0)
    if np.any(zeros & (numer!=0)):pass  # invalid factor yields no root
    candidate=(~zeros)
    # Extremely small nonzero denominators are resolved only in the exact reference engine.
    if np.any((np.abs(denom[candidate]) < 1e-14) & (np.abs(numer[candidate])<1e-8)):
        raise NumericUnknown('near-degenerate denominator/root; exact rational fallback required')
    rv=np.divide(numer,denom,out=np.full(2*n,np.nan),where=candidate)
    if np.any(rv[candidate]==lo) or np.any(rv[candidate]==hi):
        raise NumericUnknown('root on domain endpoint; exact fallback required')
    valid=candidate & np.isfinite(rv)& (rv>lo)&(rv<hi)
    roots=rv[valid];inds=(np.flatnonzero(valid)%n).astype(np.int64)
    deriv=-2*x*(y-roots*x)+2*calx[inds]*(caly[inds]-roots*calx[inds])
    if np.any(~np.isfinite(deriv)) or np.any(np.abs(deriv)<1e-13):
        raise NumericUnknown('tangent / ill-conditioned equality root')
    ix=np.argsort(roots,kind='stable')
    roots=roots[ix];incs=np.sign(deriv[ix]).astype(np.int32)
    if len(roots)>1 and np.any(np.diff(roots)<=0):
        raise NumericUnknown('repeated simultaneous roots: exact fallback required')
    if len(roots):
        indexes=np.searchsorted(old_edges,roots)
        prev=old_edges[np.clip(indexes-1,0,len(old_edges)-1)]
        nextv=old_edges[np.clip(indexes,0,len(old_edges)-1)]
        if np.any(roots<=prev) or np.any(roots>=nextv):
            raise NumericUnknown('root collided with previously stored atom')
        # Only reject numerically indistinguishable adjacent positions;
        # do not merge distinct roots with np.isclose.
        thresh=8*np.finfo(float).eps*np.maximum(1.,np.abs(roots))
        if np.any(np.minimum(roots-prev,nextv-roots)<=thresh):
            raise NumericUnknown('root too close to old atom to certify binary64 ordering')
        if len(roots)>1 and np.any(np.diff(roots)<=np.maximum(thresh[1:],thresh[:-1])):
            raise NumericUnknown('near-duplicate roots, exact sorting required')
    return roots,incs


def cctm_fullatom_stream(caly,calx,ony,onx,lo=-.999,hi=.999,delta=.01,alpha_det=.04,
                         D=.5,C=.1,smooth=1e-6,stop_at_alarm=True,full_path=False,
                         allow_small_exact_fallback=True):
    if not (0<delta<=1 and D>0 and 0<=C and smooth>=0 and lo<hi):raise ValueError('invalid settings')
    cy=np.asarray(caly,dtype=np.float64);cx=np.asarray(calx,dtype=np.float64)
    oy=np.asarray(ony,dtype=np.float64);ox=np.asarray(onx,dtype=np.float64)
    if len(cy)==0 or len(cy)!=len(cx) or len(oy)!=len(ox):raise ValueError('shape mismatch')
    if not (np.all(np.isfinite(cy)) and np.all(np.isfinite(cx)) and np.all(np.isfinite(oy)) and np.all(np.isfinite(ox))):raise ValueError('nonfinite')
    n=len(cy)
    eps=math.sqrt(math.log(2/delta)/(2*n)) if delta<1 else 0.
    E=np.array([lo,hi],dtype=np.float64)
    PM=np.ones(2);PE=np.zeros(2);PA=np.ones(2)
    OM=np.ones(1);OE=np.zeros(1);OA=np.ones(1)
    path=[1.]; counts=[3];root_counts=[];argmins=[np.nan];alarm=0;t_start=time.perf_counter()
    for t,(y,x) in enumerate(zip(oy,ox),1):
        try:
            rr,incs=_newroots(cy,cx,y,x,lo,hi,E)
            rlo=1+int(np.count_nonzero(np.abs(cy-lo*cx)<=abs(y-lo*x)))
            E,PM,PE,PA,OM,OE,OA,minv=_merge_atoms(E,PM,PE,PA,OM,OE,OA,rr,incs,rlo,n,eps,D,C,smooth)
        except (ValueError,FloatingPointError,ArithmeticError) as ex:
            if allow_small_exact_fallback and n*len(oy)<=4096:
                from cctm_atoms import stream as rational_stream
                # Deterministic entire-trajectory exact-reference fallback,
                # using the exact binary fractions of original float inputs.
                z=rational_stream(cy,cx,oy,ox,domain=(lo,hi),threshold=1/alpha_det,
                    max_atoms=4*n*len(oy)+3,delta=delta,D=D,C=C,smooth=smooth)
                return dict(alarm=z['alarm'],
                    profile_path=np.asarray([float(q) for q in z['path']]) if full_path else None,
                    atoms=z['counts'][-1],processed=len(oy),
                    new_roots=None,atom_counts=z['counts'] if full_path else None,
                    seconds=time.perf_counter()-t_start,
                    status='SMALL_INPUT_EXACT_BINARY_RATIONAL_FALLBACK',
                    numerical_certified=False)
            raise NumericUnknown(f'UNKNOWN at time {t}: {ex}') from ex
        if minv<0 or not np.isfinite(minv):raise NumericUnknown(f'UNKNOWN wealth at time {t}')
        path.append(float(minv));counts.append(len(PM)+len(OM));root_counts.append(len(rr))
        if full_path:
            ip=int(np.argmin(PM));io=int(np.argmin(OM))
            argmins.append(float(E[ip]) if PM[ip]<=OM[io] else float((E[io]+E[io+1])*0.5))
        if not alarm and minv>=1/alpha_det:
            alarm=t
            if stop_at_alarm:break
    return dict(alarm=alarm,profile_path=np.asarray(path) if full_path else None,
                atoms=len(PM)+len(OM),processed=len(root_counts),
                new_roots=root_counts if full_path else None,
                atom_counts=counts if full_path else None,
                worst_parameters=np.asarray(argmins) if full_path else None,
                seconds=time.perf_counter()-t_start,
                status='GENERAL_POSITION_FLOAT_VERIFIED_BY_SMALL_REFERENCE',
                numerical_certified=False)

if __name__=='__main__':
    r=np.random.default_rng(33)
    cy=r.normal(size=8);cx=r.normal(size=8);oy=r.normal(size=6);ox=r.normal(size=6)
    a=cctm_fullatom_stream(cy,cx,oy,ox,stop_at_alarm=False,full_path=True)
    print(a)
