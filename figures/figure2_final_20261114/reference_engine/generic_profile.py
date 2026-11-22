"""Generic finite-dimensional-state rank-wealth composite profiler: rational correctness oracle.

Genericity is operational: the caller supplies arbitrary finite-dimensional state
and a deterministic transition depending on the current fixed-reference rank.
Root events/atom inheritance are shared by PRM and CCTM. Exact rational input
partition, plus backend-specific wealth arithmetic. This reference is intended
for small correctness cases, not 736x600 production performance.

No future observations used; singleton root points and domain endpoints retained.
"""
from __future__ import annotations
from fractions import Fraction as F
from bisect import bisect_right
from dataclasses import dataclass
from typing import Callable
from atom_exact import Q, rank_at, merge_sorted_unique, equality_roots
from atom_sweep import sweep_current, ranks_for_partition

@dataclass(frozen=True)
class Backend:
    name: str
    initial: Callable
    transition: Callable  # signature (state, rank, N, zero_based_t) -> state
    wealth: Callable


def stream(caly, calx, ony, onx, backend:Backend, domain=(F(-999,1000),F(999,1000)), max_atoms=2000):
    cy,cx,oy,ox = [tuple(map(Q,s)) for s in (caly,calx,ony,onx)]
    lo,hi=map(Q,domain)
    assert 0<len(cy)==len(cx) and len(oy)==len(ox) and lo<hi
    breaks=[lo,hi]; p={lo:backend.initial(),hi:backend.initial()}; op=[backend.initial()]
    path=[backend.wealth(p[lo])]; counts=[3]
    for t,(y,x) in enumerate(zip(oy,ox)):
        nr,openrank,pointmap=sweep_current(cy,cx,y,x,lo,hi)
        b=merge_sorted_unique(breaks,nr)
        if 2*len(b)-1>max_atoms:raise OverflowError('atom limit')
        pr,orr=ranks_for_partition(nr,openrank,pointmap,b)
        pp={};oo=[]; old_idx=0
        for k,a in enumerate(b):
            while old_idx+1<len(breaks) and breaks[old_idx+1]<=a:old_idx+=1
            source=p[a] if a in p else op[old_idx]
            pp[a]=backend.transition(source,pr[k],len(cy),t)
            if k+1<len(b):
                oo.append(backend.transition(op[old_idx],orr[k],len(cy),t))
        breaks,p,op=b,pp,oo
        mn=min([backend.wealth(s) for s in p.values()]+[backend.wealth(s) for s in op])
        path.append(mn);counts.append(len(p)+len(op))
    return dict(profile=path,atoms=counts,points=p,opens=op)


def batch(caly,calx,ony,onx,backend:Backend,domain=(F(-999,1000),F(999,1000)),max_atoms=2000):
    cy,cx,oy,ox = [tuple(map(Q,s)) for s in (caly,calx,ony,onx)]
    lo,hi=map(Q,domain)
    roots={lo,hi};path=[backend.wealth(backend.initial())];counts=[3]
    for t,(y,x) in enumerate(zip(oy,ox)):
        roots.update(equality_roots(cy,cx,y,x,lo,hi))
        b=sorted(roots)
        if 2*len(b)-1>max_atoms:raise OverflowError('atom limit')
        reps=[]
        for j,z in enumerate(b):
            reps.append(z)
            if j+1<len(b):reps.append((z+b[j+1])/2)
        wealth=[]
        for a in reps:
            ss=backend.initial()
            for it,(yy,xx) in enumerate(zip(oy[:t+1],ox[:t+1])):
                rank=rank_at(cy,cx,yy,xx,a)
                ss=backend.transition(ss,rank,len(cy),it)
            wealth.append(backend.wealth(ss))
        path.append(min(wealth));counts.append(len(reps))
    return dict(profile=path,atoms=counts)


def prm_backend():
    from atom_exact import State,advance
    return Backend('EP-PRM-Order',State,advance,lambda s:s.wealth)


def cctm_backend(D=F(1,2),C=F(1,10),smooth=F(1,1000000),delta=F(1,100)):
    from cctm_atoms import initial,advance
    return Backend('CCTM-independent-formula',initial,
                   lambda state,rank,n,t:advance(state,rank,n,delta=delta,D=D,C=C,smooth=smooth),
                   lambda s:s.wealth)
