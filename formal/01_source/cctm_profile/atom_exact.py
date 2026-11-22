"""Exact rational reference for the finite-rank-atom streaming PRM.

Purpose: correctness oracle for degenerate inputs and short deterministic tests,
not a replacement for the fast full-scale floating-point monitor.
Every Python float is interpreted as its EXACT stored binary rational.
The rank-tie convention is calibration-first: R=1+#(cal_score <= online_score).

No random seeds, approximate roots or future observations are used by stream_atoms.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as F
from bisect import bisect_right
from math import isfinite
from typing import Sequence


def Q(v):
    if isinstance(v, F): return v
    if isinstance(v, float):
        if not isfinite(v): raise ValueError('Nonfinite input')
        return F.from_float(v)
    return F(v)


def _vec(v): return tuple(Q(x) for x in v)


@dataclass(frozen=True)
class State:
    wealth: F = F(1)
    lam: F = F(0)
    curvature: F = F(1)
    seen: F = F(0)


def _clip(v, lower, upper): return max(lower, min(upper, v))


def advance(s: State, rank: int, n: int, tzero: int) -> State:
    """Order PRM with original Eq.(1) predictive centering / Eq.(2) ONS.

    tzero is the zero-based online observation index. Uses rational constants,
    including 3/4 and 9/2 for ONS, and keeps all arithmetic exact.
    """
    assert n >= 1 and 1 <= rank <= n+1
    half=F(1,2)
    h=F(rank-1,n)-half
    center=s.seen/F(n+tzero+1)
    B=max(abs(-half-center),abs(half-center))
    b=min(F(1),F(3,4)/B) if B else F(1)
    lam=_clip(s.lam,-b,b)
    z=h-center
    fac=F(1)+lam*z
    assert fac >= F(1,4)
    wealth=s.wealth*fac
    g=z/fac
    curvature=s.curvature+g*g
    seen=s.seen+h
    center2=seen/F(n+tzero+2)
    B2=max(abs(-half-center2),abs(half-center2))
    b2=min(F(1),F(3,4)/B2) if B2 else F(1)
    lam2=_clip(lam+F(9,2)*g/curvature,-b2,b2)
    return State(wealth,lam2,curvature,seen)


def rank_at(caly,calx,y,x,a):
    score=abs(y-a*x)
    return 1+sum(abs(c-a*v)<=score for c,v in zip(caly,calx))


def equality_roots(caly,calx,y,x,lo,hi):
    """All algebraic score equality roots in CLOSED parameter domain.

    If denominator and numerator are both zero, that linear factor is
    identically zero; the full tie is handled by rank_at at every candidate.
    If denominator is zero and numerator nonzero: that factor has no root.
    """
    result=set()
    for c,v in zip(caly,calx):
        for numerator,denominator in ((y-c,x-v),(y+c,x+v)):
            if denominator:
                root=numerator/denominator
                if lo<=root<=hi:result.add(root)
    return sorted(result)


def _atoms(breaks):
    # Ordered atoms: boundary point, open interval, ..., boundary point.
    for i,a in enumerate(breaks):
        yield ('point',a)
        if i+1<len(breaks): yield ('open',(a,breaks[i+1]))


def merge_sorted_unique(a,b):
    """O(len(a)+len(b)) order merge; both input sequences are sorted."""
    i=j=0;out=[]
    while i<len(a) or j<len(b):
        if j>=len(b) or (i<len(a) and a[i]<b[j]):
            value=a[i];i+=1
        elif i>=len(a) or b[j]<a[i]:
            value=b[j];j+=1
        else:
            value=a[i];i+=1;j+=1
        if not out or out[-1]!=value:out.append(value)
    return out


def stream_atoms(caly,calx,ony,onx,domain=(F(-999,1000),F(999,1000)),
                 threshold=20,max_atoms=1200,stop_at_alarm=False,efficient_sweep=True):
    """True ONLINE exact rank-atom algorithm, with root point states.

    Returns exact Fraction profile path and first alarm. Inadmissible large
    atom sets raise OverflowError, not a guessed non-alarm.
    """
    cy,cx,oy,ox=map(_vec,(caly,calx,ony,onx))
    if not cy or len(cy)!=len(cx) or len(oy)!=len(ox):raise ValueError('Bad shape')
    lo,hi=map(Q,domain)
    if lo>=hi:raise ValueError('Non-increasing domain')
    thresh=Q(threshold)
    breaks=[lo,hi]
    pt={lo:State(),hi:State()}
    opens=[State()]
    path=[F(1)]; argmin=[lo]; count=[3]; new_roots=[]; first_alarm=0
    n=len(cy)
    for t,(y,x) in enumerate(zip(oy,ox)):
        if efficient_sweep:
            from atom_sweep import sweep_current, ranks_for_partition
            roots,open_rank,point_rank=sweep_current(cy,cx,y,x,lo,hi)
        else:
            roots=equality_roots(cy,cx,y,x,lo,hi)
        nb=merge_sorted_unique(breaks,roots)
        if efficient_sweep:
            point_ranks,open_ranks=ranks_for_partition(roots,open_rank,point_rank,nb)
        if 2*len(nb)-1>max_atoms:
            raise OverflowError('Exact atom count exceeds max_atoms; return UNKNOWN')
        next_points={};next_opens=[]
        oldidx=0
        for k,a in enumerate(nb):
            while oldidx+1<len(breaks) and breaks[oldidx+1]<=a:
                oldidx+=1
            if a==breaks[oldidx]:parent_point=pt[a]
            else:parent_point=opens[oldidx]
            rank=point_ranks[k] if efficient_sweep else rank_at(cy,cx,y,x,a)
            next_points[a]=advance(parent_point,rank,n,t)
            if k+1<len(nb):
                # all previous roots are included in nb, so the next open
                # child belongs to the previous open parent right of a.
                rank_open=open_ranks[k] if efficient_sweep else rank_at(
                    cy,cx,y,x,(a+nb[k+1])/2)
                next_opens.append(advance(opens[oldidx],rank_open,n,t))
        breaks,pt,opens=nb,next_points,next_opens
        vals=[(s.wealth,a) for a,s in pt.items()]
        vals += [(s.wealth,(l+r)/2) for (l,r),s in zip(zip(nb[:-1],nb[1:]),opens)]
        val,arg=min(vals)
        path.append(val);argmin.append(arg)
        count.append(len(vals));new_roots.append(len(roots))
        if not first_alarm and val>=thresh:
            first_alarm=t+1
            if stop_at_alarm:break
    return {'path':path,'argmin':argmin,'counts':count,'roots_per_step':new_roots,
            'first_alarm':first_alarm,'breakpoints':breaks,'num_atoms':2*len(breaks)-1,
            'status':'CERTIFIED_EXACT_RATIONAL'}


def direct_rank_path_wealth(caly,calx,ony,onx,a):
    """Independent direct path evaluation at a *single* parameter atom."""
    cy,cx,oy,ox=map(_vec,(caly,calx,ony,onx))
    a=Q(a);s=State();ret=[s.wealth]
    for t,(y,x) in enumerate(zip(oy,ox)):
        s=advance(s,rank_at(cy,cx,y,x,a),len(cy),t)
        ret.append(s.wealth)
    return ret


def batch_atoms(caly,calx,ony,onx,domain=(F(-999,1000),F(999,1000)),threshold=20,
                max_atoms=1200):
    """INDEPENDENT batch exhaustive: re-derive all atoms and resimulate every
    history from scratch for every prefix. Does NOT inherit states.
    """
    cy,cx,oy,ox=map(_vec,(caly,calx,ony,onx));lo,hi=map(Q,domain)
    roots={lo,hi}; path=[F(1)];first=0;nb_at=[]
    for t,(y,x) in enumerate(zip(oy,ox)):
        roots.update(equality_roots(cy,cx,y,x,lo,hi))
        if 2*len(roots)-1>max_atoms:raise OverflowError('Batch atom cap')
        rep=[a if typ=='point' else sum(a,F())/2
             for typ,a in _atoms(sorted(roots))]
        arr=[(direct_rank_path_wealth(cy,cx,oy[:t+1],ox[:t+1],a)[-1],a)
             for a in rep]
        v,_=min(arr)
        path.append(v);nb_at.append(len(rep))
        if not first and v>=Q(threshold):first=t+1
    return {'path':path,'first_alarm':first,'num_atoms_by_time':nb_at,
            'status':'CERTIFIED_BATCH_RATIONAL'}


def demonstrate_root_only_path():
    """At a=0 rank=3, while BOTH neighboring open cells have rank=2."""
    cy=(F(1),F(1));cx=(F(1),F(-1));y=F(1);x=F(0)
    return [(str(a),rank_at(cy,cx,y,x,a)) for a in (F(-1,10),F(0),F(1,10))]


if __name__=='__main__':
    print('root-only counterexample, (a,rank):',demonstrate_root_only_path())
    caly=[0,0];calx=[1,-1];ony=[0,1];onx=[2,0]
    s=stream_atoms(caly,calx,ony,onx,max_atoms=1000)
    b=batch_atoms(caly,calx,ony,onx)
    assert s['path']==b['path']
    print('Exact rational streaming vs batch PASS', [float(x) for x in s['path']])
