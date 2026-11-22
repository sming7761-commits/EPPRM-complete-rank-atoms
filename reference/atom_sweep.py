"""Grouped rank-event sweep for |y-a*x| against N calibration residuals.

Handles identical score functions, denominator zero, repeated/tangent roots,
open interval ranks, point-root ranks and endpoints without N comparisons per atom.
With unit-cost exact rational operations: O(N log N) to build all per-step events,
then O(K_t) for rank assignment to sorted historical atoms.
"""
from __future__ import annotations
from fractions import Fraction as F
from bisect import bisect_left, bisect_right
from atom_exact import Q,rank_at


def _root_event(c,v,y,x,lo,hi):
    # f(a)= (y-c - a(x-v)) (y+c - a(x+v)).
    p,y1=y-c,y+c
    d1,d2=x-v,x+v
    if (p==0 and d1==0) or (y1==0 and d2==0):
        # Score functions identically equal, counted as a tie everywhere.
        return []
    roots=set()
    if d1:
        a=p/d1
        if lo<=a<=hi:roots.add(a)
    if d2:
        a=y1/d2
        if lo<=a<=hi:roots.add(a)
    arr=[]
    for a in roots:
        f1=p-a*d1;f2=y1-a*d2
        derivative=-d1*f2-d2*f1
        if derivative>0:delta,bump=1,1
        elif derivative<0:delta,bump=-1,0
        else:
            # True double root/tangency: sign of f off root = d1*d2.
            # If f<0 on both sides, the root contributes a tie-only bump.
            if f1!=0 or f2!=0:raise AssertionError('Non-root')
            if d1*d2<0:delta,bump=0,1
            elif d1*d2>0:delta,bump=0,0
            else: raise AssertionError('Unexpected identically-tied factor')
        arr.append((a,delta,bump))
    return arr


def sweep_current(caly,calx,y,x,lo,hi):
    """Returns (interior_root_positions, open_ranks, point_rank_map, boundaries).
    Rank convention = 1 + number of calibration scores <= online score.
    All quantities are exact Fractions; repeated roots are grouped atomically.
    """
    cy=tuple(map(Q,caly));cx=tuple(map(Q,calx));y=Q(y);x=Q(x)
    lo,hi=Q(lo),Q(hi)
    groups={}
    for c,v in zip(cy,cx):
        for a,delta,bump in _root_event(c,v,y,x,lo,hi):
            if a not in groups:groups[a]=[0,0]
            groups[a][0]+=delta;groups[a][1]+=bump
    roots=sorted(a for a in groups if lo<a<hi)
    first=(lo+roots[0])/2 if roots else (lo+hi)/2
    current=rank_at(cy,cx,y,x,first)
    open_ranks=[current]
    point_ranks={lo:rank_at(cy,cx,y,x,lo),hi:rank_at(cy,cx,y,x,hi)}
    for root in roots:
        delta,bump=groups[root]
        point_ranks[root]=current+bump
        current+=delta
        open_ranks.append(current)
    if any(not (1<=r<=len(cy)+1) for r in open_ranks+list(point_ranks.values())):
        raise AssertionError('Invalid grouped rank')
    return roots,open_ranks,point_ranks


def ranks_for_partition(current_roots,open_ranks,point_ranks,breaks):
    """Linear order merge of rank events and a sorted global partition.
    Returns point ranks and open interval ranks in break order.
    """
    point=[];op=[];j=0;nr=len(current_roots)
    lo=breaks[0];hi=breaks[-1]
    for k,a in enumerate(breaks):
        while j<nr and current_roots[j]<a:j+=1
        if a in point_ranks:point.append(point_ranks[a])
        else: point.append(open_ranks[j])
        if k+1<len(breaks):
            nxt=breaks[k+1]
            # Midpoint is strictly between a and nxt, no current roots there
            # because all current roots are included in breaks.
            j2=j
            if j2<nr and current_roots[j2]==a:j2+=1
            op.append(open_ranks[j2])
    return point,op


def selftest():
    from test_exactness import CASES
    for name,(cy,cx,oy,ox) in CASES.items():
        lo,hi=F(-999,1000),F(999,1000)
        previous={lo,hi}
        for y,x in zip(oy,ox):
            roots,openr,ptr=sweep_current(cy,cx,y,x,lo,hi)
            previous.update(roots); br=sorted(previous)
            points,op=ranks_for_partition(roots,openr,ptr,br)
            for a,pr in zip(br,points):
                assert pr==rank_at(tuple(map(Q,cy)),tuple(map(Q,cx)),Q(y),Q(x),a),(name,'point',a,pr)
            for (a,b),orank in zip(zip(br[:-1],br[1:]),op):
                mid=(a+b)/2
                assert orank==rank_at(tuple(map(Q,cy)),tuple(map(Q,cx)),Q(y),Q(x),mid),(name,'open',mid,orank)
    print(f'GROUPED ROOT SWEEP PASS: {len(CASES)} adversarial cases; all point/open ranks exact')

if __name__=='__main__':selftest()
