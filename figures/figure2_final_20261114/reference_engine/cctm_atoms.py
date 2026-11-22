"""Independent CCTM formula-level, high-precision exact-rank-atom reference.

All root positions are Fraction values from the exact binary floats; root
singletons, domain endpoints, equal roots, identical score functions included.
CCTM wealth uses mpmath arbitrary precision because log/sqrt are not rational.
This is NOT a certified interval-arithmetic baseline and NOT authors' source.

Formula matches public cond_ctm.py / utils.py / optimization.py for smooth>0,
D=.5, C=.1, ci_delta=.01.  Rank at tie: calibration <= online.
"""
from __future__ import annotations
from fractions import Fraction as F
from dataclasses import dataclass
from bisect import bisect_right
from math import isfinite
import mpmath as mp

mp.mp.dps=70

def Q(x):
    if isinstance(x,F):return x
    if isinstance(x,float):
        if not isfinite(x):raise ValueError('nonfinite')
        return F.from_float(x)
    return F(x)

def as_mp(x):
    if isinstance(x,F):return mp.mpf(x.numerator)/x.denominator
    return mp.mpf(str(x))

@dataclass(frozen=True)
class CState:
    wealth:object
    eta:object
    accum:object

def initial():return CState(mp.mpf(1),mp.mpf(0),mp.mpf(1))

def advance(s,rank,N,delta=F(1,100),D=F(1,2),C=F(1,10),smooth=F(1,10**6)):
    eps=mp.sqrt(mp.log(2/as_mp(delta))/(2*N))
    sm=as_mp(smooth); lim=as_mp(D); clip=as_mp(C)
    u=mp.mpf(rank-1)/N
    norm=mp.mpf('.5')+(1+sm)*eps
    bet_eta=s.eta if abs(s.eta)>=clip else mp.mpf(0)
    if sm == 0:
        # Public optimization.py has a separate nonsmooth subgradient branch.
        sign_bet=mp.sign(bet_eta)
        factor=1+bet_eta*(u-mp.mpf('.5')-sign_bet*eps)/norm
        if factor<=0:raise ArithmeticError('negative CCTM betting factor')
        sign_eta=mp.sign(s.eta)
        v=(u-mp.mpf('.5')-sign_eta*eps)/norm
        grad=v/(1+s.eta*v)
    else:
        factor=1+(bet_eta*(u-mp.mpf('.5'))-mp.sqrt(bet_eta**2+sm**2)*eps)/norm
        if factor<=0:raise ArithmeticError('negative CCTM betting factor')
        root=mp.sqrt(s.eta**2+sm**2)
        denom=1+(s.eta*(u-mp.mpf('.5'))-root*eps)/norm
        grad=(u-mp.mpf('.5')-s.eta*eps/root)/(norm*denom)
    wealth=s.wealth*factor
    acc=s.accum+grad*grad
    etas=s.eta+2/(2-mp.log(3))*grad/acc
    return CState(wealth,max(-lim,min(lim,etas)),acc)

def rank(caly,calx,y,x,a):
    a=Q(a)
    online=abs(y-a*x)
    return 1+sum(abs(c-a*v)<=online for c,v in zip(caly,calx))

def roots_for(caly,calx,y,x,lo,hi):
    roots=set()
    for c,v in zip(caly,calx):
        for numerator,denominator in ((y-c,x-v),(y+c,x+v)):
            if denominator:
                a=numerator/denominator
                if lo<a<hi:roots.add(a)
    return sorted(roots)

def reps(breaks):
    for i,a in enumerate(breaks):
        yield ('point',a)
        if i+1<len(breaks):yield ('open',(a,breaks[i+1]))

def _input(caly,calx,ony,onx,domain):
    cy,cx,oy,ox=[tuple(map(Q,z)) for z in (caly,calx,ony,onx)]
    lo,hi=map(Q,domain)
    if not cy or len(cy)!=len(cx) or len(oy)!=len(ox) or lo>=hi:raise ValueError('shape/domain')
    return cy,cx,oy,ox,lo,hi

def stream(caly,calx,ony,onx,domain=(F(-999,1000),F(999,1000)),threshold=20,max_atoms=20000,
           delta=F(1,100),D=F(1,2),C=F(1,10),smooth=F(1,10**6)):
    cy,cx,oy,ox,lo,hi=_input(caly,calx,ony,onx,domain)
    breaks=[lo,hi];points={lo:initial(),hi:initial()};opens={(lo,hi):initial()}
    path=[mp.mpf(1)];counts=[3];alarm=0
    for t,(y,x) in enumerate(zip(oy,ox),1):
        old_breaks=breaks
        newroots=roots_for(cy,cx,y,x,lo,hi)
        breaks=sorted(set(breaks).union(newroots))
        if 2*len(breaks)-1>max_atoms:raise OverflowError('too many atoms')
        npnt={};nopen={}
        for i,a in enumerate(breaks):
            if a in points:parent=points[a]
            else:
                j=bisect_right(old_breaks,a)-1
                parent=opens[(old_breaks[j],old_breaks[j+1])]
            npnt[a]=advance(parent,rank(cy,cx,y,x,a),len(cy),delta=delta,D=D,C=C,smooth=smooth)
            if i+1<len(breaks):
                b=breaks[i+1];mid=(a+b)/2
                j=bisect_right(old_breaks,mid)-1
                par=opens[(old_breaks[j],old_breaks[j+1])]
                nopen[(a,b)]=advance(par,rank(cy,cx,y,x,mid),len(cy),delta=delta,D=D,C=C,smooth=smooth)
        points,opens=npnt,nopen
        mn=min([v.wealth for v in points.values()]+[v.wealth for v in opens.values()])
        path.append(mn);counts.append(len(points)+len(opens))
        if alarm==0 and mn>=as_mp(threshold):alarm=t
    return {'path':path,'alarm':alarm,'counts':counts,'points':points,'opens':opens}

def direct(caly,calx,ony,onx,a):
    cy,cx,oy,ox,_,_=_input(caly,calx,ony,onx,(-1,1))
    a=Q(a);s=initial();path=[s.wealth]
    for y,x in zip(oy,ox):
        s=advance(s,rank(cy,cx,y,x,a),len(cy));path.append(s.wealth)
    return path

def batch(caly,calx,ony,onx,domain=(F(-999,1000),F(999,1000)),threshold=20,max_atoms=20000):
    cy,cx,oy,ox,lo,hi=_input(caly,calx,ony,onx,domain)
    breaks={lo,hi};path=[mp.mpf(1)];counts=[3];alarm=0
    for t,(y,x) in enumerate(zip(oy,ox),1):
        breaks.update(roots_for(cy,cx,y,x,lo,hi));bs=sorted(breaks)
        if 2*len(bs)-1>max_atoms:raise OverflowError('too many atoms')
        vals=[]
        for typ,obj in reps(bs):
            a=obj if typ=='point' else (obj[0]+obj[1])/2
            vals.append(direct(cy,cx,oy[:t],ox[:t],a)[-1])
        mn=min(vals);path.append(mn);counts.append(len(vals))
        if alarm==0 and mn>=as_mp(threshold):alarm=t
    return {'path':path,'alarm':alarm,'counts':counts}
