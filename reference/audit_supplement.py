"""Check SI recursions against frozen reference code on the original 64 inputs.

This replays algebraic checks, not new detection experiments or new seeds.
"""
from pathlib import Path
from fractions import Fraction as F
from typing import NamedTuple
from decimal import Decimal
import csv,json,time,hashlib
import numpy as np
import mpmath as mp
from generic_profile import Backend,stream,batch,prm_backend,cctm_backend

ROOT=Path(__file__).resolve().parent
mp.mp.dps=80

class State(NamedTuple):
    wealth:object
    bet:object
    curvature:object
    total:object=0

def clip(x,b):return min(b,max(-b,x))
def prm_si(s,r,n,tzero):
    # Supplement S2: one-based t = tzero+1.
    t=tzero+1;h=F(r-1,n)-F(1,2);mu=s.total/(n+t)
    b=min(F(1),F(3,4)/max(abs(-F(1,2)-mu),abs(F(1,2)-mu)))
    bet=clip(s.bet,b);z=h-mu;factor=1+bet*z
    wealth=s.wealth*factor;g=z/factor;curve=s.curvature+g*g;total=s.total+h
    mu2=total/(n+t+1);b2=min(F(1),F(3,4)/max(abs(-F(1,2)-mu2),abs(F(1,2)-mu2)))
    return State(wealth,clip(bet+F(9,2)*g/curve,b2),curve,total)

def cctm_si(s,r,n,tzero):
    half=mp.mpf('0.5');eps=mp.sqrt(mp.log(200)/(2*n));rho=mp.mpf('1e-6')
    B=half+(1+rho)*eps;v=mp.mpf(r-1)/n-half
    bet=s.bet if abs(s.bet)>=mp.mpf('0.1') else mp.mpf(0)
    factor=1+(bet*v-eps*mp.sqrt(bet*bet+rho*rho))/B
    g=(v-eps*s.bet/mp.sqrt(s.bet*s.bet+rho*rho))/(B+s.bet*v-eps*mp.sqrt(s.bet*s.bet+rho*rho))
    curve=s.curvature+g*g;new=clip(s.bet+2*g/((2-mp.log(3))*curve),half)
    return State(s.wealth*factor,new,curve)

def compare(a,b,exact):
    if exact:assert a==b;return 0.
    error=abs(a-b);assert error<mp.mpf('1e-65');return float(error)

def main():
    start=time.perf_counter();old=json.loads((ROOT/'legacy_cases.json').read_text())
    cases=[(name,[[F(n,d) for n,d in seq] for seq in raw]) for name,raw in old.items()]
    rng=np.random.default_rng(260908)
    for j in range(50):
        n=int(rng.integers(2,5));t=int(rng.integers(2,5))
        v=[[F(int(z),6) for z in rng.integers(-5,6,size=k)] for k in (n,n,t,t)]
        cases.append((f'random_{j:02d}',v))
    archived=json.loads((ROOT/'historical_random_48_input.json').read_text())
    assert next(v for name,v in cases if name=='random_48')==[[F(z) for z in seq] for seq in archived['vectors']]
    assert len(cases)==64
    si=[Backend('SI_PRM',lambda:State(F(1),F(0),F(1),F(0)),prm_si,lambda s:s.wealth),
        Backend('SI_CCTM',lambda:State(mp.mpf(1),mp.mpf(0),mp.mpf(1)),cctm_si,lambda s:s.wealth)]
    rows=[];state_checks=0
    for name,v in cases:
        for idx,ref in enumerate([prm_backend(),cctm_backend()]):
            exact=idx==0;a=stream(*v,backend=ref);b=batch(*v,backend=ref);c=stream(*v,backend=si[idx])
            assert a['atoms']==c['atoms']
            # Batch may retain redundant roots of identically tied comparisons.
            # Its extra refinement must preserve the same wealth minimum.
            assert all(x<=y for x,y in zip(a['atoms'],b['atoms']))
            assert all(k<=4*len(v[0])*t+3 for t,k in enumerate(a['atoms']))
            err=max(compare(x,y,exact) for x,y in zip(a['profile'],b['profile']))
            err_si=max(compare(x,y,exact) for x,y in zip(a['profile'],c['profile']))
            for ar,cs in list(zip(a['points'].values(),c['points'].values()))+list(zip(a['opens'],c['opens'])):
                rr=(ar.wealth,ar.lam,ar.curvature,ar.seen) if exact else (ar.wealth,ar.eta,ar.accum)
                for x,y in zip(rr,cs):compare(x,y,exact);state_checks+=1
            rows.append(dict(case=name,backend=ref.name,atoms=a['atoms'][-1],stream_batch_error=err,si_recursion_error=err_si,status='PASS'))
    with (ROOT/'supplement_equation_checks.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (ROOT/'source_witness.csv').open() as f: witnesses=list(csv.DictReader(f))
    for r in witnesses:
        assert Decimal(r['witness_max'])<Decimal(r['threshold'])
        assert abs(Decimal(r['threshold'])-Decimal(r['witness_max'])-Decimal(r['margin']))<Decimal('1e-25')
        assert int(r['dps_low'])==100 and int(r['dps_high'])==160
    result=dict(status='PASS_FIXED_INPUT_SI_REPLAY',inputs=64,backend_input_checks=len(rows),state_field_comparisons=state_checks,
        witness_summary_rows_checked=len(witnesses),cctm_decimal_digits=80,seconds=round(time.perf_counter()-start,3),
        scope='SI equations versus archived reference implementations and stream/batch replay on the original 64 inputs. Witness check uses archived maxima, not fresh raw-data or interval-arithmetic certification.')
    (ROOT/'supplement_independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
