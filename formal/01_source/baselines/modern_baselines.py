"""Independent minimal formula-level reconstructions for audit ONLY.

CCTM: follows Shaer et al. 2026 publicly displayed cond_ctm.py,
utils.py and optimization.py, but this is NOT an execution of the official
repository and does not implement all published options.

Conformal CUSUM: growing-reference insertion-rank p-values and a fixed
power betting function, CUSUM maximum draw-up. Its threshold c certifies
an ARL-type statement, NOT probability of at least one alarm <= 1/c.
"""
from __future__ import annotations
import math
import numpy as np


def cctm_wealth(cal, online, delta=.01, D=.5, C=.1, smooth=1e-6):
    cal = np.sort(np.asarray(cal, dtype=float))
    online = np.asarray(online, dtype=float)
    if len(cal) == 0 or not (0 < delta < 1):
        raise ValueError('invalid reference or delta')
    eps = math.sqrt(math.log(2./delta)/(2*len(cal)))
    eta=0.; accum=1.; w=1.; curve=[1.]
    norm=0.5+(1+smooth)*eps
    for y in online:
        u=float(np.searchsorted(cal,y,side='right'))/len(cal)
        if abs(eta)<C: e_bet=0.
        else: e_bet=eta
        def factor(x):
            return 1 + (x*(u-0.5)-math.hypot(x,smooth)*eps)/norm
        bet=factor(e_bet)
        if bet < -1e-12:
            raise ArithmeticError('betting factor negative')
        w*=max(bet,0.)
        curve.append(w)
        # Official ONS gradient from public optimization.py (independent expression)
        numer=(u-.5-eta*eps/max(math.hypot(eta,smooth),1e-300))/norm
        g=numer/factor(eta)
        accum+=g*g
        eta=float(np.clip(eta+(2/(2-math.log(3)))*g/accum, -D,D))
    return np.asarray(curve)


def growing_conformal_p(cal, online, seed=991):
    """Randomized upper-rank insertion p-values for fixed iid scores.
    Existing calibration observations are treated as earlier observations.
    Monte Carlo jitter is only for the insertion rank; no data leak.
    """
    past=list(map(float,cal)); rng=np.random.default_rng(seed)
    out=[]
    for score in online:
        # insertion ranks are uniform in {1,...,n+1} under iid continuity
        r=1+sum(x>score for x in past)
        p=(r-1+rng.uniform())/(len(past)+1)
        out.append(p);past.append(float(score))
    return np.asarray(out)


def conformal_cusum_arl(cal, online, c=20., power=.5,seed=991):
    """Operational minimal conformal CUSUM, NOT PFA threshold.
    f(p) = power*p**(power-1), integral_0^1 f=1.
    CUSUM=max(0,prev+log(f(p))), where threshold log(c) has
    ARL-related meaning (paper Eq. 1/2), not a 5% all-window guarantee.
    """
    ps=growing_conformal_p(cal,online,seed=seed)
    path=np.zeros(len(ps)+1);logc=math.log(c)
    for t,p in enumerate(ps,1):
        path[t]=max(0.,path[t-1]+math.log(power)+(power-1)*math.log(p))
    alarm=next((i for i,x in enumerate(path) if x>=logc),0)
    return path, alarm


def first_crossing(w, threshold):
    j=np.flatnonzero(np.asarray(w)>=threshold)
    return int(j[0]) if len(j) else 0


if __name__=='__main__':
    from numpy.random import default_rng
    r=default_rng(33)
    a=cctm_wealth(abs(r.standard_normal(736)),abs(r.standard_normal(30)))
    b,p=conformal_cusum_arl(abs(r.standard_normal(736)),abs(r.standard_normal(30)))
    assert np.all(np.isfinite(a)) and np.all(a>=0)
    assert np.all(np.isfinite(b))
    print('modern_baselines smoke PASS',a[-1],p)
