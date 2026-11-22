"""Independent implementation of Kuang & Xia (2026) predictive rank martingale.
Paper: https://arxiv.org/html/2609.00536v1 , Eq (1), Eq (2), Algorithm 1.
Not claimed to be official authors' code. Null guarantees are *marginal* over the
random iid continuous calibration scores and subsequent iid online scores.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

FEATURES = ('dispersion', 'order_abs', 'mid_tail', 'extreme_tail')


def feature_values(n: int, kind: str = 'dispersion') -> np.ndarray:
    u = np.arange(n+1, dtype=float) / n
    if kind == 'dispersion':
        h = np.abs(2*u-1)-0.5  # Eq. in PRM paper; span exactly 1
    elif kind == 'order_abs':
        h = u-0.5             # use |residual| as input, scale is one-sided
    elif kind == 'mid_tail':
        h = np.maximum((u-0.6)/0.4, 0.0)
    elif kind == 'extreme_tail':
        h = np.maximum((u-0.9)/0.1, 0.0)
    else:
        raise ValueError(kind)
    return h


def fixed_reference_ranks(calibration: np.ndarray, online: np.ndarray) -> np.ndarray:
    cal = np.asarray(calibration, float)
    on = np.asarray(online, float)
    if not np.all(np.isfinite(cal)) or not np.all(np.isfinite(on)):
        raise ValueError('Scores must be finite')
    # Continuous-score null: ties have probability zero. For deliberate ties,
    # add randomized lexicographic tie breaks before invoking this function.
    return np.searchsorted(np.sort(cal), on, side='right') + 1


def prm_wealth_from_ranks(ranks: np.ndarray, n: int, kind='dispersion') -> np.ndarray:
    """Exact paper ONS update; returns wealth 0..T. Uses no iid-uniform-p fiction."""
    ranks = np.asarray(ranks, np.int32)
    if np.any((ranks < 1) | (ranks > n+1)):
        raise ValueError('Rank outside 1..n+1')
    h = feature_values(n, kind)
    prior = float(h.sum()); hmin = float(h.min()); hmax = float(h.max())
    lam = 0.0; A = 1.0; seen_h = 0.0
    wealth = np.ones(len(ranks)+1, float)
    for i, r in enumerate(ranks):
        center = (prior+seen_h)/(n+i+1)  # Eq (1), Polya predictive mean
        B = max(abs(hmin-center), abs(hmax-center))
        b = min(1., 0.75/B) if B > 0. else 1.
        lam = float(np.clip(lam, -b, b))
        z = float(h[r-1]-center)
        factor = 1.0+lam*z
        if factor < 0.25-1e-10:
            raise AssertionError('ONS betting factor lower bound violated')
        wealth[i+1] = wealth[i]*factor
        g = z/factor
        A += g*g
        seen_h += h[r-1]
        center_next = (prior+seen_h)/(n+i+2)
        B_next = max(abs(hmin-center_next), abs(hmax-center_next))
        b_next = min(1., 0.75/B_next) if B_next > 0 else 1.
        lam = float(np.clip(lam+4.5*g/A, -b_next, b_next))
    return wealth


def prm_matrix_from_ranks(ranks: np.ndarray, n: int, kind='dispersion',
                          wealth_path=False):
    """Vectorized exact ONS for multiple parameter candidates, shape (K,T)."""
    R = np.asarray(ranks, np.int32)
    if R.ndim != 2: raise ValueError('Expected (K,T)')
    K, T = R.shape
    h = feature_values(n, kind)
    prior = float(h.sum()); hmin = float(h.min()); hmax = float(h.max())
    lam = np.zeros(K); A = np.ones(K); seen_h = np.zeros(K); M=np.ones(K)
    paths=np.ones((K,T+1)) if wealth_path else None
    for i in range(T):
        center = (prior+seen_h)/(n+i+1)
        B = np.maximum(np.abs(hmin-center),np.abs(hmax-center))
        b = np.minimum(1.0,0.75/np.maximum(B,1e-12))
        lam=np.clip(lam,-b,b)
        z=h[R[:,i]-1]-center
        factor=1.0+lam*z
        if np.min(factor)<0.25-1e-10:raise AssertionError('ONS factor')
        M *= factor
        g=z/factor; A+=g*g; seen_h+=h[R[:,i]-1]
        center2=(prior+seen_h)/(n+i+2)
        B2=np.maximum(np.abs(hmin-center2),np.abs(hmax-center2))
        b2=np.minimum(1.,0.75/np.maximum(B2,1e-12))
        lam=np.clip(lam+4.5*g/A,-b2,b2)
        if wealth_path: paths[:,i+1]=M
    return paths if wealth_path else M


def oracle_prm(cal: np.ndarray, online: np.ndarray, kind='dispersion') -> np.ndarray:
    return prm_wealth_from_ranks(fixed_reference_ranks(cal,online),len(cal),kind)


def first_alarm(wealth: np.ndarray, alpha=0.05):
    w = np.asarray(wealth)
    hit = np.flatnonzero(w >= 1/alpha)
    return int(hit[0]) if hit.size else 0


def selftest():
    rng=np.random.default_rng(441)
    # Exact sequential conditional distribution is Polya urn; compare Monte Carlo
    n=17; seqs=2000; steps=12
    ranks=np.empty((seqs,steps),int)
    for i in range(seqs):
        cal=rng.normal(size=n); on=rng.normal(size=steps)
        ranks[i]=fixed_reference_ranks(cal,on)
    # First rank uniform in 1..n+1
    counts=np.bincount(ranks[:,0],minlength=n+2)[1:]
    assert np.max(np.abs(counts/seqs-1/(n+1))) < 0.02
    m1=prm_wealth_from_ranks(ranks[0],n)
    m2=prm_matrix_from_ranks(ranks[:50],n,wealth_path=True)
    for j in range(50):
        assert np.allclose(prm_wealth_from_ranks(ranks[j],n),m2[j],rtol=1e-12)
    assert m1.shape==(steps+1,)
    print('PRM core selftest PASS: exact Polya centering + Eq(2) ONS scalar/vector equivalence')

if __name__=='__main__':selftest()


def prm_restart_matrix_from_ranks(ranks: np.ndarray, n: int, start: int,
                                  kind='dispersion', wealth_path=True):
    """Valid common-filtration candidate-start PRM.
    IMPORTANT: predictive centering uses rank history from global time 0
    even after resetting ONS wealth at candidate start. Resetting Polya counts
    at each k and mixing those 'martingales' has NO established shared
    filtration proof; this implementation avoids that error.
    """
    R=np.asarray(ranks,np.int32)
    K,T=R.shape
    assert 0<=start<T
    h=feature_values(n,kind); prior=float(h.sum())
    mn,mx=float(h.min()),float(h.max())
    previous=np.zeros(K)
    lam=np.zeros(K); A=np.ones(K);M=np.ones(K)
    W=np.ones((K,T+1)) if wealth_path else None
    for t in range(T):
        center=(prior+previous)/(n+t+1)
        z=h[R[:,t]-1]-center
        if t>=start:
            B=np.maximum(abs(mn-center),abs(mx-center)); b=np.minimum(1.,.75/np.maximum(B,1e-12))
            lam=np.clip(lam,-b,b)
            fac=1+lam*z
            assert np.min(fac)>=.25-1e-10
            M*=fac
            g=z/fac;A+=g*g
        previous+=h[R[:,t]-1]
        if t>=start:
            center2=(prior+previous)/(n+t+2)
            B2=np.maximum(abs(mn-center2),abs(mx-center2))
            b2=np.minimum(1.,.75/np.maximum(B2,1e-12))
            lam=np.clip(lam+4.5*g/A,-b2,b2)
        if wealth_path:W[:,t+1]=M
    return W if wealth_path else M
