"""Named, transparent baselines. These are independent implementations,
NOT official PRM/CCTM/WATCH code. Gaussian CUSUM uses empirical calibration,
NOT a distribution-free/anytime-valid guarantee for a plug-in residual stream.
"""
import numpy as np
from prm_core import oracle_prm,first_alarm


def ols_slope(train):
    x=np.asarray(train[:-1]);y=np.asarray(train[1:]);return float(np.dot(x,y)/np.dot(x,x))


def prm_residual(train,cal,on,slope,feature='order_abs'):
    return oracle_prm(np.abs(cal[1:]-slope*cal[:-1]),
                      np.abs(on[1:]-slope*on[:-1]),feature)


def oracle(train,cal,on,true_a,feature='order_abs'):
    return prm_residual(train,cal,on,true_a,feature)


def plugin(train,cal,on,feature='order_abs'):
    return prm_residual(train,cal,on,ols_slope(train),feature)


def gaussian_scale_cusum(train,on,scale=1.5):
    """Known shape Gaussian N(0,sigma0^2), estimated slope and sigma0.
    This is a Page CUSUM of a Gaussian scale log-likelihood ratio; empirical
    null thresholds must be separately calibrated and their finite-sample
    guarantees depend on that calibration design.
    """
    a=ols_slope(train)
    health=train[1:]-a*train[:-1]
    sigma0=np.sqrt(np.mean(health**2))
    r=(on[1:]-a*on[:-1])/max(sigma0,1e-9)
    llr=-np.log(scale)+.5*(1-1/(scale*scale))*r*r
    S=np.zeros(len(r)+1)
    for i,z in enumerate(llr):S[i+1]=max(0.,S[i]+z)
    return S


def empirical_cusum_threshold(simulate_healthy,N=3000,alpha=.04):
    """Independent calibration from null trajectories. Finite sample index
    chosen to bound marginal exceedance on one future fresh trajectory by
    approximately ceil((1-alpha)(N+1))/(N+1) under exchangeability of maxima.
    Returns inf if insufficient calibration size for nonrandomized threshold.
    """
    maxima=np.array([np.max(simulate_healthy(i)) for i in range(N)])
    rank=int(np.ceil((1-alpha)*(N+1)))
    if rank>N:return np.inf
    return float(np.sort(maxima)[rank-1])


def selftest():
    from simulate_core import split_data
    rng=np.random.default_rng(53)
    tr,cal,on=split_data(rng,.9,M=224,N=128,T=150)
    assert np.isfinite(ols_slope(tr))
    assert oracle(tr,cal,on,.9).shape==(151,)
    assert gaussian_scale_cusum(tr,on).shape==(151,)
    print('Baseline selftest PASS')
if __name__=='__main__':selftest()


# -------------------------------------------------------------------------
# Historical K=1 / K=8 shared-coefficient tail monitor, ONE regime only.
# Unlike PRM, this method relies on symmetric innovations for ratio-sign CI.
# This is an independent portable adaptation of the archived stage2 method.
# -------------------------------------------------------------------------

def _shared_max_tail_count(lag,y,c,lo,hi):
    import numpy as np
    inside=0; events=[]
    for x,yy in zip(lag,y):
        if abs(x)<1e-14:
            if abs(yy)<=c:inside+=1
            continue
        left=(yy-c)/x; right=(yy+c)/x
        aa=min(left,right);bb=max(left,right)
        if aa<=lo<=bb:inside+=1
        if lo<aa<hi:events.append((aa,1))
        if lo<=bb<hi:events.append((bb,-1))
    min_inside=inside
    events.sort()
    j=0
    while j<len(events):
        pt=events[j][0];ds=0
        while j<len(events) and events[j][0]==pt:
            ds+=events[j][1];j+=1
        inside+=ds
        min_inside=min(min_inside,inside)
    return len(y)-min_inside


def old_k_calibrate(train,cal,thresholds=(1.5,2.,2.5,3.),
                    delta_a=.004,delta_tail=.006):
    """Only proved for conditionally sign-symmetric iid innovations.
    Historical K monitor uses FULL common-a max tail count, not naive grid.
    """
    from scipy.stats import binom,beta
    M=len(train)-1;N=len(cal)-1
    ratio=np.sort(train[1:]/train[:-1])
    kl=int(binom.ppf(delta_a/2,M,.5));ku=int(binom.isf(delta_a/2,M,.5))
    lo=float(ratio[M-ku-1]);hi=float(ratio[M-kl]);
    p=[]
    for c in thresholds:
        J=_shared_max_tail_count(cal[:-1],cal[1:],c,lo,hi)
        v=1. if J==N else float(beta.ppf(1-delta_tail/len(thresholds),J+1,N-J))
        p.append(v)
    return lo,hi,np.array(p)


def old_k_tail_monitor(on,calibrated,K=8,thresholds=(1.5,2.,2.5,3.),
                       alpha_monitor=.04):
    """One-mode K subintervals. Return e-process envelope path.
    The minimum over K candidates is NOT in general an e-process; size is
    bounded by domination by the true candidate's nonnegative supermartingale.
    """
    lo,hi,p=calibrated
    c=lo+(hi-lo)*(np.arange(K)+.5)/K
    rad=(hi-lo)/(2*K)
    H=len(on)-1
    run=np.zeros((len(thresholds),K))
    path=np.ones(H+1)
    thresh=np.asarray(thresholds)[:,None]
    p=np.maximum(p[:,None],1e-15)
    for t in range(1,H+1):
        r=abs(on[t]-c*on[t-1]);margin=rad*abs(on[t-1])
        event=r[None,:]>thresh+margin
        factor=.75+.25*event/p
        run=factor*(1+run)
        e=(H-t)/H +np.mean(run,axis=0)/H
        path[t]=min(e)
    return path
