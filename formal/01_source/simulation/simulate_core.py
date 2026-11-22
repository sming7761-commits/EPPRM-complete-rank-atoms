"""Independent-trajectory AR(1) simulation. One 'healthy transition' != independent trajectory.
Each call makes a fresh path; eps_t across distinct paths and between segments
are independently sampled. Stationarity is approximated by fixed burn-in.
"""
import numpy as np

def noise(rng, name, n):
    if name=='gaussian':return rng.standard_normal(n)
    if name=='t5':return rng.standard_t(5,n)*np.sqrt(3/5)
    if name=='t10':return rng.standard_t(10,n)*np.sqrt(8/10)
    if name=='exp':return rng.exponential(size=n)-1.0
    if name=='skew_mix':
        z=np.where(rng.random(n)<0.85,rng.normal(loc=-0.35,scale=0.6,size=n),
                   rng.normal(loc=1.98333333333333,scale=0.6,size=n))
        return z/np.sqrt(0.36 + 0.85*0.35**2 + 0.15*(1.98333333333333**2))
    raise ValueError(name)

def ar_path(rng, a=.9, n=600, dist='gaussian', scale=1.0, fault_at=None, burn=100):
    eps=noise(rng,dist,n+burn)
    if fault_at is not None: eps[burn+fault_at:]*=scale
    y=np.zeros(n+burn+1)
    for t in range(n+burn): y[t+1]=a*y[t]+eps[t]
    return y[burn:],eps[burn:]

def split_data(rng, a=.9, M=224, N=512, T=300, dist='gaussian',
               scale=1.0,fault_at=None):
    train,_=ar_path(rng,a,M,dist)
    cal,_=ar_path(rng,a,N,dist)
    online,_=ar_path(rng,a,T,dist,scale,fault_at)
    return train,cal,online

if __name__=='__main__':
    rng=np.random.default_rng(0)
    a=.9; y,e=ar_path(rng,a,400,'exp')
    assert np.allclose(y[1:]-a*y[:-1],e)
    print('simulate_core selftest PASS')
