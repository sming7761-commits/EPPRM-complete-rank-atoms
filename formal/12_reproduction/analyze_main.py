"""CSV-only formal statistics; incomplete groups clearly marked. No figures."""
from __future__ import annotations
from pathlib import Path
import sys,csv,collections,math,json
import numpy as np
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[1]
DIST=['gaussian','t5','exp_centered','skew_mixture']
METHOD=['EP_PRM_736','Profiled_CCTM_fair','OLS_PlugIn_PRM','Gaussian_GLR','Gaussian_CUSUM','Conformal_CUSUM','Old_K8','Oracle_PRM']

def output(name,rows,fields):
 p=ROOT/'08_statistical_analysis'/name
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 print('SAVED',p,'rows',len(rows))

def main():
 f=ROOT/'07_tidy_results/method_records.csv'
 rows=list(csv.DictReader(f.open())) if f.exists() else []
 groups=collections.defaultdict(list)
 for r in rows:groups[(r['distribution'],r['scenario'],r['method'])].append(r)
 perf=[];fa=[];delay=[];runtime=[];paired=[]
 for dist in DIST:
  for m in METHOD:
   for scene,target in [('health',1000),('fault',500)]:
    r=groups[(dist,scene,m)];n=len(r);complete=n==target
    valid=[x for x in r if int(x['first_alarm'])>=0 and x['numeric_status'] not in ('UNKNOWN','ERROR')]
    alarm=sum(int(x['first_alarm'])>0 for x in valid)
    detected=sum(int(x['detected_after_change']) for x in valid)
    early=sum(int(x['prefault_alarm']) for x in valid)
    report=dict(distribution=dist,scenario=scene,method=m,expected=target,processed=n,valid=len(valid),
       status='COMPLETE' if complete and n==len(valid) else 'INCOMPLETE_DO_NOT_REPORT_AS_FORMAL',
       alarms=alarm,early_alarms=early,detected=detected,
       alarm_rate=alarm/len(valid) if valid else '',
       detection_rate=detected/len(valid) if valid and scene=='fault' else '',
       numeric_failures=n-len(valid))
    perf.append(report)
    if scene=='health':
     ci=binomtest(alarm,len(valid)).proportion_ci(.95,method='exact') if valid else None
     fa.append(dict(**report,ci_low=ci.low if ci else '',ci_high=ci.high if ci else '',
                    one_sided_95_upper=binomtest(alarm,len(valid),alternative='less').proportion_ci(.95,method='exact').high if valid else ''))
    if scene=='fault':
     delays=[int(float(x['delay'])) for x in valid]
     delay.append(dict(distribution=dist,method=m,processed=n,status=report['status'],
       median_censored_delay=float(np.median(delays)) if delays else '',
       mean_censored_delay=float(np.mean(delays)) if delays else '',
       undetected_or_prefault=len(valid)-detected,
       detected_median_delay=float(np.median([int(x['delay']) for x in valid if int(x['detected_after_change'])])) if detected else ''))
    secs=[float(x['runtime_seconds']) for x in valid]
    atom=[int(x['atom_count_final']) for x in valid if x['atom_count_final'] not in ('','None')]
    runtime.append(dict(distribution=dist,scenario=scene,method=m,processed=n,status=report['status'],
         median_seconds=float(np.median(secs)) if secs else '',total_seconds=sum(secs),
         median_final_atoms=float(np.median(atom)) if atom else '', numeric_failures=n-len(valid)))
  for competitor in METHOD:
   if competitor=='EP_PRM_736':continue
   a={x['trajectory_id']:x for x in groups[(dist,'fault','EP_PRM_736')] if int(x['first_alarm'])>=0}
   b={x['trajectory_id']:x for x in groups[(dist,'fault',competitor)] if int(x['first_alarm'])>=0}
   keys=sorted(set(a)&set(b));n=len(keys)
   if n:
    x=np.array([int(a[k]['detected_after_change'])-int(b[k]['detected_after_change']) for k in keys],float)
    diff=float(x.mean())
    if n>=30:
     rng=np.random.default_rng(202611090011)
     boots=np.mean(x[rng.integers(0,n,(5000,n))],axis=1)
     lo,hi=np.quantile(boots,[.025,.975])
    else:lo=hi='NOT_COMPUTED_N_LT_30'
   else:diff=lo=hi=''
   paired.append(dict(distribution=dist,method_a='EP_PRM_736',method_b=competitor,
           n_paired=n,difference=diff,bootstrap_ci_low=lo,bootstrap_ci_high=hi,
           status='COMPLETE' if n==500 else 'INCOMPLETE_DO_NOT_REPORT_AS_FORMAL'))
 output('main_performance.csv',perf,list(perf[0]))
 output('false_alarm_intervals.csv',fa,list(fa[0]))
 output('delay_summary.csv',delay,list(delay[0]))
 output('runtime_summary.csv',runtime,list(runtime[0]))
 output('paired_power_differences.csv',paired,list(paired[0]))
if __name__=='__main__':main()
