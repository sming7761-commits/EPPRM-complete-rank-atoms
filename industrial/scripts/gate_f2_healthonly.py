from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2

EVENTS = [
('30102001',1,'f18',58800,59800),('09112001',1,'f16',57275,57550),
('09112001',1,'f18',58520,58625),('09112001',1,'f18',58830,58930),
('17112001',1,'f18',54600,54700),('17112001',1,'f16',56670,56770),
('20112001',1,'f17',37780,38400),('17112001',2,'f17',53780,53794),
('17112001',2,'f17',54193,54215),('17112001',2,'f19',55482,55517),
('17112001',2,'f19',55977,56015),('17112001',2,'f19',57030,57072),
('20112001',2,'f17',44400,45000),('30102001',3,'f18',57340,57890),
('09112001',3,'f16',60650,60700),('09112001',3,'f16',60870,60960),
('17112001',3,'f16',57475,57530),('17112001',3,'f16',57675,57800),
('17112001',3,'f19',58150,58325)]
CV={1:5,2:21,3:27}; X={1:6,2:22,3:28}
ORDERS=(1,2,4,8,12,20,30)
SEL_N=3000; VAL_N=1500; CAL_N=737; CONTROL_N=601; GAP=60; MON_N=601


def clusters():
    groups={}
    for d,a,f,s,e in EVENTS: groups.setdefault((d,a),[]).append((s,e,f))
    out=[]
    for (d,a),ev in sorted(groups.items()):
        cur=[]
        for s,e,f in sorted(ev):
            if not cur or s-cur[-1][1]>600:
                if cur: out.append((d,a,cur))
                cur=[(s,e,f)]
            else: cur.append((s,e,f))
        if cur: out.append((d,a,cur))
    return out


def design(x,p):
    y=x[p:]
    A=np.ones((len(y),p+1),float)
    for j in range(1,p+1): A[:,j]=x[p-j:len(x)-j]
    return y,A


def fit_ar_bic(selection):
    rows=[]; best=None
    for p in ORDERS:
        y,A=design(selection,p)
        beta=np.linalg.lstsq(A,y,rcond=None)[0]
        resid=y-A@beta; rss=max(float(resid@resid),np.finfo(float).tiny); n=len(y)
        bic=n*np.log(rss/n)+(p+1)*np.log(n)
        row=dict(order=p,bic=float(bic),rss=rss,n=n)
        rows.append(row)
        if best is None or (bic,p)<(best[0],best[1]): best=(bic,p,beta)
    return best[1],best[2],rows


def innovations_at(z,beta,p,start,end):
    idx=np.arange(start,end)
    A=np.ones((len(idx),p+1),float)
    for j in range(1,p+1): A[:,j]=z[idx-j]
    return z[idx]-A@beta


def metrics(x):
    x=np.asarray(x,float); y=x[1:]; z=x[:-1]; den=float(z@z)
    a=float(z@y/den) if den>1e-15 else np.nan
    e=y-a*z; e=e-np.mean(e); n=len(e); v=float(e@e); ac=[]
    for k in range(1,min(10,n-2)+1):
        ac.append(float(e[k:]@e[:-k]/v) if v>0 else np.nan)
    h=len(ac); Q=n*(n+2)*sum(ac[k-1]**2/(n-k) for k in range(1,h+1)) if h else np.nan
    lb=float(chi2.sf(Q,h)) if h else np.nan
    half=n//2; v1=float(np.var(e[:half],ddof=1)); v2=float(np.var(e[half:],ddof=1))
    vr=max(v1,v2)/max(min(v1,v2),1e-15)
    sd=max(float(np.std(e,ddof=1)),1e-15)
    md=abs(float(np.mean(e[:half])-np.mean(e[half:])))/sd
    tie=1-len(np.unique(np.round(e,12)))/max(n,1)
    rho=max(abs(q) for q in ac) if ac else np.nan
    ok=bool(np.isfinite(a) and abs(a)<.99 and rho<=.10 and lb>=.01 and
            vr<=2 and md<=.5 and tie<=.01 and n>=100)
    return dict(a_hat=a,max_abs_resid_acf_1_10=rho,ljung_box_p=lb,
                residual_variance_ratio=vr,residual_half_mean_shift_sd=md,
                residual_tie_fraction=tie,n_transitions=n,confirmatory_pass=int(ok))


def save_npz(path,**kw):
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**kw)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    rawdir=a.data/'extracted/damadics/Lublin_all_data'; features=a.output/'features'
    all_clusters=clusters(); by_unit={}
    for d,act,ev in all_clusters: by_unit.setdefault((d,act),[]).append(ev)
    unit_rows=[]; case_rows=[]; bic_rows=[]; cache={}
    for (date,act),unit_clusters in sorted(by_unit.items()):
        if date not in cache: cache[date]=np.loadtxt(rawdir/f'{date}.txt',dtype=float)
        state=cache[date][:,X[act]]-cache[date][:,CV[act]]
        earliest=min(q[0] for ev in unit_clusters for q in ev)
        fault_monitor_start=earliest-200
        control_end=fault_monitor_start-GAP
        control_start=control_end-CONTROL_N
        cal_end=control_start-GAP; cal_start=cal_end-CAL_N
        val_end=cal_start; val_start=val_end-VAL_N
        sel_end=val_start; sel_start=sel_end-SEL_N
        unit=f'{date}_A{act}'
        if sel_start<max(ORDERS) or fault_monitor_start+MON_N>len(state):
            unit_rows.append(dict(statistical_unit=unit,eligible=0,reason='insufficient_history',selected_order='',**metrics(np.arange(102,dtype=float))))
            continue
        mu=float(np.mean(state[sel_start:sel_end])); scale=max(float(np.std(state[sel_start:sel_end],ddof=1)),1e-12)
        z=(state-mu)/scale
        p,beta,bics=fit_ar_bic(z[sel_start:sel_end])
        for r in bics: bic_rows.append(dict(statistical_unit=unit,**r))
        val=innovations_at(z,beta,p,val_start,val_end); met=metrics(val)
        cal=innovations_at(z,beta,p,cal_start,cal_end)
        control=innovations_at(z,beta,p,control_start,control_end)
        eligible=bool(met['confirmatory_pass'])
        unit_rows.append(dict(statistical_unit=unit,eligible=int(eligible),reason='PASS' if eligible else 'heldout_validation_failed',selected_order=p,selection_start=sel_start,selection_end=sel_end,validation_start=val_start,validation_end=val_end,calibration_start=cal_start,calibration_end=cal_end,healthy_control_start=control_start,healthy_control_end=control_end,earliest_fault_onset=earliest,mean=mu,scale=scale,coefficients=json.dumps(beta.tolist()),**met))
        save_npz(features/f'F2_{unit}_HEALTH.npz',reference=cal,online=control,tau=np.int32(0),failed=np.int8(0),selected_order=np.int32(p),coefficients=beta,mean=np.float64(mu),scale=np.float64(scale))
        case_rows.append(dict(case_id=f'F2_{unit}_HEALTH',dataset='DAMADICS',statistical_unit=unit,label='health',fault_types='',tau=0,eligible=int(eligible),monitor_start=control_start,monitor_end=control_end))
        for ev in unit_clusters:
            onset=min(q[0] for q in ev); mstart=onset-200
            online=innovations_at(z,beta,p,mstart,mstart+MON_N)
            cid=f'F2_DAM_{date}_A{act}_{onset}'
            save_npz(features/f'{cid}.npz',reference=cal,online=online,tau=np.int32(201),failed=np.int8(1),selected_order=np.int32(p),coefficients=beta,mean=np.float64(mu),scale=np.float64(scale))
            case_rows.append(dict(case_id=cid,dataset='DAMADICS',statistical_unit=unit,label='fault',fault_types='+'.join(sorted(set(q[2] for q in ev))),tau=201,eligible=int(eligible),monitor_start=mstart,monitor_end=mstart+MON_N))
    for name,rows in [('unit_gate.csv',unit_rows),('case_registry.csv',case_rows),('bic_audit.csv',bic_rows)]:
        with (a.output/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary=dict(statistical_units=len(unit_rows),confirmatory_pass=sum(int(r['eligible']) for r in unit_rows),cases=len(case_rows),eligible_cases=sum(int(r['eligible']) for r in case_rows),fault_information_used_for_selection=False,protocol='chronological selection-validation-calibration-control-fault')
    (a.output/'gate_f2_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    protocol=dict(selection_length=SEL_N,validation_length=VAL_N,calibration_length=CAL_N,healthy_control_length=CONTROL_N,gap=GAP,monitor_length=MON_N,orders=ORDERS,selection_rule='minimum BIC on selection only',validation=dict(abs_a_lt=.99,max_abs_residual_acf_lags_1_10_le=.10,ljung_box_p_ge=.01,variance_ratio_le=2,half_mean_shift_sd_le=.5,tie_fraction_le=.01,n_transitions_ge=100))
    (a.output/'GATE_F2_PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print('GATE_F2_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__': main()
