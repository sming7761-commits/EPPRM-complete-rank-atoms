"""Figures 4 and 5 from frozen statistical tables and archived industrial traces."""
from pathlib import Path
import sys,json,hashlib,shutil,zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

H=Path(__file__).resolve().parent;ROOT=H.parent;DATA=H/'source_data';OUT=H/'outputs';ASSET=H/'assets'
for p in [DATA,OUT,ASSET]:p.mkdir(exist_ok=True)
if not (ASSET/'nature_style.py').exists():shutil.copy2(ROOT/'figure6_production_20261113/assets/nature_style.py',ASSET/'nature_style.py')
sys.path.insert(0,str(ASSET));from nature_style import panel,apply_style,landed_report
S=ROOT/'results_source_review/EPPRM_RESS_RESULTS_SOURCE_AND_DRAFT_20261112/01_source_data'
for name in ['FIG4A_OPERATING_POINTS_5PCT_1PCT.csv','FIG4B_PAIRED_EP_VS_CALIBRATED.csv','FIG4C_ORACLE_GAP.csv','FIG5BC_DAMADICS_F3_AVAILABLE.csv']:
    if not (DATA/name).exists():shutil.copy2(S/name,DATA/name)
for name in ['FIG5A_F2_UNIT_GATE.csv','FIG5A_F2_CASE_REGISTRY.csv']:
    if not (DATA/name).exists():shutil.copy2(ROOT/'EPPRM_RESS_TASK_E_RESULTS_REVIEW_20261112/01_source_data'/name,DATA/name)
if not (DATA/'formal_method_records.csv').exists():shutil.copy2(ROOT/'formal_runtime_extract/EPPRM_COMPLETE_RESEARCH_PACKAGE/07_tidy_results/method_records.csv',DATA/'formal_method_records.csv')
for archive,prefix in [('EPPRM_GATE_F3_DETECTION_RESULTS_20261112.zip','F3'),('EPPRM_GATE_F2_HEALTHONLY_RESULTS_20261112.zip','F2')]:
    target=DATA/prefix
    if not target.exists():
        target.mkdir()
        with zipfile.ZipFile(Path('C:/Users/symin/Desktop/winscp')/archive) as z:
            for n in z.namelist():
                if n.endswith('.npz') or n.endswith('case_method_results.csv') or n.endswith('GATE_F3_PROTOCOL.json') or n.endswith('GATE_F2_PROTOCOL.json'):
                    dst=target/Path(n).name;dst.write_bytes(z.read(n))
DS=['gaussian','t5','exp_centered','skew_mixture'];DN=['Gaussian','t(5)','Exp − 1','Skew mixture']
BASE=['OLS_PlugIn_PRM','Gaussian_GLR','Gaussian_CUSUM','Conformal_CUSUM']
NAME={'EP_PRM_736':'EP-PRM','EP_PRM':'EP-PRM','Profiled_CCTM':'Profiled CCTM','OLS_PlugIn_PRM':'Plug-in PRM','Gaussian_GLR':'Gaussian GLR','Gaussian_CUSUM':'Gaussian CUSUM','Conformal_CUSUM':'Conformal CUSUM','Oracle_PRM':'Oracle PRM'}
COL={'EP_PRM_736':'#008F83','EP_PRM':'#008F83','Profiled_CCTM':'#7E64A6','OLS_PlugIn_PRM':'#3C5488','Gaussian_GLR':'#A6761D','Gaussian_CUSUM':'#E69F00','Conformal_CUSUM':'#C44E52','Oracle_PRM':'#303A46'}
INK='#253345';GRAY='#7E8792';ORANGE='#D55E00'
spec=panel(180,height_mm=184,body_pt=9);apply_style(spec)
plt.rcParams.update({'savefig.bbox':None,'font.size':8,'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'mathtext.cal':'STIXGeneral:italic'})
def axin(f,b,r):
    x,y,w,h=b;a,c,d,e=r;return f.add_axes([x+a*w,y+c*h,d*w,e*h])
def clean(ax):
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',ls='--',lw=.4,alpha=.2);ax.set_axisbelow(True)
def save(f,name):
    f.canvas.draw();sizes,viol=landed_report(f,spec);assert viol==0
    for ext in ['pdf','svg','png']:f.savefig(OUT/f'{name}.{ext}',dpi=450,bbox_inches=None)
    im=Image.open(OUT/f'{name}.png');im.thumbnail((1500,1600));im.save(OUT/f'{name}_preview.png')
    plt.close(f);return {'text_sizes':sizes,'font_violations':viol}
def head(f,b,letter,title):
    x,y,w,h=b;f.text(x+.015,y+h+.012,letter,fontweight='bold',fontsize=11,color=INK);f.text(x+.047,y+h+.012,title,fontsize=8.3,color=INK)
op=pd.read_csv(DATA/'FIG4A_OPERATING_POINTS_5PCT_1PCT.csv');eff=pd.read_csv(DATA/'FIG4B_PAIRED_EP_VS_CALIBRATED.csv');gap=pd.read_csv(DATA/'FIG4C_ORACLE_GAP.csv');raw=pd.read_csv(DATA/'formal_method_records.csv')
assert len(op)==40 and len(eff)==32 and len(gap)==4 and len(raw)==48000
assert op.observed_fpr.between(0,.075).all() and op.observed_power.between(0,1).all()
assert eff.ci95_low.min()>=-.40 and eff.ci95_high.max()<=.65
curve=[]
for d in DS:
    for m in ['EP_PRM_736','Conformal_CUSUM','Oracle_PRM']:
        q=raw[(raw.distribution==d)&(raw.scenario=='fault')&(raw.method==m)];assert len(q)==500
        assert ((q.first_alarm>=357)&(q.first_alarm<=600)).sum()==q.detected_after_change.sum()
        for t in range(245):curve.append(dict(distribution=d,method=m,post_change_steps=t,cumulative_detection=((q.first_alarm>=357)&(q.first_alarm<=356+t)).sum()/500))
curve=pd.DataFrame(curve);curve.to_csv(DATA/'FIG4_CUMULATIVE_DETECTION.csv',index=False)
def f4a(f,b):
    for i,d in enumerate(DS):
        ax=axin(f,b,[.075+i*.235,.20,.195,.69]);clean(ax)
        for m in BASE:
            q=op[(op.distribution==d)&(op.method==m)]
            a=q[q.setting=='FIXED_1PCT_TARGET'].iloc[0];c=q[q.setting=='FROZEN_5PCT_MAIN'].iloc[0]
            ax.plot(100*np.array([a.observed_fpr,c.observed_fpr]),100*np.array([a.observed_power,c.observed_power]),color=COL[m],lw=.85)
            for r,mk in [(a,'o'),(c,'s')]:ax.plot(100*r.observed_fpr,100*r.observed_power,marker=mk,color=COL[m],ms=4,mec='white',mew=.3)
        ep=op[(op.distribution==d)&(op.method=='EP_PRM_736')].iloc[0]
        ax.plot(100*ep.observed_fpr,100*ep.observed_power,'*',color=COL['EP_PRM_736'],ms=8,mec='white',mew=.4,zorder=9)
        ax.set(xlim=(0,7.5),ylim=(0,103),xticks=[0,1,5],yticks=[0,50,100],xlabel='Test false alarms (%)')
        if i==0:ax.set_ylabel('Detection (%)')
        else:ax.set_yticklabels([])
        ax.text(.5,1.12,DN[i],transform=ax.transAxes,ha='center',fontsize=8,color=INK)
def f4b(f,b):
    # Four noise-law columns, same effect scale and four baseline rows.
    for i,d in enumerate(DS):
        ax=axin(f,b,[.19+i*.20,.20,.175,.69]);clean(ax);ax.axvline(0,color=GRAY,ls='--',lw=.6)
        for j,m in enumerate(BASE):
            for setting,mk,dy in [('FIXED_ONE_PERCENT','o',-.13),('FROZEN_5PCT_MAIN','s',.13)]:
                r=eff[(eff.distribution==d)&(eff.method_b==m)&(eff.comparison_setting==setting)].iloc[0]
                x=100*r.paired_power_difference;lo=100*r.ci95_low;hi=100*r.ci95_high
                ax.errorbar(x,j+dy,xerr=[[x-lo],[hi-x]],color=COL[m],marker=mk,ms=3,lw=.75,capsize=1.2)
        ax.set(xlim=(-40,65),ylim=(3.55,-.65),xticks=[-30,0,60],yticks=range(4),yticklabels=['Plug-in','GLR','G-CUSUM','C-CUSUM'] if i==0 else ['']*4)
        ax.text(.5,1.08,['G','t5','Exp','Mix'][i],transform=ax.transAxes,ha='center',fontsize=7.5)
    f.text(b[0]+b[2]*.58,b[1]+b[3]*.035,'EP-PRM − baseline (percentage points)',ha='center',fontsize=7.5)
def f4d(f,b):
    ax=axin(f,b,[.18,.20,.76,.69]);clean(ax);ax.axvline(0,color=GRAY,lw=.7,ls='--')
    for i,d in enumerate(DS):
        r=gap[gap.distribution==d].iloc[0];x=-100*r.ep_minus_oracle_paired;lo=-100*r.ci95_high;hi=-100*r.ci95_low
        ax.errorbar(x,i,xerr=[[x-lo],[hi-x]],marker='o',color=COL['EP_PRM_736'],ms=4,lw=1,capsize=2)
    ax.set(ylim=(3.55,-.65),xlim=(0,17),xticks=[0,8,16],yticks=range(4),yticklabels=['G','t5','Exp','Mix'],xlabel='Oracle − EP (pp)')
def f4c(f,b):
    for i,d in enumerate(DS):
        ax=axin(f,b,[.075+i*.235,.21,.195,.69]);clean(ax)
        for m,ls in [('EP_PRM_736','-'),('Conformal_CUSUM','-'),('Oracle_PRM','--')]:
            q=curve[(curve.distribution==d)&(curve.method==m)];ax.step(q.post_change_steps,100*q.cumulative_detection,where='post',color=COL[m],ls=ls,lw=1.15)
        ax.set(xlim=(0,244),ylim=(0,102),xticks=[0,120,240],yticks=[0,50,100],xlabel='Post-change steps')
        if i==0:ax.set_ylabel('Detected / all 500 (%)')
        else:ax.set_yticklabels([])
        ax.text(.5,1.08,DN[i],transform=ax.transAxes,ha='center',fontsize=8)

f=plt.figure(figsize=spec.figsize);f.text(.035,.976,'Detection depends on the error budget and the time horizon',fontsize=11,fontweight='bold',color=INK,va='top')
f.text(.035,.941,'Independent calibration targets · paired comparisons · complete fault-trajectory denominators',fontsize=7.8,color=GRAY)
b4=[(.01,.655,.975,.235),(.01,.335,.70,.225),(.01,.045,.975,.205),(.75,.335,.23,.225)]
for fn,b,l,t in zip([f4a,f4b,f4c,f4d],b4,'abcd',['Calibration operating points','Paired gains at 1% and 5% targets','Cumulative successful detection','Oracle gap']):fn(f,b);head(f,b,l,t)
h4=[Line2D([],[],color=COL[m],lw=1.3,label=NAME[m]) for m in ['EP_PRM_736']+BASE]
f.legend(handles=h4,loc='center',bbox_to_anchor=(.51,.63),ncol=5,frameon=False,fontsize=6.7,columnspacing=1,handlelength=1.4)
f.text(.12,.595,'Circles: 1% target     Squares: 5% target     Star: EP fixed threshold',fontsize=7,color=GRAY)
hcurve=[Line2D([],[],color=COL[m],ls=ls,label=NAME[m]) for m,ls in [('EP_PRM_736','-'),('Conformal_CUSUM','-'),('Oracle_PRM','--')]]
f.legend(handles=hcurve,loc='center',bbox_to_anchor=(.53,.29),ncol=3,frameon=False,fontsize=7)
a4=save(f,'Figure4_v1')

units=pd.read_csv(DATA/'FIG5A_F2_UNIT_GATE.csv');reg=pd.read_csv(DATA/'FIG5A_F2_CASE_REGISTRY.csv');res=pd.read_csv(DATA/'F3/case_method_results.csv');sm=pd.read_csv(DATA/'FIG5BC_DAMADICS_F3_AVAILABLE.csv')
assert len(units)==9 and units.eligible.sum()==5 and len(res)==72
reg=reg[reg.eligible==1].sort_values(['statistical_unit','label','monitor_start'],ascending=[True,False,True]);assert len(reg)==12
units=units.sort_values('statistical_unit');unitids=units[units.eligible==1].statistical_unit.tolist()
# Explicit case order: each unit's healthy control, then chronological fault clusters.
ids=[]
for u in unitids:
    sub=reg[reg.statistical_unit==u];ids+=sub[sub.label=='health'].case_id.tolist()+sub[sub.label=='fault'].sort_values('monitor_start').case_id.tolist()
methods=['EP_PRM','Profiled_CCTM']+BASE
for m in methods:
    r=res[res.method==m];s=sm[sm.method==m].iloc[0]
    assert r.false_alarm.sum()==s.health_false_alarms
    assert r.valid_detection.sum()==s.all_cluster_detections
    assert r[r.primary_fault_case==1].valid_detection.sum()==s.primary_detections
mat=[]
for m in methods:
    for cid in ids:
        r=res[(res.method==m)&(res.case_id==cid)].iloc[0]
        status='healthy quiet' if r.label=='health' and not r.false_alarm else 'healthy alarm' if r.label=='health' else 'prefault alarm' if r.pre_fault_alarm else 'detected' if r.valid_detection else 'not detected'
        mat.append(dict(method=m,case_id=cid,status=status,first_alarm=r.first_alarm))
pd.DataFrame(mat).to_csv(DATA/'FIG5_OUTCOME_MATRIX.csv',index=False)
ep=res[(res.method=='EP_PRM')&(res.primary_fault_case==1)].sort_values(['statistical_unit','case_id'])
selected=[ep[ep.valid_detection==1].iloc[0].case_id,ep[(ep.valid_detection==0)&(ep.pre_fault_alarm==0)].iloc[0].case_id]
(DATA/'FIG5_CASE_SELECTION.json').write_text(json.dumps({'rule':'First EP-detected primary case and first non-detected/no-prefault primary case in statistical_unit then case_id order. Illustrative contrast, not performance estimate.','case_ids':selected},indent=2))
ordered=reg.set_index('case_id').loc[ids].reset_index();ordered.insert(0,'figure_column',range(1,13));ordered.to_csv(DATA/'FIG5_CASE_COLUMN_ORDER.csv',index=False)
selected_rows=[]
for cid in selected:
    ft=np.load(DATA/'F2'/f'{cid}.npz');tau=int(ft['tau'])
    for m in ['EP_PRM','Conformal_CUSUM']:
        tr=np.load(DATA/'F3'/f'{cid}_{m}.npz');th=float(tr['threshold'])
        for t,(x,v) in enumerate(zip(ft['online'],tr['value'])):selected_rows.append(dict(case_id=cid,method=m,monitor_step=t,steps_from_fault=t-tau,frozen_feature=x,statistic=v,threshold=th,plot_log10_1plus_ratio=np.log10(1+v/th)))
pd.DataFrame(selected_rows).to_csv(DATA/'FIG5_SELECTED_TIME_SERIES.csv',index=False)
def fmtu(u):return f'{u[:2]}/{u[2:4]} · {u.split("_")[1]}'
def f5a(f,b):
    ax=axin(f,b,[.03,.07,.93,.85]);ax.set_axis_off()
    for x,t in [(.01,'Health unit'),(.36,'Held-out gate'),(.66,'AR order'),(.85,'H / F')]:ax.text(x,1.02,t,transform=ax.transAxes,fontsize=7.3,color=GRAY)
    for i,(_,r) in enumerate(units.iterrows()):
        y=.94-i*.102;passed=bool(r.eligible);color=COL['EP_PRM'] if passed else GRAY
        ax.text(.01,y,fmtu(r.statistical_unit),transform=ax.transAxes,fontsize=7.5,va='center')
        ax.plot(.40,y,'o' if passed else 'x',color=color,ms=4,transform=ax.transAxes)
        ax.text(.45,y,'Pass' if passed else 'Fail',color=color,fontsize=7.3,transform=ax.transAxes,va='center')
        ax.text(.71,y,str(r.selected_order),fontsize=7.5,transform=ax.transAxes,va='center',ha='center')
        q=reg[reg.statistical_unit==r.statistical_unit];txt=f'{(q.label=="health").sum()} / {(q.label=="fault").sum()}' if passed else '—'
        ax.text(.88,y,txt,fontsize=7.3,transform=ax.transAxes,va='center',ha='center')
def f5b(f,b):
    ax=axin(f,b,[.20,.20,.77,.66]);ax.set_xlim(-.5,11.5);ax.set_ylim(5.6,-1.2);ax.spines[:].set_visible(False)
    for j,cid in enumerate(ids):
        rr=reg[reg.case_id==cid].iloc[0];xmarker='H' if rr.label=='health' else 'F*' if bool(res[(res.case_id==cid)&(res.method=='EP_PRM')].iloc[0].primary_fault_case) else 'F'
        ax.text(j,-.85,xmarker,ha='center',va='center',fontsize=7,color=INK)
        for i,m in enumerate(methods):
            r=res[(res.method==m)&(res.case_id==cid)].iloc[0]
            if r.label=='health':marker='o' if not r.false_alarm else 'X';col=GRAY if not r.false_alarm else ORANGE;fc='white' if not r.false_alarm else col
            elif r.pre_fault_alarm:marker='X';col=ORANGE;fc=col
            elif r.valid_detection:marker='o';col=COL['EP_PRM'];fc=col
            else:marker='o';col=GRAY;fc='white'
            ax.scatter(j,i,marker=marker,s=29,color=fc,edgecolor=col,linewidth=.8)
    start=0
    for u in unitids:
        n=sum(reg.statistical_unit==u);ax.axvspan(start-.5,start+n-.5,color='#F0F4F6' if unitids.index(u)%2==0 else 'white',zorder=-3)
        ax.text(start+(n-1)/2,-1.65,fmtu(u),fontsize=6.8,ha='center',clip_on=False)
        if start:ax.axvline(start-.5,color='#CDD5DC',lw=.6)
        start+=n
    ax.set(xticks=[],yticks=range(6),yticklabels=[NAME[m] for m in methods]);ax.tick_params(length=0,pad=5)
    ax.set_xlabel('H: healthy control   F*: primary fault   F: additional cluster',fontsize=6.7,labelpad=8)
def f5c(f,b):
    for i,cid in enumerate(selected):
        zz=np.load(DATA/'F2'/f'{cid}.npz');on=zz['online'];tau=int(zz['tau']);assert len(on)==601
        ax=axin(f,b,[.08+i*.49,.58,.40,.28]);clean(ax);t=np.arange(len(on))-tau
        ax.plot(t,on,color='#3C5488',lw=.65);ax.axvline(0,color=ORANGE,lw=.9,ls='--');ax.set(xlim=(-200,399),xticks=[-200,0,200,400]);ax.set_xticklabels([])
        if i==0:ax.set_ylabel('Frozen feature')
        title=fmtu(reg[reg.case_id==cid].iloc[0].statistical_unit)+' · '+str(reg[reg.case_id==cid].iloc[0].fault_types)
        ax.text(.5,1.14,title,transform=ax.transAxes,ha='center',fontsize=8,color=INK)
        ax2=axin(f,b,[.08+i*.49,.18,.40,.29]);clean(ax2)
        for m,ls in [('EP_PRM','-'),('Conformal_CUSUM','-')]:
            zz=np.load(DATA/'F3'/f'{cid}_{m}.npz');val=zz['value'];th=float(zz['threshold']);assert len(val)==601
            # log1p normalized statistic preserves exact zero without a numerical floor.
            ax2.plot(t,np.log10(1+val/th),color=COL[m],lw=1.05,ls=ls)
        ax2.axhline(np.log10(2),color=GRAY,ls=':',lw=.9);ax2.axvline(0,color=ORANGE,ls='--',lw=.9)
        ax2.set(xlim=(-200,399),xticks=[-200,0,200,400],xlabel='Steps from annotated fault onset')
        if i==0:ax2.set_ylabel('log10(1 + stat / threshold)',fontsize=6.8)
        # Verify threshold crossing from the archived complete trace.
        for m in ['EP_PRM','Conformal_CUSUM']:
            z=np.load(DATA/'F3'/f'{cid}_{m}.npz');cross=np.flatnonzero(z['value'][1:]>=float(z['threshold']));fa=int(cross[0]+1) if len(cross) else 0
            assert fa==int(res[(res.case_id==cid)&(res.method==m)].iloc[0].first_alarm)

f=plt.figure(figsize=spec.figsize);f.text(.035,.976,'Industrial evidence: applicability, detection and missed faults',fontsize=11,fontweight='bold',color=INK,va='top')
f.text(.035,.941,'DAMADICS · 9 candidate health units → 5 eligible units · 5 controls + 7 fault clusters',fontsize=7.8,color=GRAY)
b5=[(.02,.55,.40,.34),(.45,.55,.53,.34),(.02,.055,.96,.36)]
for fn,b,l,t in zip([f5a,f5b,f5c],b5,'abc',['Health-only qualification','All eligible cases','Detected and non-detected primary cases']):fn(f,b);head(f,b,l,t)
hout=[Line2D([],[],ls='none',marker='o',mfc='white',mec=GRAY,label='Quiet / not detected'),Line2D([],[],ls='none',marker='o',color=COL['EP_PRM'],label='Post-fault detection'),Line2D([],[],ls='none',marker='X',color=ORANGE,label='Healthy / pre-fault alarm')]
f.legend(handles=hout,loc='center',bbox_to_anchor=(.52,.51),ncol=3,frameon=False,fontsize=7,columnspacing=1.2)
f.text(.035,.463,'EP-PRM: 0/5 healthy alarms · 1/5 primary faults detected · 2/7 fault clusters detected',fontsize=8.2,color=COL['EP_PRM'])
f.legend(handles=[Line2D([],[],color=COL[m],label=NAME[m]) for m in ['EP_PRM','Conformal_CUSUM']]+[Line2D([],[],ls=':',color=GRAY,label='Threshold')],loc='center',bbox_to_anchor=(.53,.068),ncol=3,frameon=False,fontsize=7)
a5=save(f,'Figure5_v1')
for num,draws,boxes in [(4,[f4a,f4b,f4c,f4d],b4),(5,[f5a,f5b,f5c],b5)]:
    for i,fn in enumerate(draws):
        width=180 if (num==4 and i<3) or (num==5 and i==2) else 88
        ps=panel(width,height_mm=80,body_pt=9);ff=plt.figure(figsize=ps.figsize);fn(ff,(.02,.10,.94,.80))
        for ext in ['pdf','svg','png']:ff.savefig(OUT/f'Figure{num}_{chr(97+i)}.{ext}',dpi=300,bbox_inches=None)
        plt.close(ff)
for num,handles in [(4,h4+hcurve[1:]),(5,hout)]:
    ff=plt.figure(figsize=(180/25.4,18/25.4));ff.legend(handles=handles,loc='center',ncol=4 if num==4 else 3,frameon=False,fontsize=7)
    for ext in ['pdf','svg','png']:ff.savefig(OUT/f'Figure{num}_legend.{ext}',dpi=300,bbox_inches=None)
    plt.close(ff)
audit={'skill':'nature-figure-production','geometry_mm':[180,184],'Figure4':a4,'Figure5':a5,'colors':COL,'source_records':48000,'industrial_case_method_records':72,'industrial_case_count':12,'eligible_units':5,'candidate_units':9,'selected_examples':selected,'no_new_experiments':True,'checks':['All 6 industrial method summaries match frozen endpoint counts.','Selected industrial trace first crossings match frozen first_alarm.','Cumulative detection uses all 500 fault trajectories, excludes prefault alarms.','1%/5% are independent calibration targets, not matched observed test FPR.'],'assembly':'Rectangular editable vector assembly; all panels and standalone legends exported.'}
audit['source_hashes']={str(p.relative_to(DATA)):hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.rglob('*') if p.is_file()}
(H/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print('FIGURES_4_5_COMPLETE',OUT)
