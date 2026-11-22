"""Frozen formal Figure 3: operating points and paired detection evidence."""
from pathlib import Path
import sys, shutil, json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
DATA=HERE/'source_data'; OUT=HERE/'outputs'; ASSET=HERE/'assets'
for p in [DATA,OUT,ASSET]: p.mkdir(exist_ok=True)
if not (ASSET/'nature_style.py').exists():
    shutil.copy2(ROOT/'figure6_production_20261113/assets/nature_style.py',ASSET/'nature_style.py')
sys.path.insert(0,str(ASSET))
from nature_style import panel, apply_style, landed_report
src=ROOT/'results_source_review/EPPRM_RESS_RESULTS_SOURCE_AND_DRAFT_20261112/01_source_data'
for name in ['FIG3A_FPR_POWER_POINTS.csv','FIG3B_PAIRED_EP_VS_OLDK8.csv','FIG3C_PAIRED_EP_VS_CCTM.csv']:
    if not (DATA/name).exists(): shutil.copy2(src/name, DATA/name)
if not (DATA/'method_records.csv').exists():
    shutil.copy2(ROOT/'formal_runtime_extract/EPPRM_COMPLETE_RESEARCH_PACKAGE/07_tidy_results/method_records.csv',DATA/'method_records.csv')
points=pd.read_csv(DATA/'FIG3A_FPR_POWER_POINTS.csv')
effects=pd.concat([pd.read_csv(DATA/'FIG3B_PAIRED_EP_VS_OLDK8.csv'),pd.read_csv(DATA/'FIG3C_PAIRED_EP_VS_CCTM.csv')],ignore_index=True)
raw=pd.read_csv(DATA/'method_records.csv'); assert len(raw)==48000
assert not raw.duplicated(['trajectory_id','method']).any()
DS=['gaussian','t5','exp_centered','skew_mixture']; DN=['Gaussian','t(5)','Exp − 1','Skew mixture']
METHODS=['EP_PRM_736','Profiled_CCTM_fair','Old_K8','OLS_PlugIn_PRM','Gaussian_GLR','Gaussian_CUSUM','Conformal_CUSUM','Oracle_PRM']
NAMES=['EP-PRM','Profiled CCTM','Old K8','Plug-in PRM','Gaussian GLR','Gaussian CUSUM','Conformal CUSUM','Oracle PRM']
COLORS=['#008F83','#7E64A6','#8A8F98','#3C5488','#A6761D','#E69F00','#C44E52','#303A46']
MARKERS=['o','s','v','D','^','P','X','h']
COL=dict(zip(METHODS,COLORS)); INK='#253345'; GRAY='#7E8792'
joint=[]
for d in DS:
    for m in METHODS:
        h=raw[(raw.distribution==d)&(raw.scenario=='health')&(raw.method==m)]
        f=raw[(raw.distribution==d)&(raw.scenario=='fault')&(raw.method==m)]
        q=points[(points.distribution==d)&(points.method==m)].iloc[0]
        assert len(h)==1000 and len(f)==500
        assert int((h.first_alarm>0).sum())==q.health_alarms
        assert int(f.detected_after_change.sum())==q.successful_detections
    f=raw[(raw.distribution==d)&(raw.scenario=='fault')]
    w=f.pivot(index='trajectory_id',columns='method',values='detected_after_change').astype(bool)
    assert len(w)==500
    for m in ['Old_K8','Profiled_CCTM_fair']:
        a=w.EP_PRM_736; b=w[m]
        counts=[int((a&b).sum()),int((a&~b).sum()),int((~a&b).sum()),int((~a&~b).sum())]
        e=effects[(effects.distribution==d)&(effects.method_b==m)].iloc[0]
        assert sum(counts)==500 and abs((counts[1]-counts[2])/500-e.paired_power_difference)<1e-12
        joint.append(dict(distribution=d,comparator=m,both=counts[0],ep_only=counts[1],comparator_only=counts[2],neither=counts[3],n=500))
joint=pd.DataFrame(joint);joint.to_csv(DATA/'paired_joint_outcomes.csv',index=False)
spec=panel(180,height_mm=181,body_pt=9); apply_style(spec)
plt.rcParams.update({'savefig.bbox':None,'font.size':8,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,'mathtext.cal':'STIXGeneral:italic'})
def axin(fig,box,rel):
    x,y,w,h=box; a,b,c,d=rel; return fig.add_axes([x+a*w,y+b*h,c*w,d*h])
def clean(ax):
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',ls='--',lw=.4,alpha=.2);ax.set_axisbelow(True)
def draw_a(fig,box):
    for i,d in enumerate(DS):
        ax=axin(fig,box,[.075+i*.235,.21,.195,.69]);clean(ax)
        ax.axvline(5,color=GRAY,ls='--',lw=.7,zorder=0)
        for j,m in enumerate(METHODS):
            r=points[(points.distribution==d)&(points.method==m)].iloc[0]
            x=100*r.observed_fpr;y=100*r.power
            ax.errorbar(x,y,xerr=[[x-100*r.fpr_ci95_low],[100*r.fpr_ci95_high-x]],yerr=[[y-100*r.power_ci95_low],[100*r.power_ci95_high-y]],color=COL[m],marker=MARKERS[j],ms=5.2 if j==0 else 3.7,lw=1.2 if j==0 else .65,capsize=1.4,mec='white',mew=.3,alpha=1 if j==0 else .8,zorder=5 if j==0 else 2)
        ax.set(xlim=(-.4,9),ylim=(-3,104),xticks=[0,5],yticks=[0,25,50,75,100],xlabel='False alarms (%)')
        if i==0:ax.set_ylabel('Successful detection (%)')
        else:ax.set_yticklabels([])
        ax.text(.5,1.10,DN[i],ha='center',transform=ax.transAxes,fontsize=8.5,color=INK)
def draw_b(fig,box):
    ax=axin(fig,box,[.235,.15,.72,.77]);clean(ax)
    ax.axvline(0,color=GRAY,ls='--',lw=.7)
    labels=[]
    for i,d in enumerate(DS):
        for j,m in enumerate(['Old_K8','Profiled_CCTM_fair']):
            y=2*i+j; r=effects[(effects.distribution==d)&(effects.method_b==m)].iloc[0]
            x=100*r.paired_power_difference; lo=100*r.bootstrap_ci95_low;hi=100*r.bootstrap_ci95_high
            ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],color=COL[m],marker=['v','s'][j],ms=4.5,capsize=2,lw=1.1)
            labels.append(DN[i] if j==0 else '')
            ax.text(114,y,f'{x:.1f}',ha='right',va='center',fontsize=7,color=INK)
        if i<3:ax.axhline(2*i+1.5,color='#E5E9EE',lw=.5)
    ax.set(ylim=(7.6,-.7),xlim=(-3,117),xticks=[0,25,50,75,100],yticks=range(8),yticklabels=labels,xlabel='Paired gain (percentage points)')
    ax.grid(False)
def draw_c(fig,box):
    ax=axin(fig,box,[.235,.15,.70,.77])
    arr=joint[['both','ep_only','comparator_only','neither']].to_numpy()
    cmap=LinearSegmentedColormap.from_list('count_teal',['#F4F8F8','#A2CEC7','#008F83','#005B54'])
    im=ax.pcolormesh(np.arange(5)-.5,np.arange(9)-.5,arr,cmap=cmap,vmin=0,vmax=500,shading='flat',rasterized=False)
    ax.set_xlim(-.5,3.5);ax.set_ylim(7.5,-.5)
    for i in range(8):
        for j in range(4):ax.text(j,i,str(arr[i,j]),ha='center',va='center',color='white' if arr[i,j]>275 else INK,fontsize=8)
    ax.set(xticks=range(4),xticklabels=['Both','EP only','Other only','Neither'],yticks=range(8),yticklabels=[f'{d} · {m}' for d in ['G','t5','Exp','Mix'] for m in ['K8','CCTM']])
    ax.tick_params(length=0,pad=4); ax.xaxis.tick_top(); ax.tick_params(axis='x',labelsize=7);ax.tick_params(axis='y',labelsize=7)
    for i in [1.5,3.5,5.5]:ax.axhline(i,color='white',lw=3)
    ax.set_xticks(np.arange(-.5,4),minor=True);ax.set_yticks(np.arange(-.5,8),minor=True)
    ax.grid(which='minor',color='white',lw=.7); ax.tick_params(which='minor',length=0)
    for s in ax.spines.values():s.set_visible(False)
    ca=axin(fig,box,[.235,.035,.70,.025]);cb=fig.colorbar(im,cax=ca,orientation='horizontal',ticks=[0,250,500]);cb.set_label('Paired fault trajectories',fontsize=7);cb.ax.tick_params(labelsize=7,length=2);cb.solids.set_rasterized(False)

draws=[draw_a,draw_b,draw_c]
boxes=[(.015,.575,.97,.315),(.015,.095,.47,.385),(.52,.095,.465,.385)]
fig=plt.figure(figsize=spec.figsize,facecolor='white')
fig.text(.035,.977,'Reliability and weak-fault detection under unknown dynamics',fontsize=11,fontweight='bold',color=INK,va='top')
fig.text(.035,.941,'Four noise laws · 1,000 healthy + 500 fault trajectories per law · scale ×1.5',fontsize=8,color=GRAY)
titles=['False-alarm–detection operating points','Paired detection gains','Joint detection outcomes']
for i,(fn,box,title) in enumerate(zip(draws,boxes,titles)):
    fn(fig,box);x,y,w,h=box;fig.text(x+.02,y+h+.018,chr(97+i),fontsize=11,fontweight='bold',color=INK);fig.text(x+.052,y+h+.018,title,fontsize=8.5,color=INK)
handles=[Line2D([],[],ls='none',marker=MARKERS[j],color=COL[m],ms=5,label=NAMES[j]) for j,m in enumerate(METHODS)]
fig.legend(handles=handles,loc='center',bbox_to_anchor=(.5,.555),ncol=4,frameon=False,columnspacing=1.4,handletextpad=.35,fontsize=7.5)
fig.text(.13,.473,'EP-PRM −',fontsize=7,color=GRAY)
fig.legend(handles=[handles[2],handles[1]],loc='center left',bbox_to_anchor=(.205,.476),ncol=2,frameon=False,fontsize=7,handletextpad=.3,columnspacing=.8)
fig.text(.035,.022,'N = 736 · T = 600 · fault onset = 357 · no threshold tuning on fault outcomes',fontsize=7,color=GRAY)
fig.canvas.draw(); sizes,viol=landed_report(fig,spec);assert viol==0
for ext in ['pdf','svg','png']:fig.savefig(OUT/f'Figure3_v1.{ext}',dpi=450,bbox_inches=None)
plt.close(fig)
for i,fn in enumerate(draws):
    sp=panel(180 if i==0 else 88,height_mm=75,body_pt=9)
    f=plt.figure(figsize=sp.figsize);fn(f,(.015,.11,.97,.82))
    for ext in ['pdf','svg','png']:f.savefig(OUT/f'Figure3_{chr(97+i)}.{ext}',dpi=300,bbox_inches=None)
    plt.close(f)
f=plt.figure(figsize=(180/25.4,18/25.4));f.legend(handles=handles,loc='center',ncol=4,frameon=False)
for ext in ['pdf','svg','png']:f.savefig(OUT/f'Figure3_legend.{ext}',dpi=300,bbox_inches=None)
plt.close(f)
report={'skill':'nature-figure-production','geometry_mm':[180,181],'font_violations':viol,'text_sizes':sizes,'source_records':48000,'operating_points':32,'paired_contrasts':8,'joint_tables':8,'pairing':'trajectory_id, fault only, 500 pairs per noise/comparison','numeric_validation':'All 32 health and fault totals match frozen source; all 8 discordant differences match frozen paired effects.','colors':COL,'no_new_experiments':True,'assembly':'Simple rectangular native vector assembly, with separate panel and legend exports.','statistical_boundaries':['Operating points have different threshold/calibration protocols; not equal-FPR superiority.','Zero observed false alarms are not zero population risk.','Power excludes prefault alarms and retains all 500 fault trajectories in denominator.','K8 symmetry guarantee does not cover skewed innovations.','Frozen bootstrap intervals reused, not recomputed or selected.']}
report['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob('*.csv')}
(HERE/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('FIGURE3_COMPLETE',OUT)
