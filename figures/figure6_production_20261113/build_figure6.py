"""Figure 6. Frozen formal records; recorded profiling cost, not an all-method benchmark."""
from pathlib import Path
import sys,json,csv,shutil,hashlib,zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
DATA=HERE/'source_data';OUT=HERE/'outputs';ASSET=HERE/'assets'
for p in [DATA,OUT,ASSET]:p.mkdir(exist_ok=True)
if not (ASSET/'nature_style.py').exists():shutil.copy2(ROOT/'figure1_production_20261113/assets/nature_style.py',ASSET/'nature_style.py')
sys.path.insert(0,str(ASSET))
from nature_style import panel,apply_style,landed_report
sources={
 'runtime_atoms.csv':ROOT/'EPPRM_RESS_TASK_E_RESULTS_REVIEW_20261112/01_source_data/FIG6_TRAJECTORY_RUNTIME_ATOMS.csv',
 'cost_power_frozen.csv':ROOT/'results_source_review/EPPRM_RESS_RESULTS_SOURCE_AND_DRAFT_20261112/01_source_data/FIG6C_GROUP_LEVEL_RESOURCE_POWER.csv'}
for name,p in sources.items():
    if not (DATA/name).exists():shutil.copy2(p,DATA/name)
archive=Path('C:/Users/symin/Desktop/winscp/EPPRM_COMPLETE_RESEARCH_PACKAGE_FORMAL_MAIN_COMPLETE_20261110_185236.zip')
if not (DATA/'timing_source_excerpt.txt').exists():
    with zipfile.ZipFile(archive) as z:
        p=next(n for n in z.namelist() if n.endswith('/run_sharded.py'))
        (DATA/'timing_source_excerpt.txt').write_text('\n'.join(z.read(p).decode().splitlines()[51:102]),encoding='utf-8')
        p=next(n for n in z.namelist() if n.endswith('/server_runtime_info.txt'))
        (DATA/'server_runtime_info.txt').write_bytes(z.read(p))

raw=pd.read_csv(DATA/'runtime_atoms.csv'); assert len(raw)==48000
assert not raw.duplicated(['trajectory_id','method']).any()
METHODS=['EP_PRM_736','Profiled_CCTM_fair'];DS=['gaussian','t5','exp_centered','skew_mixture']
DN=['Gaussian','t(5)','Exp − 1','Skew mixture'];SHORT=['G','t5','Exp','Mix']
COL={'EP_PRM_736':'#008F83','Profiled_CCTM_fair':'#7E64A6'}
NAME={'EP_PRM_736':'EP-PRM','Profiled_CCTM_fair':'Profiled CCTM'}
INK='#253345';GRAY='#7E8792';BLUE='#3C5488'
prof=raw[raw.method.isin(METHODS)].copy()
valid=prof.atom_count_final.notna() & np.isfinite(prof.runtime_seconds) & (prof.runtime_seconds>0)
excluded=prof[~valid].copy(); excluded.to_csv(DATA/'excluded_repair_records.csv',index=False)
prof=prof[valid];assert len(prof)==11994 and len(excluded)==6
wide=prof.pivot(index=['trajectory_id','distribution','scenario'],columns='method',values='runtime_seconds').dropna()
wide['runtime_ratio']=wide.EP_PRM_736/wide.Profiled_CCTM_fair
wide.reset_index().to_csv(DATA/'paired_runtime_ratios.csv',index=False)
fault=prof[prof.scenario=='fault'];assert len(fault)==4000
power=pd.read_csv(DATA/'cost_power_frozen.csv');power=power[power.method.isin(METHODS)].copy()
assert len(power)==8
summary=prof.groupby(['distribution','scenario','method']).agg(n=('runtime_seconds','size'),median_seconds=('runtime_seconds','median'),p95_seconds=('runtime_seconds',lambda x:x.quantile(.95)),median_atoms=('atom_count_final','median')).reset_index()
summary.to_csv(DATA/'runtime_distribution_summary.csv',index=False)
# Verify the plotting points against the frozen cost-power table before drawing.
for _,r in power.iterrows():
    q=summary[(summary.distribution==r.distribution)&(summary.method==r.method)&(summary.scenario=='fault')].iloc[0]
    assert abs(q.median_seconds-r.median_fault_run_seconds)<1e-9

spec=panel(180,height_mm=182,body_pt=9);apply_style(spec)
plt.rcParams.update({'savefig.bbox':None,'font.size':8,'axes.labelsize':7.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,'mathtext.cal':'STIXGeneral:italic'})
def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=2.5,width=.6)
    ax.grid(axis='y',ls='--',lw=.4,alpha=.20);ax.set_axisbelow(True)
def axes(fig,box,r):
    x,y,w,h=box;a,b,c,d=r;return fig.add_axes([x+a*w,y+b*h,c*w,d*h])
def ecdf(ax,vals,**kw):
    vals=np.sort(np.asarray(vals));ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',**kw)
def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return 100*(mid-half),100*(mid+half)
def draw_a(fig,box):
    for i,d in enumerate(DS):
        ax=axes(fig,box,[.07+i*.239,.19,.204,.70]);clean(ax)
        for m in METHODS:
            for sc,ls in [('health','-'),('fault','--')]:
                ecdf(ax,prof[(prof.distribution==d)&(prof.method==m)&(prof.scenario==sc)].runtime_seconds,color=COL[m],ls=ls,lw=1.15)
        ax.set(xlim=(0,22),ylim=(0,1.02),xticks=[0,10,20],yticks=[0,.5,1])
        if i==0:ax.set_ylabel('Cumulative fraction')
        else:ax.set_yticklabels([])
        ax.set_xlabel('Runtime (s)')
        ax.text(.5,1.10,DN[i],transform=ax.transAxes,ha='center',fontsize=8,color=INK)
        nh=int(((prof.distribution==d)&(prof.scenario=='health')).sum()/2)
        ax.text(.04,.12,f'H: {nh}\nF: 500',transform=ax.transAxes,fontsize=6.5,color=GRAY)

def draw_b(fig,box):
    # Fault-only scans: identical horizon, no healthy/fault mixing in the density.
    for i,m in enumerate(METHODS):
        ax=axes(fig,box,[.075+i*.49,.21,.385,.69]);clean(ax)
        q=fault[fault.method==m]
        hb=ax.hexbin(q.atom_count_final/1000,q.runtime_seconds,gridsize=(23,16),extent=(855,955,5,14),mincnt=1,cmap='Blues',norm=LogNorm(vmin=1,vmax=160),linewidths=.15,edgecolors='white')
        assert hb.get_array().max()<=160, 'Colour scale would clip counts'
        ax.set(xlim=(855,955),ylim=(5,14),xticks=[860,900,940],yticks=[6,10,14],xlabel='Final atoms (thousands)')
        if i==0:ax.set_ylabel('Runtime (s)')
        else:ax.set_yticklabels([])
        ax.text(.5,1.10,NAME[m]+' · fault scans',transform=ax.transAxes,ha='center',color=COL[m],fontsize=8)
        ax.text(.04,.91,'n = 2,000',transform=ax.transAxes,fontsize=7,color=GRAY)
    cax=axes(fig,box,[.975,.21,.014,.69]);cb=fig.colorbar(hb,cax=cax,ticks=[1,10,100]);cb.set_label('Trajectories / bin',fontsize=7);cb.ax.tick_params(labelsize=6.5,length=2)
    cb.solids.set_rasterized(False)

def draw_c(fig,box):
    ax=axes(fig,box,[.24,.23,.72,.68]);clean(ax)
    paired=wide.reset_index();paired=paired[paired.scenario=='fault']
    groups=[paired[paired.distribution==d].runtime_ratio.to_numpy() for d in DS]
    for i,x in enumerate(groups):
        # Fixed symmetric jitter is for visibility only; all 500 observations retained.
        jitter=np.random.default_rng(6133+i).uniform(-.15,.15,len(x))
        ax.scatter(x,i+jitter,s=2.2,color=BLUE,alpha=.13,lw=0,rasterized=False)
        qs=np.quantile(x,[.25,.5,.75]);ax.plot(qs[[0,2]],[i,i],color=BLUE,lw=3,solid_capstyle='butt')
        ax.scatter(qs[1],i,s=21,color='white',ec=BLUE,lw=1,zorder=3)
    ax.axvline(1,color=GRAY,lw=.8,ls='--')
    ax.set(yticks=range(4),yticklabels=DN,ylim=(3.5,-.5),xlim=(.8,1.6),xticks=[.8,1,1.2,1.4,1.6],xlabel='EP-PRM / CCTM runtime')

def draw_d(fig,box):
    ax=axes(fig,box,[.17,.23,.79,.68]);clean(ax)
    markers=['o','s','^','D']
    for i,d in enumerate(DS):
        for m in METHODS:
            r=power[(power.distribution==d)&(power.method==m)].iloc[0]
            y=100*r.observed_power; k=round(r.observed_power*500);l,u=wilson(k,500)
            ax.errorbar(r.median_fault_run_seconds,y,yerr=[[y-l],[u-y]],color=COL[m],marker=markers[i],ms=4.8,lw=.85,capsize=1.8,mec='white',mew=.4)
    marker_handles=[Line2D([],[],ls='none',marker=markers[i],color=GRAY,ms=4,label=SHORT[i]) for i in range(4)]
    ax.legend(handles=marker_handles,loc='lower left',bbox_to_anchor=(-.05,1.025),ncol=4,fontsize=6.5,frameon=False,columnspacing=.7,handletextpad=.25)
    ax.set(xlim=(7.7,10.6),ylim=(0,102),xticks=[8,9,10],yticks=[0,50,100],xlabel='Median fault runtime (s)',ylabel='Detection (%)')

draws=[draw_a,draw_b,draw_c,draw_d]
boxes=[(.015,.695,.945,.235),(.015,.385,.925,.245),(.015,.055,.465,.245),(.525,.055,.435,.245)]
titles=['Runtime distributions','Atom count and recorded runtime','Paired runtime ratios','Detection–cost trade-off']
fig=plt.figure(figsize=spec.figsize,facecolor='white')
fig.text(.035,.982,'The computational cost of complete parameter profiling',fontsize=11.3,fontweight='bold',color=INK,va='top')
for i,(fn,box,title) in enumerate(zip(draws,boxes,titles)):
    fn(fig,box);x,y,w,h=box;fig.text(x+.02,y+h+.012,chr(97+i),fontsize=11,fontweight='bold',color=INK);fig.text(x+.052,y+h+.012,title,fontsize=9,color=INK)
handles=[Line2D([],[],color=COL[m],lw=1.5,label=NAME[m]) for m in METHODS]+[Line2D([],[],color=GRAY,ls='-',label='Healthy'),Line2D([],[],color=GRAY,ls='--',label='Fault')]
fig.legend(handles=handles,loc='center',bbox_to_anchor=(.5,.684),ncol=4,frameon=False,fontsize=7.5,handlelength=2.4,columnspacing=2)
fig.text(.035,.016,'Recorded wall-clock profiling time · N = 736 · T = 600 · Xeon Gold 6133 server',fontsize=7,color=GRAY)
fig.canvas.draw();sizes,viol=landed_report(fig,spec);assert viol==0
for ext in ['svg','pdf','png']:fig.savefig(OUT/f'Figure6_v1.{ext}',dpi=450,bbox_inches=None)
plt.close(fig)
for i,fn in enumerate(draws):
    ps=panel(180 if i<2 else 88,height_mm=65,body_pt=9)
    f=plt.figure(figsize=ps.figsize,facecolor='white');fn(f,(.03,.07,.92,.85))
    for ext in ['svg','pdf','png']:f.savefig(OUT/f'Figure6_{chr(97+i)}.{ext}',dpi=300,bbox_inches=None)
    plt.close(f)
lf=plt.figure(figsize=(180/25.4,15/25.4));lf.legend(handles=handles,loc='center',ncol=4,frameon=False)
for ext in ['svg','pdf','png']:lf.savefig(OUT/f'Figure6_legend.{ext}',dpi=300,bbox_inches=None)
plt.close(lf)
report={'skill':'nature-figure-production','geometry_mm':[180,182],'text_sizes':sizes,'font_violations':viol,'total_source_rows':len(raw),'profile_rows':12000,'full_scan_rows_plotted':len(prof),'excluded_repair_rows':len(excluded),'fault_ratio_pairs':2000,'semantic_colors':COL,'timing':'perf_counter wall time inside compute_method; full end-to-end baseline timing unavailable','data_integrity':'Frozen fault median times matched exactly; no new experiments','caveats':['6 repair records omitted only from cost analysis because atom counts are absent.','Recorded wall time is not CPU time, isolated steady-state benchmark, or per-step latency.','Full time includes monitoring overhead and potential first-call compilation; no post hoc trimming.','Basic paths were precomputed outside this timer; therefore no all-method speed ranking is shown.','Only fixed N=736,T=600; no empirical asymptotic scaling claim.']}
report['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob('*') if p.is_file()}
(HERE/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'complete_scans':len(prof),'excluded':len(excluded),'fault_pairs':2000,'out':str(OUT)}))
