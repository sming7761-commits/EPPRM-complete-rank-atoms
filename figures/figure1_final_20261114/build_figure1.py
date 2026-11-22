"""Figure 1: exact residual crossings, complete historical ranks, audited singleton minimum."""
from pathlib import Path
from fractions import Fraction as Q
import sys,json,csv,shutil,hashlib
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Polygon,Circle,FancyBboxPatch
from matplotlib.lines import Line2D
from PIL import Image

H=Path(__file__).resolve().parent;ROOT=H.parent;DATA=H/'source_data';OUT=H/'outputs';ASSET=H/'assets'
for p in [DATA,OUT,ASSET]:p.mkdir(exist_ok=True)
if not (ASSET/'nature_style.py').exists():shutil.copy2(ROOT/'figure1_production_20261113/assets/nature_style.py',ASSET/'nature_style.py')
sys.path.insert(0,str(ASSET));from nature_style import panel,apply_style,landed_report
for fn in ['illustrative_geometry.csv','complete_atoms.csv','audited_cctm_root_wealth.json']:
    if not (DATA/fn).exists():shutil.copy2(ROOT/'figure1_production_20261113/source_data'/fn,DATA/fn)
with (DATA/'illustrative_geometry.csv').open(newline='',encoding='utf-8-sig') as g:rows=list(csv.DictReader(g))
ref=[r for r in rows if r['stream']=='reference'];on=[r for r in rows if r['stream']=='online']
cy=[Q(r['y_exact']) for r in ref];cx=[Q(r['x_exact']) for r in ref]
oy=[Q(r['y_exact']) for r in on];ox=[Q(r['x_exact']) for r in on]
lo,hi=Q(-999,1000),Q(999,1000);events={}
for t,(y,x) in enumerate(zip(oy,ox)):
    for i,(c,d) in enumerate(zip(cy,cx)):
        for s in [-1,1]:
            den=x-s*d
            if den:
                a=(y-s*c)/den
                if lo<a<hi:events.setdefault(a,[]).append(dict(time=t+1,reference=i+1,sign=s))
roots=sorted(events);assert len(roots)==4
edges=[lo,*roots,hi]
def rank(t,a):return sum(abs(c-a*d)<=abs(oy[t]-a*ox[t]) for c,d in zip(cy,cx))
atoms=[]
for i,a in enumerate(edges):
    atoms.append(dict(label=f'P{i}',kind='point',left_exact=str(a),right_exact=str(a),rank_t1=rank(0,a),rank_t2=rank(1,a)))
    if i<len(edges)-1:
        b=edges[i+1];m=(a+b)/2;atoms.append(dict(label=f'I{i+1}',kind='interval',left_exact=str(a),right_exact=str(b),rank_t1=rank(0,m),rank_t2=rank(1,m)))
with (DATA/'complete_atoms.csv').open(encoding='utf-8-sig',newline='') as g:old=list(csv.DictReader(g))
assert len(old)==len(atoms)==11
for a,b in zip(atoms,old):
    for k in ['kind','left_exact','right_exact','rank_t1','rank_t2']:assert str(a[k])==str(b[k])
with (DATA/'labelled_complete_atoms.csv').open('w',encoding='utf-8',newline='') as g:
    w=csv.DictWriter(g,fieldnames=atoms[0].keys());w.writeheader();w.writerows(atoms)
count_audit=[]
for atom in atoms:
    l,r=Q(atom['left_exact']),Q(atom['right_exact']);a=l if l==r else (l+r)/2
    for t in range(2):
        ref_scores=[abs(c-a*d) for c,d in zip(cy,cx)];score=abs(oy[t]-a*ox[t])
        strict=sum(v<score for v in ref_scores);ties=sum(v==score for v in ref_scores)
        q=strict+ties;assert q==atom[f'rank_t{t+1}']
        if l<r:
            for test in [(3*l+r)/4,(l+3*r)/4]:assert rank(t,test)==q
        count_audit.append(dict(atom=atom['label'],kind=atom['kind'],time=t+1,representative_exact=str(a),reference_score_1=str(ref_scores[0]),reference_score_2=str(ref_scores[1]),online_score=str(score),strictly_smaller_count=strict,tie_count=ties,q_count=q,verified=True))
with (DATA/'q_count_audit.csv').open('w',encoding='utf-8',newline='') as g:
    w=csv.DictWriter(g,fieldnames=count_audit[0].keys());w.writeheader();w.writerows(count_audit)
eventrows=[]
for j,a in enumerate(roots):
    for e in events[a]:
        assert abs(oy[e['time']-1]-a*ox[e['time']-1])==abs(cy[e['reference']-1]-a*cx[e['reference']-1])
        eventrows.append(dict(root=f'a{j+1}',parameter_exact=str(a),parameter=float(a),absolute_residual=float(abs(oy[e['time']-1]-a*ox[e['time']-1])),**e))
with (DATA/'root_events.csv').open('w',encoding='utf-8',newline='') as g:
    w=csv.DictWriter(g,fieldnames=eventrows[0].keys());w.writeheader();w.writerows(eventrows)
audit_values=json.loads((DATA/'audited_cctm_root_wealth.json').read_text())
BLUE='#3C5488';TEAL='#008F83';ORANGE='#D55E00';PURPLE='#7E64A6';INK='#253345';GRAY='#7E8792';PALE='#EAF0F7'
width_mm=180;spec=panel(width_mm,height_mm=144,body_pt=9);apply_style(spec)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial'],'font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'savefig.bbox':None,'mathtext.cal':'STIXGeneral:italic'})
f=plt.figure(figsize=spec.figsize,facecolor='white');base=f.add_axes([0,0,1,1]);base.set(xlim=(0,180),ylim=(0,144));base.set_axis_off()
def txt(x,y,s,size=8,color=INK,ha='left',**kw):return base.text(x,y,s,fontsize=size,color=color,ha=ha,va='center',**kw)
def line(p,q,color=GRAY,lw=.7,**kw):base.plot([p[0],q[0]],[p[1],q[1]],color=color,lw=lw,**kw)
def clean(ax):
    ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=7,length=2,width=.6)
def section(x,w,color,fill,bottom=7.5):
    base.add_patch(FancyBboxPatch((x,bottom),w,119.5-bottom,boxstyle='round,pad=0,rounding_size=1.2',facecolor=fill,edgecolor=color,linewidth=.7,zorder=-2))
    line((x+2,111),(x+w-2,111),color,.5,alpha=.35)
section(4,27.5,'#C7D1DD','#F5F7FA',bottom=53.5)
section(34.5,99.5,'#9FB4CE','#FCFDFE')
section(137,40,'#CFC4E0','#FAF8FC')
txt(7,135,'Continuous uncertainty induces finite, complete rank geometry',11,fontweight='bold')
txt(7,128,'Unknown slope  →  residual equalities  →  complete historical-rank atoms',8,color=GRAY)
for x,label in [(4,'a'),(34.5,'b'),(137,'c')]:txt(x,122,label,10.5,fontweight='bold')

# Context stays compact; the geometry, rather than a workflow box, is the hero.
txt(6.5,115.3,'Unknown dynamics',7.2,fontweight='bold')
line((7,99),(30,99),BLUE,2)
base.add_patch(Polygon([[12,95],[12,103],[18.5,99]],fc=PALE,ec=BLUE,lw=.9))
base.add_patch(Polygon([[25,95],[25,103],[18.5,99]],fc=PALE,ec=BLUE,lw=.9))
line((18.5,99),(18.5,107),BLUE,.9);base.add_patch(Rectangle((14.5,107),8,3,fc=PALE,ec=BLUE,lw=.7))
base.add_patch(Circle((28,99),.9,color=TEAL))
txt(6.5,87,r'$x_t=a_0x_{t-1}+\varepsilon_t$',7.1)
txt(6.5,79,r'Unknown $a_0\in\mathcal{A}$',7,color=BLUE)
txt(6.5,68,'Fixed healthy',7,color=BLUE)
txt(6.5,63.5,'reference',7,color=BLUE)
txt(6.5,57.5,'+ online observations',6.8,color=TEAL)

# Both online times are drawn, so all four roots in the accumulated partition appear.
txt(39,115.3,'Residual crossings and complete atoms',8.3,fontweight='bold')
txt(43,106.3,r'$|r_t(a)|=|r_i^{\mathrm{ref}}(a)|$',8.3,color=BLUE)
txt(131,106.3,'N = 2; t = 2',6.5,color=GRAY,ha='right')
X0=43;XW=88;XM=X0+XW/2
ax=f.add_axes([X0/180,73/144,XW/180,28/144]);clean(ax);ax.set_facecolor('#FCFDFE')
a=np.linspace(float(lo),float(hi),1200)
for i,(c,d) in enumerate(zip(cy,cx)):ax.plot(a,np.abs(float(c)-a*float(d)),color=BLUE,lw=.85,ls=['--',':'][i],label=f'Ref {i+1}')
for j,(y,x) in enumerate(zip(oy,ox)):ax.plot(a,np.abs(float(y)-a*float(x)),color=TEAL,lw=1.1,ls=['-',(0,(4,1,1,1))][j],label=f'Online t={j+1}')
for j,r in enumerate(roots):
    e=events[r][0];v=float(abs(oy[e['time']-1]-r*ox[e['time']-1]))
    ax.plot(float(r),v,'o',color=ORANGE,ms=3,mec='white',mew=.3,zorder=5)
    ax.axvline(float(r),color=ORANGE,lw=.5,ls=':',alpha=.7)
ax.set(xlim=(float(lo),float(hi)),ylim=(0,1.12),xticks=[float(lo),0,float(hi)],xticklabels=['−0.999','0','0.999'],yticks=[0,.5,1])
ax.set_ylabel('Absolute residual',fontsize=7.2,labelpad=0)
handles=[Line2D([],[],color=BLUE,ls='--',lw=.85,label='Ref 1'),Line2D([],[],color=BLUE,ls=':',lw=.85,label='Ref 2'),Line2D([],[],color=TEAL,lw=1.1,label='Online t=1'),Line2D([],[],color=TEAL,ls=(0,(4,1,1,1)),lw=1.1,label='Online t=2')]
f.legend(handles=handles,loc='center',bbox_to_anchor=(XM/180,65/144),ncol=4,frameon=False,fontsize=7,columnspacing=.8,handlelength=1.7,handletextpad=.3)

# Exact metric partition and an aligned, equally spaced inventory of all histories.
def xmap(v):return X0+XW*float((v-lo)/(hi-lo))
txt(X0,57,'Complete partition',8.3,fontweight='bold')
for i,(l,r) in enumerate(zip(edges[:-1],edges[1:])):
    x=xmap(l);w=xmap(r)-x
    base.add_patch(Rectangle((x,47.5),w,4.8,fc=PALE,ec=BLUE,lw=.5))
    txt(x+w/2,49.9,f'I{i+1}',7,color=BLUE,ha='center')
for i,r in enumerate(edges):
    x=xmap(r);base.plot(x,46.1,marker='D',ms=3,mec=ORANGE,mfc='white',mew=.8,zorder=8)
    txt(x,42.2,f'P{i}',7,color=ORANGE,ha='center')
    line((x,47.4),(x,46.9),ORANGE,.65)
    if 0<i<len(edges)-1:
        # Interrupt projection guides at the legend and heading to keep type clear.
        for top,bottom in [(69.5,67.6),(62.4,60.0),(54.6,53.1)]:line((x,top),(x,bottom),ORANGE,.45,ls=':',alpha=.55)
txt(XM,36.5,'5 open intervals + 6 singletons (including endpoints)',7,color=GRAY,ha='center')
left=X0;cw=XW/11
table_text=[]
for j,r in enumerate(atoms):
    xc=left+(j+.5)*cw
    table_text.append(txt(xc,30.8,r['label'],7.5,color=ORANGE if r['kind']=='point' else BLUE,ha='center'))
    for k,y in [('rank_t1',24.4),('rank_t2',18.8)]:
        base.add_patch(Rectangle((left+j*cw,y-2.6),cw,5.2,fc='#FFF6ED' if r['kind']=='point' else 'white',ec='#D4DEE9',lw=.45))
        table_text.append(txt(xc,y,str(r[k]),8,ha='center'))
txt(X0-2,24.4,r'$q_1$',7.5,ha='right');txt(X0-2,18.8,r'$q_2$',7.5,ha='right')
txt(XM,11.6,'Reference scores ≤ online score; ties included',7,color=BLUE,ha='center')

# This is a separate audited counterexample. Its root coordinate is not available.
txt(139,115.3,'Why singletons matter',7.6,fontweight='bold')
txt(139,106.5,'Separate CCTM rank-tie example',6.5,color=PURPLE)
cxax=f.add_axes([147/180,74/144,27/180,26/144]);clean(cxax);cxax.set_facecolor('#FAF8FC')
rv=audit_values['root'];lv=audit_values['left'];rr=audit_values['right']
cxax.plot([-1,-.04],[lv,lv],color=PURPLE,lw=1.4);cxax.plot([.04,1],[rr,rr],color=PURPLE,lw=1.4)
cxax.plot(0,lv,marker='o',mfc='white',mec=PURPLE,ms=4,zorder=5)
cxax.plot(0,rv,'o',color=ORANGE,ms=4,zorder=6)
cxax.text(-.95,lv+.019,f'{lv:.4f}',fontsize=7,color=PURPLE)
cxax.text(.10,rv+.018,f'{rv:.4f}',fontsize=7,color=ORANGE)
cxax.set(xlim=(-1.1,1.1),ylim=(.47,.72),xticks=[-.7,0,.7],xticklabels=['Left',r'$a_\star$','Right'],yticks=[.5,.7])
cxax.set_ylabel('Wealth',fontsize=7.2,labelpad=2)
txt(160,67,'Symbolic local position',6.5,color=GRAY,ha='center')
txt(139,53,'Intervals alone miss this dip',6.8,color=INK)
txt(139,44,r'$W_t(a_\star)=0.5000$',8,color=ORANGE)
txt(139,31,'Keep open intervals,',7.1)
txt(139,25,'root points and endpoints.',7.1)
txt(139,12,r'At most $4Nt+3$ atoms',7.4,color=BLUE)

f.canvas.draw();sizes,viol=landed_report(f,spec);assert viol==0
renderer=f.canvas.get_renderer()
assert min(t.get_fontsize() for t in table_text)==7.5
table_overlaps=[]
for i,t in enumerate(table_text):
    for u in table_text[i+1:]:
        if t.get_window_extent(renderer).overlaps(u.get_window_extent(renderer)):table_overlaps.append([t.get_text(),u.get_text()])
assert not table_overlaps
assert not any(isinstance(p,matplotlib.patches.FancyArrowPatch) for p in base.patches)
f.savefig(OUT/'Figure1_FINAL.svg',bbox_inches=None)
f.savefig(OUT/'Figure1_FINAL.pdf',bbox_inches=None)
f.savefig(OUT/'Figure1_FINAL.png',dpi=600,bbox_inches=None)
f.savefig(OUT/'Figure1_FINAL.tiff',dpi=600,bbox_inches=None,pil_kwargs={'compression':'tiff_lzw'})
im=Image.open(OUT/'Figure1_FINAL.png');im.thumbnail((1700,1500));im.save(OUT/'Figure1_FINAL_preview.png');plt.close(f)
lf=plt.figure(figsize=(88/25.4,12/25.4));lf.legend(handles=handles,loc='center',ncol=4,frameon=False,fontsize=7)
for ext in ['pdf','svg','png']:lf.savefig(OUT/f'Figure1_curve_legend.{ext}',dpi=300,bbox_inches=None)
plt.close(lf)
audit={'skills':['nature-figure','nature-figure-production specification/style'],'backend':'python','geometry_mm':[180,144],'archetype':'geometry-led connected composite, without equal ABCD boxes','text_sizes':sizes,'font_violations':viol,'root_count':4,'interval_count':5,'singleton_count':6,'all_atoms':11,'source_validation':'All 11 atoms and both historical ranks exactly match the frozen example.','root_counterexample':'Separate audited CCTM case; numerical root coordinate not supplied and not inferred.','boundaries':['Two online residual curves are shown, so all four accumulated roots correspond to visible comparisons.','No current-rank colour encoding is used; both historical ranks are printed for every atom.','Interval strip uses true parameter distances; history table columns are equally spaced for legibility.','P0 and P5 are domain endpoints; P1–P4 are interior roots.','The two neighbouring open intervals have equal wealth in the independent audit.','No new detection experiments; no streaming/backend/readout material duplicated from Figure2.'],'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob('*')}}
audit['layout_change']='Three light section plates and thin borders; central parameter/table width 88 mm, enlarged from 79 mm (+11.39%).'
audit['count_definition']='q_t(a) = sum_i 1{|r_i_ref(a)| <= |r_t(a)|}, in {0,1,2} for N=2; not a one-based insertion rank.'
audit['q_audit_rows']=len(count_audit)
audit['final_cleanup']={'residual_connector_arrows':0,'panel_labels':['a','b','c'],'history_header_font_pt':7.5,'history_value_font_pt':8,'history_table_text_bbox_overlaps':0,'left_context_card':'Compacted to system, model, unknown scalar, fixed reference and online observations.'}
(H/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print('FIGURE1_FINAL_COMPLETE',OUT)
