"""Refined Figure 2: local inheritance, rank interface, historical replay and guarantee transfer."""
from pathlib import Path
from fractions import Fraction as F
import sys,json,csv,hashlib
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch,FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from PIL import Image

H=Path(__file__).resolve().parent;DATA=H/'source_data';OUT=H/'outputs';OUT.mkdir(exist_ok=True)
sys.path[:0]=[str(H/'assets'),str(H/'reference_engine')]
from nature_style import panel,apply_style,landed_report
from generic_profile import stream,batch,prm_backend,cctm_backend
from atom_exact import rank_at

# Replay the exact saved input. No new random draws or simulation trajectories.
case=json.loads((DATA/'historical_random_48_input.json').read_text());values=[[F(v) for v in a] for a in case['vectors']]
N=len(values[0]);T=len(values[2]);assert N==3 and T==4
checks=list(csv.DictReader((DATA/'generic_checks.csv').open(encoding='utf-8-sig')))
assert len(checks)==128 and len(set(r['case'] for r in checks))==64 and all(r['pass_flag']=='PASS' for r in checks)
paths={};replay=[]
for backend in [prm_backend(),cctm_backend()]:
    s=stream(*values,backend=backend);b=batch(*values,backend=backend)
    name='PRM' if backend.name.startswith('EP') else 'CCTM'
    errors=[abs(x-y) for x,y in zip(s['profile'],b['profile'])]
    assert max(errors)<F(1,10**48)
    if name=='PRM':assert s['profile']==b['profile']
    paths[name]={'stream':np.array([float(x) for x in s['profile']]),'batch':np.array([float(x) for x in b['profile']])}
    for t,(x,y,e,k) in enumerate(zip(s['profile'],b['profile'],errors,s['atoms'])):replay.append(dict(backend=name,t=t,stream_wealth=str(x),batch_wealth=str(y),absolute_error=str(e),atoms=k))
with (DATA/'stream_batch_replay.csv').open('w',encoding='utf-8',newline='') as g:
    w=csv.DictWriter(g,fieldnames=replay[0].keys());w.writeheader();w.writerows(replay)
# Confirm the notation bridge against the actual archived reference function.
for a in [F(-999,1000),F(-1,2),F(0),F(1,2),F(999,1000)]:
    for y,x in zip(values[2],values[3]):
        q=sum(abs(c-a*d)<=abs(y-a*x) for c,d in zip(values[0],values[1]))
        assert rank_at(values[0],values[1],y,x,a)==q+1

BLUE='#3C5488';TEAL='#008F83';PURPLE='#7E64A6';ORANGE='#D55E00';INK='#253345';GRAY='#7E8792';PALE='#EAF0F7'
width_mm=180;spec=panel(width_mm,height_mm=167,body_pt=9);apply_style(spec)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial'],'font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'savefig.bbox':None,'mathtext.cal':'STIXGeneral:italic'})
f=plt.figure(figsize=spec.figsize,facecolor='white');base=f.add_axes([0,0,1,1]);base.set(xlim=(0,180),ylim=(0,167));base.set_axis_off()
arrow_lengths=[]
def tx(x,y,s,size=8,color=INK,ha='left',**kw):return base.text(x,y,s,fontsize=size,color=color,va='center',ha=ha,**kw)
def line(p,q,color=GRAY,lw=.8,**kw):base.plot([p[0],q[0]],[p[1],q[1]],color=color,lw=lw,**kw)
def arr(p,q,color=BLUE):
    length=float(np.hypot(q[0]-p[0],q[1]-p[1]));assert length>=3
    arrow_lengths.append(length);base.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=8,linewidth=.8,color=color,shrinkA=0,shrinkB=0,zorder=3))
def card(x,y,w,h,fill,edge):base.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=1.2',fc=fill,ec=edge,lw=.7,zorder=-4))
def node(x,y,w,h,fill=PALE,edge=BLUE):base.add_patch(Rectangle((x,y),w,h,fc=fill,ec=edge,lw=.7,zorder=2))
def clean(ax):ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=7,length=2,width=.6)
tx(7,159,'Exact streaming updates through inherited rank states',11,fontweight='bold')
tx(7,151,'Refine atoms · inherit shared states · update once with the current rank',8,color=GRAY)
card(4,82,108,61,'#FCFDFE','#9FB4CE');card(117,82,59,61,'#F5F7FA','#C7D1DD')
card(4,8,81,64,'#FCFDFE','#C7D1DD');card(90,8,86,64,'#F6FAF9','#B4CFC7')
for x,y,l,title in [(7,138,'a','Refine, inherit and update'),(120,138,'b','A shared rank interface'),(7,67,'c','Stream–batch equivalence'),(93,67,'d','Transfer the alarm guarantee')]:
    tx(x,y,l,10.5,fontweight='bold');tx(x+5,y,title,8.5,fontweight='bold')

# a: A local split inside an existing open parent interval.
node(15,123,87,8)
tx(58.5,127,r'Parent atom $C$ : shared state $S_{t-1}(C)$',8.5,ha='center')
base.plot(58.5,132.6,marker='D',ms=3.6,mfc=ORANGE,mec='white',mew=.3,zorder=4)
tx(63,133.2,r'New equality event $a_\star$',6.8,color=ORANGE)
children=[(10.5,25,r'C_L',BLUE),(45.5,25,r'\{a_\star\}',ORANGE),(80.5,25,r'C_R',BLUE)]
for x,w,label,col in children:
    xc=x+w/2
    arr((58.5,122.8),(xc,114.5))
    node(x,106,w,8,fill='white' if col==ORANGE else PALE,edge=col)
    tx(xc,112,r'$'+label+r'$',8,color=col,ha='center')
    tx(xc,108.1,r'$S_{t-1}(C)$',7.4,ha='center')
    arr((xc,105.5),(xc,98),TEAL)
    tx(xc+2,101.8,r'$R_t('+label+r')$',6.8,color=TEAL)
    node(x,89,w,8,fill='#E8F4F1',edge=TEAL)
    tx(xc,93,r'$S_t('+label+r')$',8,color=TEAL,ha='center')
tx(58,85.4,'Shared past; one current-rank update per child',7,color=GRAY,ha='center')

# b: Rank conventions agree with Figure 1's reference count q.
node(123,126,47,7,fill=PALE)
tx(146.5,129.5,r'$R_t(C)=1+q_t(C)$',8.5,ha='center')
arr((146.5,125.5),(146.5,121.6))
tx(146.5,118.5,r'$S_t=\Phi_t(S_{t-1},R_t)$',9,ha='center')
for x,name,state,col in [(121,'PRM',r'$(M,\lambda,A,H)$',TEAL),(149,'CCTM',r'$(W,\eta,A)$',PURPLE)]:
    arr((146.5,115),(x+11.5,110.5),col)
    node(x,99,23,11,fill='white',edge=col)
    tx(x+11.5,106.7,name,8.5,color=col,ha='center');tx(x+11.5,102.1,state,6.8,ha='center')
    arr((x+11.5,98.5),(146.5,94.7),col)
tx(146.5,104.5,'or',6.5,color=GRAY,ha='center')
tx(146.5,91.5,r'$W_t(C)=\Psi_t(S_t(C))$',8.5,ha='center')
tx(146.5,84.8,'Alternative finite-dimensional states',6.6,color=GRAY,ha='center')

# c: Same fixed historical input, not a detection-power experiment.
tx(44.5,60,r'$G_t^{\mathrm{stream}}=G_t^{\mathrm{batch}}=\inf_a W_t(a)$',8.6,ha='center')
tx(44.5,55,'Fixed input replay (N = 3, T = 4)',7,color=GRAY,ha='center')
ax=f.add_axes([15/180,27/167,66/180,25/167]);clean(ax);ax.set_facecolor('#FCFDFE')
t=np.arange(T+1)
for name,col in [('PRM',TEAL),('CCTM',PURPLE)]:
    ax.plot(t,paths[name]['stream'],color=col,lw=1.3)
    ax.plot(t,paths[name]['batch'],color=col,ls='none',marker='o',mfc='white',ms=4,mew=.85)
ax.axhline(1,color='#B7C1CA',ls=':',lw=.6)
ax.set(xlim=(-.15,4.15),ylim=(0,1.1),xticks=range(5),yticks=[0,.5,1],xlabel='Online step',ylabel='Minimum wealth')
ax.set_xlabel('Online step',fontsize=7);ax.set_ylabel('Minimum wealth',fontsize=7,labelpad=2)
handles=[Line2D([],[],color=col,lw=1.3,label=f'{name}: stream') for name,col in [('PRM',TEAL),('CCTM',PURPLE)]]+[Line2D([],[],ls='none',marker='o',mfc='white',mec=col,ms=4,label=f'{name}: batch') for name,col in [('PRM',TEAL),('CCTM',PURPLE)]]
f.legend(handles=handles,loc='center',bbox_to_anchor=(45/180,14/167),ncol=2,frameon=False,fontsize=6.7,handlelength=1.5,columnspacing=1,handletextpad=.35)

# d: Event areas are schematic; no probability is encoded by rectangle area.
tx(133,58.5,r'$G_t\leq W_t(a_0),\quad a_0\in\mathcal{A}$',10,ha='center')
node(98,32,70,21,fill='#EEF3F8',edge=BLUE)
tx(133,47.5,r'$\{\sup_t W_t(a_0)\geq c\}$',9,color=BLUE,ha='center')
node(108,34,50,8,fill='#E2F0EB',edge=TEAL)
tx(133,38,r'$\{\sup_t G_t\geq c\}$',9,color=TEAL,ha='center')
tx(133,25,r'$\Pr(\sup_t G_t\geq c)\leq\alpha$',10,color=TEAL,ha='center')
tx(133,15.5,'Under the point-parameter guarantee',7,color=GRAY,ha='center')
tx(7,3,'Audit archive: 64 fixed inputs × 2 backends = 128 passed checks',7,color=GRAY)

f.canvas.draw();sizes,viol=landed_report(f,spec);assert viol==0
renderer=f.canvas.get_renderer();overlaps=[]
for i,ta in enumerate(base.texts):
    for tb in base.texts[i+1:]:
        if ta.get_window_extent(renderer).overlaps(tb.get_window_extent(renderer)):overlaps.append([ta.get_text(),tb.get_text()])
assert not overlaps,overlaps
# The shared input/readout labels and state nodes are intentionally schematic.
assert min(arrow_lengths)>=3 and len(arrow_lengths)==11
f.savefig(OUT/'Figure2_FINAL.svg',bbox_inches=None);f.savefig(OUT/'Figure2_FINAL.pdf',bbox_inches=None)
f.savefig(OUT/'Figure2_FINAL.png',dpi=600,bbox_inches=None);f.savefig(OUT/'Figure2_FINAL.tiff',dpi=600,bbox_inches=None,pil_kwargs={'compression':'tiff_lzw'})
for letter,(x,y,w,h) in zip('abcd',[(4,82,108,61),(117,82,59,61),(4,8,81,64),(90,8,86,64)]):
    crop=Bbox.from_bounds(x/25.4,y/25.4,w/25.4,h/25.4)
    for ext in ['svg','pdf','png']:f.savefig(OUT/f'Figure2_{letter}.{ext}',dpi=600,bbox_inches=crop,pad_inches=0)
im=Image.open(OUT/'Figure2_FINAL.png');im.thumbnail((1650,1600));im.save(OUT/'Figure2_FINAL_preview.png');plt.close(f)
lf=plt.figure(figsize=(88/25.4,14/25.4));lf.legend(handles=handles,loc='center',ncol=2,frameon=False,fontsize=7)
for ext in ['pdf','svg','png']:lf.savefig(OUT/f'Figure2_legend.{ext}',dpi=300,bbox_inches=None)
plt.close(lf)
audit={'skill':'nature-figure-production','geometry_mm':[180,167],'font_violations':viol,'text_sizes':sizes,'arrow_count':len(arrow_lengths),'shortest_arrow_mm':min(arrow_lengths),'rank_bridge':'R = 1 + q, calibration-first ties (<=), verified against rank_at on saved input.','historical_case':case['case'],'existing_audit_inputs':64,'existing_audit_backend_checks':128,'replay_max_abs_errors':{name:max(float(r['absolute_error']) for r in replay if r['backend']==name) for name in paths},'scope':'Schematic a,b,d; replay of one archived fixed input c. The 128 archived PASS rows are checked, not newly rerun.','guarantees':'Point-parameter event domination; marginal PRM and budgeted CCTM conditions are distinct. G is not asserted to be a martingale/e-process.','semantic_colors':{'geometry':BLUE,'PRM/update':TEAL,'CCTM':PURPLE,'root':ORANGE},'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob('*') if p.is_file()}}
audit['schematic_text_bbox_overlaps']=overlaps
audit['final_wording']='Shared rank interface; fixed input replay; detector-specific guarantee notes in caption; shortened archival audit footer.'
(H/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print('FIGURE2_FINAL_COMPLETE',OUT)
