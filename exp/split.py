import sys, json; sys.path.insert(0,'src_s2'); sys.path.insert(0,'C:/Users/neo/Koding/barcahackathon/personal/loop')
import numpy as np, data, s2, shape, loopmetrics as lm, possession_metrics as pm
inp=data.load(); base=s2.Params.load('src_s2/params_s7b.json'); inp['shape']=shape.get_proba(inp,base)
over=json.loads(sys.argv[1]) if len(sys.argv)>1 else {}
sg=float(sys.argv[2]) if len(sys.argv)>2 else 0.2
p=s2.Params({**base,**over}); tsets=lm.truth_sets()
def noisy(sd):
    r=np.random.default_rng(sd); return inp['bx']+r.normal(0,sg,inp['bx'].shape), inp['by']+r.normal(0,sg,inp['by'].shape)
def run(mode,sd):
    nb=noisy(sd); cb=(inp['bx'],inp['by'])
    bc = nb if mode in('contacts','both') else cb
    bd = nb if mode in('dead','both') else cb
    info={}
    x,y,valid,seen,vx,vy=s2.clean_ball(*bc,p,info); fl=s2.noise_floors(info['sigma'],p) if p.get('noise_q',0)>0 else None
    con=s2.contacts(inp,x,y,vx,vy,p,fl)
    cr=s2.contact_runs(con['c'],con['kick'],p); own=s2.shift(s2.owner_from_runs(cr,inp['n'],p),p.lag)
    own=s2.shape_gaps(own,valid,inp['shape'],cr,p)
    x2,y2,v2,s_,vx2,vy2=s2.clean_ball(*bd,p)
    dead,_=s2.dead_mask(inp,x2,y2,v2,s_,vx2,vy2,con['kick_raw'],p)
    lab=s2.assemble(own,dead,cr,p,inp['shape']); pred=pm.label_codes(s2.LABELS[lab])
    tabs=[lm.match_table(t,pred) for _,t in tsets]
    return pm.evaluate(tsets[0][1],pred,bootstrap_n=0)['spell_f1_mean'], lm.pooled_f1(tabs)
for mode in ('contacts','dead','both'):
    r=[run(mode,sd) for sd in (0,1)]
    print(mode,'public %.3f pooled %.3f'%tuple(np.mean(r,0)))
