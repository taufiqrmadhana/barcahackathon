import sys; sys.path.insert(0,'src_s2')
import numpy as np, data, s2, shape
inp=data.load(); import os,json; p=s2.Params.load(os.environ.get('PAR','src_s2/params_s7b.json'),**json.loads(os.environ.get('OVR','{}'))); inp['shape']=shape.get_proba(inp,p)
lab,d=s2.predict(inp,p,debug=True)
a,b=int(sys.argv[1]),int(sys.argv[2]); st=int(sys.argv[3]) if len(sys.argv)>3 else 10
from scipy.ndimage import uniform_filter1d
pd_=uniform_filter1d(inp['shape'][:,2],75)
pv=inp['pv']; sp=np.nanmean(pv,1)
for t in range(a,b,st):
    print(t,'lab',s2.LABELS[lab[t]],'own',d['own'][t],'dead',int(d['dead'][t]),'seen',int(d['seen'][t]),'out',int(d['out'][t]),'still',int(d['still'][t]),'sm %.2f'%d['sm'][t],'psp %.2f'%sp[t],'pdead %.2f'%pd_[t],'c',d['con']['c'][t],'dH %.1f dA %.1f'%(d['con']['dH'][t],d['con']['dA'][t]), 'ball %.1f %.1f'%(inp['bx'][t],inp['by'][t]))
