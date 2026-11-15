import sys,os,json; sys.path.insert(0,'src_s2')
import numpy as np, data, s2, shape
inp=data.load(); p=s2.Params.load(os.environ.get('PAR','src_s2/params_s7b.json'),**json.loads(os.environ.get('OVR','{}'))); inp['shape']=shape.get_proba(inp,p)
lab,d=s2.predict(inp,p,debug=True)
a,b,st=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])
c=d['con']
x,y,valid,seen,vx,vy=s2.clean_ball(inp['bx'],inp['by'],p)
for t in range(a,b,st):
    print(t,'lab',s2.LABELS[lab[t]],'c',c['c'][t],'kick',int(c['kick'][t]),'dv %.1f'%c['dv'][t],'carry',int(c['carry'][t]),'dH %.1f dA %.1f'%(c['dH'][t],c['dA'][t]),'spd %.1f'%np.hypot(vx[t],vy[t]),'ball %.1f %.1f'%(x[t],y[t]),'near',c['team'][t])
print([(int(a_),int(b_),t_) for a_,b_,t_ in d['cruns'] if a-50<a_<b+50])
