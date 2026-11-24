import csv, json, math, random, sys
from pathlib import Path
from controller import Supervisor
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.benchmark_support import config, MOTION_STEPS, ROBOTS

c=config(); s=Supervisor(); dt=int(s.getBasicTimeStep()); rng=random.Random(c['scene_seed']); out=Path(c['phase_dir'])
training=c['phase']=='train'; robots={}; obstacles=[]
for i,name in enumerate(ROBOTS,1):
    n=s.getFromDef(f'ROBOT_{i}')
    if n is None: raise RuntimeError(f'Missing ROBOT_{i}')
    tr=n.getField('translation'); rot=n.getField('rotation'); data=n.getField('customData')
    if any(x is None for x in (tr,rot,data)): raise RuntimeError(f'Missing robot fields: {name}')
    p=list(tr.getSFVec3f()); robots[name]=dict(node=n,tr=tr,rot=rot,data=data,initial=p,orientation=list(rot.getSFRotation()),last=p.copy(),anchor=p.copy(),motion=0.,distance=0.,start=0.,episodes=0,active=True,hit=False)
for i in range(1,101):
    n=s.getFromDef(f'OBSTACLE_{i}')
    if n is None: continue
    tr=n.getField('translation'); obstacles.append(dict(tr=tr,origin=list(tr.getSFVec3f()),axis=(i-1)%2,phase=rng.uniform(0,2*math.pi),frequency=.1))
if not obstacles: raise RuntimeError('No OBSTACLE_i found')

def tick():
    if s.step(dt)==-1: raise SystemExit(0)

trial_start=0.; trial=1
def move():
    t=s.getTime() if training else max(0.,s.getTime()-trial_start)
    for o in obstacles:
        # Deterministic continuous frequency change at fixed simulated time, not dependent on policy.
        t0=c['shift_seconds']; freq=o['frequency']; new=.2 if c['scenario']=='dynamic_shift' else freq
        phase=2*math.pi*(freq*min(t,t0)+new*max(0,t-t0))+o['phase']
        p=o['origin'].copy(); p[o['axis']]+=.3*math.sin(phase); o['tr'].setSFVec3f(p)

def reset(r):
    r['tr'].setSFVec3f(r['initial']); r['rot'].setSFRotation(r['orientation']); r['node'].resetPhysics()
    r['last']=r['initial'].copy(); r['anchor']=r['initial'].copy(); r['distance']=0.; r['hit']=False
    r['start']=r['motion']=s.getTime()+dt/1000

with (out/'contacts.csv').open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['cycle','time','robot','x','y','z','node_id','candidate'])
    move(); tick(); now=s.getTime(); trial_start=now
    for r in robots.values(): r['start']=r['motion']=now
    cycle=0
    while True:
        cycle+=1
        for r in robots.values(): r['hit']=False
        for _ in range(MOTION_STEPS+1):
            move(); tick(); now=s.getTime()
            for name,r in robots.items():
                if not r['active']: continue
                p=list(r['tr'].getSFVec3f()); r['distance']+=math.dist(p[:2],r['last'][:2]); r['last']=p
                if math.dist(p[:2],r['anchor'][:2])>=.05: r['anchor']=p.copy(); r['motion']=now
                for cp in r['node'].getContactPoints(True):
                    x,y,z=cp.getPoint(); candidate=z>c['contact_height']
                    if candidate: r['hit']=True
                    # all contacts in calibration mode, otherwise candidate contacts only
                    if c['calibration'] or candidate: w.writerow([cycle,now,name,x,y,z,cp.getNodeId(),int(candidate)])
        now=s.getTime(); reasons={}
        for name,r in robots.items():
            reason=''
            if r['active']:
                if r['hit']: reason='collision'
                elif now-r['motion']>=8.: reason='stuck'
                elif now-r['start']+2*dt/1000>=c['episode_seconds']: reason='timeout'
                if reason: r['episodes']+=1
            reasons[name]=reason
        new_trial=not training and all(r['episodes']>=trial for r in robots.values())
        finish=cycle==c['budget'] if training else new_trial and trial==c['eval_episodes']
        for name,r in robots.items():
            r['data'].setSFString(json.dumps(dict(cycle=cycle,reason=reasons[name],seconds=now-r['start'],distance_m=r['distance'],finish=finish,new_trial=new_trial,episode_scene_seed=c['scene_seed']+trial-1,next_scene_seed=c['scene_seed']+trial)))
        tick() # outcome delivered to agents, no reset before their successor sample
        for name,r in robots.items():
            if training and reasons[name]: reset(r)
            elif not training and reasons[name]: r['active']=False
        if new_trial and not finish:
            trial+=1
            phase_rng=random.Random(c['scene_seed']+trial-1)
            for o in obstacles: o['phase']=phase_rng.uniform(0,2*math.pi)
            for r in robots.values(): reset(r); r['active']=True
            trial_start=s.getTime()+dt/1000
            move()
        tick() # reset settles; agents clear temporal history
        f.flush()
        if finish:
            (out/'supervisor.done').write_text('ok'); tick(); s.simulationQuit(0); break
