#!/usr/bin/env python3
import argparse, csv, hashlib, json, os, platform, re, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.benchmark_support import METHODS, ROBOTS, SCENARIOS
import torch

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def git():
    try: return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError: return None

def prepared_world(source, scene_seed):
    content=source.read_text(); content,n=re.subn(r'(?m)(\bcontroller\s+)"frl_agent"',r'\1"bench_agent"',content)
    if n!=3: raise ValueError(f'{source}: expected 3 frl_agent controller fields, got {n}')
    content,n=re.subn(r'(?m)(\bcontroller\s+)"superviser"',r'\1"bench_supervisor"',content)
    if n!=1: raise ValueError(f'{source}: expected 1 superviser controller field, got {n}')
    # Set Webots physics seed as well as the Python scenario seed.
    if re.search(r'(?m)^\s*randomSeed\s+-?\d+',content):
        content,n=re.subn(r'(?m)^(\s*randomSeed\s+)-?\d+',lambda m:m[1]+str(scene_seed),content)
        if n!=1: raise ValueError('Ambiguous randomSeed fields')
    else:
        content,n=re.subn(r'\bWorldInfo\s*\{',lambda m:m[0]+f'\n  randomSeed {scene_seed}',content)
        if n!=1: raise ValueError('Expected a single WorldInfo')
    target=source.with_name(source.stem+'_benchmark.wbt')
    if target.exists(): raise FileExistsError(f'Generated world already exists (possibly left by a failed run): {target}')
    target.write_text(content); return target

def launch(cfg,world,webots,timeout):
    folder=Path(cfg['phase_dir']); folder.mkdir(parents=True,exist_ok=False)
    config=folder/'config.json'; config.write_text(json.dumps(cfg,indent=2))
    env=os.environ.copy(); env['PROJECT_BENCH_CONFIG']=str(config)
    print(f"{cfg['method']} seed={cfg['seed']} {cfg['phase']} {cfg['scenario']}",flush=True)
    generated=prepared_world(world,cfg['scene_seed'])
    try:
        with (folder/'webots.log').open('w') as log:
            subprocess.run([str(webots),'--batch','--mode=fast','--stdout','--stderr',str(generated)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,check=True)
    finally:
        generated.unlink(missing_ok=True)
    if not all((folder/f'{n}.done').is_file() for n in (*ROBOTS,'supervisor')): raise RuntimeError(f'Incomplete phase: inspect {folder}/webots.log')
    return folder

def combine(folders,target,training,budget,episodes):
    rows=[]; keys=set()
    for folder in folders:
        for robot in ROBOTS:
            with (folder/f'{robot}.csv').open() as f:
                data=list(csv.DictReader(f))
            expected=budget if training else episodes
            if len(data)!=expected or any(r['robot']!=robot for r in data): raise ValueError(f'Incomplete or wrong robot: {folder}/{robot}.csv')
            idx='interaction' if training else 'episode'
            if [int(r[idx]) for r in data]!=list(range(1,expected+1)): raise ValueError('Invalid sequence')
            for r in data:
                key=(r['robot'],r[idx]) if training else (r['scenario'],r['robot'],r['scene_seed'],r[idx])
                if key in keys: raise ValueError('Duplicate row')
                keys.add(key); rows.append(r)
    tmp=target.with_suffix('.tmp');
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    tmp.replace(target)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); p.add_argument('--methods',nargs='+',choices=METHODS); p.add_argument('--seeds',nargs='+',type=int); p.add_argument('--budget',type=int); p.add_argument('--eval-episodes',type=int); p.add_argument('--federation-interval',type=int); p.add_argument('--world',type=Path,default=ROOT/'worlds/FederatedRL.wbt'); p.add_argument('--unseen-world',type=Path); p.add_argument('--webots',type=Path,default=Path('/Applications/Webots.app/Contents/MacOS/webots')); p.add_argument('--timeout',type=int,default=86400); p.add_argument('--contact-height',type=float,default=.03); p.add_argument('--contacts-validated',action='store_true',help='Only after visual calibration on both worlds'); p.add_argument('--min-success-distance',type=float,default=1.0); a=p.parse_args()
    methods=a.methods or (['ddqn_fedavg'] if a.smoke else list(METHODS)); seeds=a.seeds or ([11] if a.smoke else [11,22,33,44,55]); budget=a.budget or (1000 if a.smoke else 30000); episodes=a.eval_episodes or (2 if a.smoke else 30); interval=a.federation_interval or (1000 if a.smoke else 1000)
    if budget<interval or interval<=500 or budget%interval or episodes<1: p.error('Budget must be a multiple of federation interval; interval > 500 (warmup).')
    if any(m.endswith('_fault') for m in methods) and budget<3*interval: p.error('Fault methods require at least three federation rounds.')
    if not a.smoke and (not a.contacts_validated or not a.unseen_world): p.error('Confirmatory run requires calibrated contacts and an independent --unseen-world. Start with --smoke.')
    if not a.smoke and (budget!=30000 or episodes!=30 or not set(seeds)<=set([11,22,33,44,55])): p.error('Notebook defaults require 30000 interactions, 30 episodes and preregistered seeds.')
    if not a.webots.is_file(): p.error(f'Webots executable not found: {a.webots}')
    source=a.world.resolve(); unseen=a.unseen_world.resolve() if a.unseen_world else None
    if unseen and sha(unseen)==sha(source): p.error('Unseen world must differ from training world.')
    if source.parent!=ROOT/'worlds' or (unseen and unseen.parent!=ROOT/'worlds'): p.error('Both world files must be in Project_lab/worlds')
    worlds={'seen':source}
    if unseen: worlds['unseen']=unseen
    base=ROOT/('research_smoke' if a.smoke else 'research_runs'); base.mkdir(exist_ok=True)
    for method in methods:
        for seed in seeds:
            run=base/method/str(seed); run.mkdir(parents=True,exist_ok=False)
            settings=dict(method=method,seed=seed,budget=budget,eval_episodes=episodes,federation_interval=interval,fault_round=3,fault_scale=10.,noise_std=.02,shift_seconds=30.,episode_seconds=60.,contact_height=a.contact_height,calibration=a.smoke,min_success_distance=a.min_success_distance,run_dir=str(run),contacts_validated=a.contacts_validated)
            sources=[ROOT/'src'/n for n in ('dqn.py','environment.py','trainer.py','reward.py','replay_buffer.py','benchmark_support.py')]+[ROOT/'controllers/bench_agent/bench_agent.py',ROOT/'controllers/bench_supervisor/bench_supervisor.py',Path(__file__)]
            meta={**settings,'git_commit':git(),'python':sys.version,'torch':torch.__version__,'platform':platform.platform(),'world_hash':sha(source),'unseen_world_hash':sha(unseen) if unseen else None,'source_hashes':{str(x.relative_to(ROOT)):sha(x) for x in sources},'webots_executable':str(a.webots),'world_header':source.read_text().splitlines()[0],'status':'running'}
            (run/'metadata.json').write_text(json.dumps(meta,indent=2))
            if method!='reactive':
                folder=launch({**settings,'phase':'train','scenario':'seen','scene_seed':seed,'phase_dir':str(run/'train_raw')},worlds['seen'],a.webots,a.timeout)
                combine([folder],run/'train.csv',True,budget,episodes)
            folders=[]
            for j,scenario in enumerate(SCENARIOS):
                if scenario=='unseen' and not unseen: continue
                folder=launch({**settings,'phase':'eval','scenario':scenario,'scene_seed':100000+seed*100+(50 if scenario=='unseen' else 0),'phase_dir':str(run/f'eval_{scenario}_raw')},worlds.get(scenario,worlds['seen']),a.webots,a.timeout); folders.append(folder)
            combine(folders,run/('eval_candidate.csv' if a.smoke else 'eval.csv'),False,budget,episodes)
            meta['status']='smoke_complete' if a.smoke else 'complete'; (run/'metadata.json').write_text(json.dumps(meta,indent=2))
    print('Completed:',base)
if __name__=='__main__': main()
