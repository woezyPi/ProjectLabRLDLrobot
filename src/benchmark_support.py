import json, os, time
from pathlib import Path
import torch

ROBOTS = ('robot_1','robot_2','robot_3')
METHODS = ('reactive','dqn_local','ddqn_local','dueling_ddqn_local','ddqn_fedavg','ddqn_median','ddqn_fedavg_fault','ddqn_median_fault')
SCENARIOS = ('seen','unseen','dynamic_shift','sensor_noise')
MOTION_STEPS = 5

def config():
    return json.loads(Path(os.environ['PROJECT_BENCH_CONFIG']).read_text())

def atomic_save(payload, path):
    path = Path(path); tmp = path.with_suffix('.tmp')
    torch.save(payload,tmp); os.replace(tmp,path)

def wait(paths, timeout=180):
    until=time.monotonic()+timeout
    while not all(p.is_file() for p in paths):
        if time.monotonic()>until:
            raise TimeoutError(f'Missing: {[str(p) for p in paths if not p.is_file()]}')
        time.sleep(.02)

def aggregate(models,counts,median=False):
    if len(models)!=3 or len(counts)!=3 or any(n<0 for n in counts) or sum(counts)<=0:
        raise ValueError('Invalid federation contributions')
    if any(set(m)!=set(models[0]) for m in models): raise ValueError('Incompatible weights')
    out={}
    for k in models[0]:
        stack=torch.stack([m[k] for m in models]);
        if not torch.isfinite(stack).all(): raise ValueError('Nonfinite weights')
        out[k]=stack.median(0).values if median else sum(m[k]*(n/sum(counts)) for m,n in zip(models,counts))
    return out

class Federation:
    def __init__(self, cfg, name, trainer):
        self.cfg=cfg; self.name=name; self.root=Path(cfg['phase_dir'])/'federation'
        self.round=0; self.prev={k:v.detach().cpu().clone() for k,v in trainer.online.state_dict().items()}; self.last_updates=0
    def sync(self, trainer):
        self.round+=1; folder=self.root/f'round_{self.round:05d}'; folder.mkdir(parents=True,exist_ok=True)
        weights={k:v.detach().cpu().clone() for k,v in trainer.online.state_dict().items()}
        faulty=self.cfg['method'].endswith('_fault') and self.name=='robot_3' and self.round>=self.cfg['fault_round']
        if faulty:
            # A single fixed, finite, amplified client update; not a general attack guarantee.
            weights={k:self.prev[k]+self.cfg['fault_scale']*(v-self.prev[k]) for k,v in weights.items()}
        count=trainer.training_steps-self.last_updates
        atomic_save({'weights':weights,'count':count,'faulty':faulty},folder/f'{self.name}.pt')
        paths=[folder/f'{n}.pt' for n in ROBOTS]; glob=folder/'global.pt'
        if self.name==ROBOTS[0]:
            wait(paths); clients=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
            weights=aggregate([c['weights'] for c in clients],[c['count'] for c in clients],median='median' in self.cfg['method'])
            atomic_save(weights,glob)
        wait([glob]); weights=torch.load(glob,map_location='cpu',weights_only=True)
        trainer.online.load_state_dict(weights); trainer.target.load_state_dict(weights); trainer.optimizer.state.clear()
        self.prev={k:v.clone() for k,v in weights.items()}; self.last_updates=trainer.training_steps
        return self.round, (folder/f'{self.name}.pt').stat().st_size, glob.stat().st_size
