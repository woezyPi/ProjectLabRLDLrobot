import csv, json, random, sys
from pathlib import Path
from controller import Robot
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.environment import RobotEnvironment
from src.reward import compute_reward
from src.replay_buffer import ReplayBuffer
from src.trainer import DDQNTrainer
from src.benchmark_support import config, Federation, MOTION_STEPS, ROBOTS, atomic_save

c=config(); robot=Robot(); name=robot.getName(); idx=ROBOTS.index(name)
training=c['phase']=='train'; method=c['method']; reactive=method=='reactive'; out=Path(c['phase_dir'])
seed=c['seed']*100+idx; random.seed(seed); np.random.seed(seed); torch.manual_seed(c['seed']); torch.set_num_threads(1)
env=RobotEnvironment(robot); rng=np.random.default_rng(c['scene_seed']*100+idx)
trainer=None if reactive else DDQNTrainer(observation_size=48,n_actions=5,double=method!='dqn_local',dueling=method=='dueling_ddqn_local')
if trainer is not None and not training:
    weights=torch.load(Path(c['run_dir'])/'checkpoints'/f'{name}.pt',map_location='cpu',weights_only=True)
    trainer.online.load_state_dict(weights); trainer.online.eval(); trainer.epsilon=0.
if trainer is not None:
    (out/f'{name}_parameters.json').write_text(json.dumps({'parameters':sum(p.numel() for p in trainer.online.parameters())}))
buffer=ReplayBuffer(capacity=50_000) if training else None
fed=Federation(c,name,trainer) if training and ('fedavg' in method or 'median' in method) else None

def tick():
    if robot.step(env.time_step)==-1: raise SystemExit(0)

def observe():
    s=np.asarray(env.get_temporal_state(),dtype=np.float32).copy()
    if s.shape!=(48,) or not np.isfinite(s).all(): raise ValueError(f'Unexpected observation {s.shape}')
    raw=s.copy()
    if c['scenario']=='sensor_noise': s=np.clip(s+rng.normal(0,c['noise_std'],s.shape),0,1).astype(np.float32)
    return s,raw

def action_for(s):
    if not reactive: return trainer.select_action(s)
    # Direction-independent conservative baseline: advance in clear surroundings, rotate otherwise.
    # No assumption about the LiDAR sector order.
    return 0 if float(s[-12:].min()*env.max_range)>.30 else 4

fields=['robot','interaction','reward','loss','epsilon','episode','reason','federation_round','upload_bytes','download_bytes'] if training else ['checkpoint','scenario','scene_seed','robot','episode','epsilon','episode_return','success','length','seconds','distance_m','collisions','stuck']
with (out/f'{name}.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
    env.stop(); tick(); state,raw_state=observe(); episode=1; ret=0.; length=0; cycle=0; active=True
    while True:
        cycle+=1
        if active: action=action_for(state); env.apply_action(action)
        else: env.stop()
        for _ in range(MOTION_STEPS): tick()
        env.stop()
        tick() # braking step, still included in the measured transition
        # Capture the actual successor BEFORE the supervisor resets the pose.
        next_state,raw_next=observe() if active else (None,None)
        tick() # supervisor publishes the outcome before this step
        msg=json.loads(robot.getCustomData())
        if msg['cycle']!=cycle: raise RuntimeError(f'Cycle mismatch: {cycle}, {msg}')
        if active:
            reason=msg['reason']; terminal=bool(reason)
            reward=compute_reward(raw_state,raw_next,action,env.max_range,terminal_reason=reason or None)
            ret+=reward; length+=1
            if training:
                # Time limit is a truncation; bootstrap from the pre-reset successor.
                buffer.push(state,action,reward,next_state,reason in ('collision','stuck'))
                loss=trainer.train_step(buffer); rnd=0; up=down=0
                if fed is not None and cycle%c['federation_interval']==0:
                    rnd,up,down=fed.sync(trainer)
                writer.writerow(dict(robot=name,interaction=cycle,reward=reward,loss=loss,epsilon=trainer.epsilon,episode=episode,reason=reason,federation_round=rnd,upload_bytes=up,download_bytes=down))
            elif terminal:
                writer.writerow(dict(checkpoint=c['budget'],scenario=c['scenario'],scene_seed=msg['episode_scene_seed'],robot=name,episode=episode,epsilon=0,episode_return=ret,success=int(reason=='timeout' and msg['distance_m']>=c['min_success_distance']),length=length,seconds=msg['seconds'],distance_m=msg['distance_m'],collisions=int(reason=='collision'),stuck=int(reason=='stuck')))
            if terminal:
                ret=0.; length=0; episode+=1; env.reset_history()
            if training and cycle==c['budget']:
                p=Path(c['run_dir'])/'checkpoints'; p.mkdir(exist_ok=True)
                atomic_save(trainer.online.state_dict(),p/f'{name}.pt'); active=False
            elif not training and terminal: active=False
            f.flush()
        # reset/settle step. Nonterminal history already advanced above: reuse that sample.
        tick()
        if not training and msg['new_trial'] and not msg['finish']:
            active=True; env.reset_history()
            rng=np.random.default_rng(msg['next_scene_seed']*100+idx)
            state,raw_state=observe()
        elif active:
            state,raw_state=observe() if terminal else (next_state,raw_next)
        if msg['finish']:
            (out/f'{name}.done').write_text('ok'); env.stop(); tick(); break
