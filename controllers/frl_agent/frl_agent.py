import sys
import csv
import random
from pathlib import Path

from controller import Robot

# Configuration des imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.environment import RobotEnvironment
from src.reward import compute_reward
from src.replay_buffer import ReplayBuffer
from src.trainer import DDQNTrainer
from src.federation import FederatedClient


# ==========================================
# CONFIGURATION
# ==========================================

ACTION_INTERVAL = 5
SAVE_INTERVAL = 1000
FEDERATION_INTERVAL = 5000

robot = Robot()
name = robot.getName()

# Exploration indépendante
seed = 42 + sum(ord(c) for c in name)

random.seed(seed)
np.random.seed(seed)

# Initialisation neuronale commune
torch.manual_seed(42)
torch.set_num_threads(1)


# ==========================================
# INITIALISATION
# ==========================================

env = RobotEnvironment(robot)

buffer = ReplayBuffer(capacity=50_000)

trainer = DDQNTrainer(
    observation_size=48,
    n_actions=5
)

federation = FederatedClient(
    robot_name=name,
    root=ROOT / "federation"
)

federation_round = 0
last_federation_step = 0

step = 0
episode = 1
episode_reward = 0.0

state = None
action = None
loss = None

last_custom_data = robot.getCustomData()


# ==========================================
# STOCKAGE
# ==========================================

RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"

RESULTS.mkdir(exist_ok=True)
CHECKPOINTS.mkdir(exist_ok=True)

log_path = RESULTS / f"{name}_federated_training.csv"

print(
    f"[{name}] Federated Double DQN initialisé",
    flush=True
)


# ==========================================
# ENTRAINEMENT
# ==========================================

with open(log_path, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "step",
        "time",
        "robot",
        "episode",
        "action",
        "reward",
        "episode_reward",
        "min_distance",
        "epsilon",
        "loss",
        "done",
        "reason",
        "buffer_size",
        "federation_round"
    ])

    file.flush()

    while robot.step(env.time_step) != -1:

        step += 1

        # ==================================
        # FEDERATED LEARNING
        # ==================================

        if step % FEDERATION_INTERVAL == 0:

            local_updates = (
                trainer.training_steps
                - last_federation_step
            )

            print(
                f"[{name}] Synchronisation fédérée | "
                f"Updates: {local_updates}",
                flush=True
            )

            federation_round = federation.synchronize(
                trainer,
                count=local_updates
            )

            last_federation_step = (
                trainer.training_steps
            )

        # ==================================
        # EVENEMENTS DU SUPERVISOR
        # ==================================

        current_custom_data = robot.getCustomData()

        if current_custom_data != last_custom_data:

            last_custom_data = current_custom_data

            reason = current_custom_data.split(":")[-1]

            if reason in ("collision", "stuck", "timeout"):

                if state is not None and action is not None:

                    reward = compute_reward(
                        state,
                        state,
                        action,
                        env.max_range,
                        terminal_reason=reason
                    )

                    episode_reward += reward

                    buffer.push(
                        state,
                        action,
                        reward,
                        state,
                        True
                    )

                    loss = trainer.train_step(buffer)

                    writer.writerow([
                        step,
                        robot.getTime(),
                        name,
                        episode,
                        action,
                        reward,
                        episode_reward,
                        float(
                            state[-12:].min()
                            * env.max_range
                        ),
                        trainer.epsilon,
                        loss,
                        True,
                        reason,
                        len(buffer),
                        federation_round
                    ])

                    file.flush()

                print(
                    f"[{name}] Episode {episode} terminé | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Epsilon: {trainer.epsilon:.3f} | "
                    f"Reason: {reason} | "
                    f"Round: {federation_round}",
                    flush=True
                )

                episode += 1
                episode_reward = 0.0

                env.stop()
                env.reset_history()

                state = None
                action = None

                continue

        # ==================================
        # OBSERVATION
        # ==================================

        if step % ACTION_INTERVAL != 0:
            continue

        next_state = env.get_temporal_state()

        # ==================================
        # APPRENTISSAGE LOCAL
        # ==================================

        if state is not None and action is not None:

            reward = compute_reward(
                state,
                next_state,
                action,
                env.max_range
            )

            episode_reward += reward

            buffer.push(
                state,
                action,
                reward,
                next_state,
                False
            )

            loss = trainer.train_step(buffer)

            writer.writerow([
                step,
                robot.getTime(),
                name,
                episode,
                action,
                reward,
                episode_reward,
                float(
                    next_state[-12:].min()
                    * env.max_range
                ),
                trainer.epsilon,
                loss,
                False,
                "",
                len(buffer),
                federation_round
            ])

            file.flush()

        # ==================================
        # SELECTION DE L'ACTION
        # ==================================

        state = next_state.copy()

        action = trainer.select_action(state)

        env.apply_action(action)

        # ==================================
        # MONITORING
        # ==================================

        if step % 100 == 0:

            loss_str = (
                f"{loss:.4f}"
                if loss is not None
                else "warmup"
            )

            print(
                f"[{name}] "
                f"Episode: {episode} | "
                f"Step: {step} | "
                f"Reward: {episode_reward:.2f} | "
                f"Epsilon: {trainer.epsilon:.3f} | "
                f"Loss: {loss_str} | "
                f"Buffer: {len(buffer)} | "
                f"FedRound: {federation_round}",
                flush=True
            )

        # ==================================
        # SAUVEGARDE
        # ==================================

        if step % SAVE_INTERVAL == 0:

            trainer.save(
                CHECKPOINTS / f"{name}_federated_ddqn.pt"
            )