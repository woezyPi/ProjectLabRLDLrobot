import sys
import csv
import random
from pathlib import Path

from controller import Robot

# Importation des modules du projet
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.environment import RobotEnvironment

# Initialisation
robot = Robot()
env = RobotEnvironment(robot)

name = robot.getName()
step = 0
escape_steps = 0
forward_steps = 0

# Paramètres d'exploration
SAFETY_DISTANCE = 0.18
ACTION_INTERVAL = 10
ESCAPE_DURATION = 15
FORWARD_DURATION = 5

# Stockage des données
results = ROOT / "results"
results.mkdir(exist_ok=True)

file_path = results / f"{name}_exploration.csv"

print(f"[{name}] Initialisation réussie", flush=True)

with open(file_path, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "step",
        "time",
        "robot",
        *[f"sector_{i}" for i in range(12)],
        "min_distance",
        "action",
        "mode"
    ])

    while robot.step(env.time_step) != -1:

        if step % ACTION_INTERVAL == 0:

            # Observation de l'environnement
            state = env.get_state()

            min_distance = float(
                state.min() * env.max_range
            )

            # Rotation de dégagement
            if escape_steps > 0:

                action = 4
                escape_steps -= 1
                mode = "escape"

                if escape_steps == 0:
                    forward_steps = FORWARD_DURATION

            # Tentative de déplacement après rotation
            elif forward_steps > 0:

                action = 0
                forward_steps -= 1
                mode = "recovery"

            # Détection d'un obstacle proche
            elif min_distance < SAFETY_DISTANCE:

                action = 4
                escape_steps = ESCAPE_DURATION - 1
                mode = "escape"

            # Exploration aléatoire pondérée
            else:

                action = random.choices(
                    [0, 1, 2, 3, 4],
                    weights=[60, 15, 15, 5, 5],
                    k=1
                )[0]

                mode = "exploration"

            # Exécution de l'action
            env.apply_action(action)

            # Enregistrement des observations
            writer.writerow([
                step,
                robot.getTime(),
                name,
                *state.tolist(),
                min_distance,
                action,
                mode
            ])

            file.flush()

            # Affichage dans la console Webots
            print(
                f"[{name}] "
                f"Step: {step} | "
                f"Distance: {min_distance:.2f} m | "
                f"Action: {action} | "
                f"Mode: {mode}",
                flush=True
            )

        step += 1