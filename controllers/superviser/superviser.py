import csv
import math
import random
from pathlib import Path

from controller import Supervisor


# ==========================================
# CONFIGURATION
# ==========================================

supervisor = Supervisor()
TIME_STEP = int(supervisor.getBasicTimeStep())

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

EPISODE_DURATION = 60.0
STUCK_DURATION = 8.0
MIN_DISPLACEMENT = 0.05

ROBOT_DEFS = [
    "ROBOT_1",
    "ROBOT_2",
    "ROBOT_3"
]

OBSTACLE_DEFS = [
    "OBSTACLE_1",
    "OBSTACLE_2"
]

random.seed(42)


# ==========================================
# INITIALISATION DES ROBOTS
# ==========================================

robots = {}

for robot_def in ROBOT_DEFS:

    node = supervisor.getFromDef(robot_def)

    if node is None:
        raise RuntimeError(
            f"Robot introuvable : {robot_def}"
        )

    translation = node.getField("translation")
    rotation = node.getField("rotation")
    custom_data = node.getField("customData")

    if translation is None or rotation is None:
        raise RuntimeError(
            f"Champs de position manquants : {robot_def}"
        )

    if custom_data is None:
        raise RuntimeError(
            f"Champ customData introuvable : {robot_def}"
        )

    initial_position = list(translation.getSFVec3f())

    robots[robot_def] = {
        "node": node,
        "translation": translation,
        "rotation": rotation,
        "initial_position": initial_position,
        "initial_rotation": list(rotation.getSFRotation()),
        "last_position": initial_position.copy(),
        "last_motion_time": supervisor.getTime(),
        "resets": 0,
        "custom_data": custom_data
    }

    custom_data.setSFString("0:init")

    print(
        f"[SUPERVISOR] {robot_def} connecté",
        flush=True
    )


# ==========================================
# INITIALISATION DES OBSTACLES
# ==========================================

obstacles = {}

for i, obstacle_def in enumerate(OBSTACLE_DEFS):

    node = supervisor.getFromDef(obstacle_def)

    if node is None:

        print(
            f"[SUPERVISOR] {obstacle_def} absent",
            flush=True
        )

        continue

    translation = node.getField("translation")

    if translation is None:
        raise RuntimeError(
            f"Translation introuvable : {obstacle_def}"
        )

    obstacles[obstacle_def] = {
        "translation": translation,
        "origin": list(translation.getSFVec3f()),
        "amplitude": 0.3,
        "frequency": 0.1,
        "phase": random.uniform(0, 2 * math.pi),
        "axis": i % 2
    }

    print(
        f"[SUPERVISOR] {obstacle_def} connecté",
        flush=True
    )


# ==========================================
# REINITIALISATION DES ROBOTS
# ==========================================

def reset_robot(robot_def, reason="stuck"):

    data = robots[robot_def]

    # Réinitialisation de la position
    data["translation"].setSFVec3f(
        data["initial_position"]
    )

    # Réinitialisation de l'orientation
    data["rotation"].setSFRotation(
        data["initial_rotation"]
    )

    # Réinitialisation physique
    data["node"].resetPhysics()

    # Réinitialisation du suivi
    data["last_position"] = (
        data["initial_position"].copy()
    )

    data["last_motion_time"] = supervisor.getTime()
    data["resets"] += 1

    # Notification de l'agent
    data["custom_data"].setSFString(
        f"{data['resets']}:{reason}"
    )

    print(
        f"[SUPERVISOR] {robot_def} | "
        f"Reset: {data['resets']} | "
        f"Reason: {reason}",
        flush=True
    )


# ==========================================
# DETECTION DES BLOCAGES
# ==========================================

def check_stuck(robot_def, current_time):

    data = robots[robot_def]

    position = list(
        data["translation"].getSFVec3f()
    )

    # Déplacement horizontal uniquement
    distance = math.dist(
        position[:2],
        data["last_position"][:2]
    )

    if distance >= MIN_DISPLACEMENT:

        data["last_position"] = position
        data["last_motion_time"] = current_time

        return False

    elapsed = (
        current_time - data["last_motion_time"]
    )

    return elapsed >= STUCK_DURATION


# ==========================================
# ENVIRONNEMENT DYNAMIQUE
# ==========================================

def move_obstacles(current_time):

    for obstacle in obstacles.values():

        position = obstacle["origin"].copy()

        displacement = (
            obstacle["amplitude"]
            * math.sin(
                2 * math.pi
                * obstacle["frequency"]
                * current_time
                + obstacle["phase"]
            )
        )

        position[obstacle["axis"]] += displacement

        obstacle["translation"].setSFVec3f(position)


def change_environment(current_time):

    for obstacle in obstacles.values():

        old_frequency = obstacle["frequency"]

        new_frequency = random.uniform(
            0.05, 0.2
        )

        # Continuité de la trajectoire lors du changement
        obstacle["phase"] += (
            2 * math.pi
            * (old_frequency - new_frequency)
            * current_time
        )

        obstacle["phase"] %= 2 * math.pi
        obstacle["frequency"] = new_frequency

    print(
        "[SUPERVISOR] Environnement modifié",
        flush=True
    )


# ==========================================
# INITIALISATION DES EPISODES
# ==========================================

episode = 1
episode_start = supervisor.getTime()

log_path = RESULTS / "supervisor_events.csv"

print(
    "[SUPERVISOR] Initialisation terminée",
    flush=True
)

print(
    f"[SUPERVISOR] Agents : {len(robots)} | "
    f"Obstacles mobiles : {len(obstacles)}",
    flush=True
)


# ==========================================
# BOUCLE PRINCIPALE
# ==========================================

with open(log_path, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "time",
        "episode",
        "robot",
        "event",
        "resets"
    ])

    file.flush()

    while supervisor.step(TIME_STEP) != -1:

        current_time = supervisor.getTime()

        # Déplacement des obstacles
        move_obstacles(current_time)

        # Durée de l'épisode actuel
        elapsed = current_time - episode_start

        # ==================================
        # FIN D'EPISODE
        # ==================================

        if elapsed >= EPISODE_DURATION:

            print(
                f"[SUPERVISOR] Fin épisode {episode}",
                flush=True
            )

            for robot_def in robots:

                reset_robot(
                    robot_def,
                    reason="timeout"
                )

                writer.writerow([
                    current_time,
                    episode,
                    robot_def,
                    "episode_end",
                    robots[robot_def]["resets"]
                ])

            file.flush()

            # Modification de l'environnement
            change_environment(current_time)

            episode += 1
            episode_start = current_time

            print(
                f"[SUPERVISOR] Début épisode {episode}",
                flush=True
            )

            # Évite une deuxième réinitialisation
            # durant le même pas de simulation
            continue

        # ==================================
        # SURVEILLANCE DES ROBOTS
        # ==================================

        for robot_def in robots:

            if check_stuck(robot_def, current_time):

                reset_robot(
                    robot_def,
                    reason="stuck"
                )

                writer.writerow([
                    current_time,
                    episode,
                    robot_def,
                    "stuck_reset",
                    robots[robot_def]["resets"]
                ])

                file.flush()