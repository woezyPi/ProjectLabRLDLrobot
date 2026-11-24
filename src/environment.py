from collections import deque

import numpy as np


class RobotEnvironment:

    def __init__(self, robot):

        self.robot = robot
        self.time_step = int(robot.getBasicTimeStep())

        self.n_sectors = 12
        self.history_size = 4
        self.observation_size = 48
        self.n_actions = 5

        self.history = deque(maxlen=self.history_size)

        # LiDAR
        self.lidar = robot.getDevice("LDS-01")
        self.lidar.enable(self.time_step)
        self.max_range = self.lidar.getMaxRange()

        # Moteurs
        self.left_motor = robot.getDevice("left wheel motor")
        self.right_motor = robot.getDevice("right wheel motor")

        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))

        self.actions = {
            0: (1.5, 1.5),
            1: (0.5, 1.5),
            2: (1.5, 0.5),
            3: (-1.0, -1.0),
            4: (-1.0, 1.0)
        }

        self.stop()

    def get_state(self):

        distances = np.array(
            self.lidar.getRangeImage(),
            dtype=np.float32
        )

        distances = np.nan_to_num(
            distances,
            nan=self.max_range,
            posinf=self.max_range,
            neginf=0.0
        )

        distances = np.clip(
            distances, 0, self.max_range
        )

        sectors = np.array_split(
            distances, self.n_sectors
        )

        state = np.array(
            [np.min(s) for s in sectors],
            dtype=np.float32
        )

        return state / self.max_range

    def get_temporal_state(self):

        state = self.get_state()

        if not self.history:

            for _ in range(self.history_size):
                self.history.append(state.copy())

        else:
            self.history.append(state.copy())

        return np.concatenate(
            self.history
        ).astype(np.float32)

    def apply_action(self, action):

        left, right = self.actions[action]

        self.left_motor.setVelocity(left)
        self.right_motor.setVelocity(right)

    def reset_history(self):

        self.history.clear()

    def stop(self):

        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)