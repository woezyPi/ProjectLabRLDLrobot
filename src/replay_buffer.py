import random
from collections import deque, namedtuple


Transition = namedtuple(
    "Transition",
    [
        "state",
        "action",
        "reward",
        "next_state",
        "done"
    ]
)


class ReplayBuffer:

    def __init__(self, capacity=50_000):

        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.buffer.append(
            Transition(
                state.copy(),
                int(action),
                float(reward),
                next_state.copy(),
                bool(done)
            )
        )

    def sample(self, batch_size):

        return random.sample(
            self.buffer,
            batch_size
        )

    def __len__(self):

        return len(self.buffer)