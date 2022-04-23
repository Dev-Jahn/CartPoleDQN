from . import *
from typing import Tuple
from collections import deque
import numpy as np


class ReplayMemory:
    def __init__(
            self,
            observation_shape: tuple = (),
            action_shape: tuple = (),
            buffer_size: int = 50000,
            num_steps: int = 1,
    ):
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.buffer_size = buffer_size
        self.num_steps = num_steps
        pass

    def write(self, state, action, reward, next_state, done):
        pass

    # 버퍼에서 Uniform Random으로 Transition들을 뽑습니다.
    def sample(self, num_samples: int = 1) -> Tuple[np.ndarray]:
        return None
        # return (
        #    states,
        #    actions,
        #    rewards,
        #    next_states,
        #    dones
        # )
