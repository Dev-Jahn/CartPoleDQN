import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

from . import *
from .memory import ReplayMemory
from .model import SimpleMLP


class DQNTrainer:
    def __init__(
            self,
            config
    ):
        self.config = config
        self.env = gym.make(config.env_id)
        self.epsilon = self.config.eps_start  # My little gift for you
        self.net = model(input_size, output_size, [64,])
        self.net_target = model(input_size, output_size, [64,])
        params = filter(lambda p: p.requires_grad, self.net.parameters())
        self.optimizer = self.config.optim_cls(params, **self.config.optim_kwargs)

    def train(self, num_train_steps: int):
        begin_steps = self.trained_steps
        i = 0
        episode_rewards = []

        # Whatever your train loop may be.
        while i < num_train_steps:

            i += 1
            episode_reward = i

            if self.config.verbose:
                status_string = f"{self.config.run_name:10}, Whatever you want to print out to the console"
                print(status_string + "\r", end="", flush=True)

            episode_rewards.append(episode_reward)

        return episode_rewards

    # Update online network with samples in the replay memory. 
    def update_network(self):
        pass

    # Update the target network's weights with the online network's one. 
    def update_target(self):
        pass

    # Return desired action(s) that maximizes the Q-value for given observation(s) by the online network.
    def predict(self, ob):
        state = torch.FloatTensor(ob)
        pass

    # Update epsilon over training process.
    def update_epsilon(self):
        pass
