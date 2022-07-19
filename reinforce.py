import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import argparse
import numpy as np
from itertools import count

args = {
    'seed': 1234,
    'gamma': 0.99,
    'log_interval': 10,
    'reward_threshold': 100
}

torch.manual_seed(args['seed'])

class Policy(nn.Module):
    def __init__(self, feature_size, action_size):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(feature_size, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, action_size)

        self.saved_log_probs = []
        self.rewards = []
        self.feature_size = feature_size
        self.action_size = action_size

    def forward(self, state):
        x = state.to(device)
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        #return F.softmax(action_scores, dim=1)
        return F.normalize(action_scores)

class ReinforceDecider:
    def __init__(self, args, env_step, get_action_from_action_probabilities = lambda x: F.softmax(x, dim=1)):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy = Policy(3, 3).to(device)
        self.optimizer = optim.Adam(policy.parameters(), lr=1e-2)
        self.eps = np.finfo(np.float32).eps.item()
        self.get_action_from_action_probabilities = get_action_from_action_probabilities
        self.env_step

    def train(self, initial_state):
        running_reward = 10
        for i_episode in count(1):
            state, ep_reward = np.array(initial_state), 0
            for t in range(1, 10000):
                action_probs = self.predict(state)
                action = self.get_action_from_action_probabilities(action_probs, state)
                new_state = self.env_step(action, state)
                reward, done = self.get_reward(action, state, new_state)

                policy.rewards.append(reward)
                ep_reward += reward
                if done:
                    break

            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            finish_episode()
            if i_episode % args['log_interval'] == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, ep_reward, running_reward))
            if running_reward > args['reward_threshold']:
                print("Solved! Running reward is now {} and "
                    "the last episode runs to {} time steps!".format(running_reward, t))
                break

    def get_reward(self, action, state, new_state, metadata):
        return 0, False

    def predict(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        state = state.to(device)
        probs = self.policy(state)
        return probs