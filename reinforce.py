
# https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py

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
    def __init__(self, feature_size, action_size, device):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(feature_size * 2, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, action_size)

        self.saved_log_probs = []
        self.rewards = []
        self.feature_size = feature_size
        self.action_size = action_size
        self.device = device

    def forward(self, state):
        x = state.to(self.device)
        x = x.view(1, -1)
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x).view(-1, 1)
        #return F.softmax(action_scores, dim=1)
        return F.normalize(action_scores)

class ReinforceDecider:
    def __init__(self, args, env, env_step, cpu, gamma, get_action_from_predictions = lambda act_probs, features: F.softmax(act_probs, dim=1)):
        if cpu:
            self.device = 'cpu'
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.gamma = gamma
        self.policy = Policy(3, 3, self.device).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.eps = np.finfo(np.float32).eps.item()
        self.get_action_from_predictions = get_action_from_predictions
        self.env_step = env_step
        self.env = env

    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.policy.rewards[::-1]:
            R = r + args['gamma'] * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(self.policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]
 
    def train(self, initial_state):
        running_reward = 10
        initial_features = initial_state.get_features()
        
        for i_episode in count(1):
            state = initial_state
            features, ep_reward = np.array(initial_features), 0
            discount = 1.0

            for t in range(1, 10000):
                action_probs = self.predict(features)
                action = self.get_action_from_predictions(action_probs, features)[0].item()
                new_state = self.env_step(self.env, state, action)
                reward, done = self.get_reward(action, state, new_state)
                self.policy.rewards.append(reward)
                ep_reward += discount * reward
                discount *= self.gamma

                state = new_state
                if done:
                    break

            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            self.finish_episode()
            if i_episode % args['log_interval'] == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, ep_reward, running_reward))
            if running_reward > args['reward_threshold']:
                print("Solved! Running reward is now {} and "
                    "the last episode runs to {} time steps!".format(running_reward, t))
                break

    def get_reward(self, action, state, new_state):
        if new_state is None:
            return -10000, True
        new_documents = len(state.parallel_documents) - len(new_state.parallel_documents)
        reward = new_documents * 100
        return reward, False

    def predict(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        state = state.to(self.device)
        probs = self.policy(state)
        m = Categorical(F.softmax(probs, dim=1))
        action = m.sample()
        self.policy.saved_log_probs.append(m.log_prob(action))
        return probs