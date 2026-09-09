import math

import torch
import torch.nn as nn
from torch.distributions import Categorical

from model.utils import orthogonal_init


def _build_actor_net(input_dim, hidden_dims, output_dim):
    net = nn.Sequential()
    for in_features, out_features in zip([input_dim, *hidden_dims], hidden_dims):
        net.append(nn.Linear(in_features, out_features))
        net.append(nn.ReLU())
        orthogonal_init(net[-2], gain=math.sqrt(2))

    output_layer = nn.Linear(hidden_dims[-1], output_dim)
    orthogonal_init(output_layer, gain=0.01)
    net.append(output_layer)
    net.append(nn.Softmax(dim=-1))
    return net


class MAPPOActor(nn.Module):
    def __init__(self, n_agents, input_dim, output_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512]
        self.n_agents = n_agents
        self.output_dim = output_dim
        self.actors = nn.ModuleList([
            _build_actor_net(input_dim, hidden_dims, output_dim)
            for _ in range(n_agents)
        ])

    def forward(self, agent_obs):
        probs_list = []
        for i in range(self.n_agents):
            probs_list.append(self.actors[i](agent_obs[:, i]))
        return torch.stack(probs_list, dim=1)

    def get_actions(self, agent_obs):
        actions_list = []
        log_probs_list = []
        for i in range(self.n_agents):
            probs = self.actors[i](agent_obs[:, i])
            dist = Categorical(probs=probs)
            a = dist.sample()
            actions_list.append(a)
            log_probs_list.append(dist.log_prob(a))
        actions = torch.stack(actions_list, dim=1)
        log_probs = torch.stack(log_probs_list, dim=1)
        return actions, log_probs

    def evaluate_actions(self, agent_obs, actions):
        log_probs_list = []
        entropies_list = []
        for i in range(self.n_agents):
            probs = self.actors[i](agent_obs[:, i])
            dist = Categorical(probs=probs)
            log_probs_list.append(dist.log_prob(actions[:, i]))
            entropies_list.append(dist.entropy())
        log_probs = torch.stack(log_probs_list, dim=1)
        entropies = torch.stack(entropies_list, dim=1)
        return log_probs, entropies


class CentralizedCritic(nn.Module):
    def __init__(self, global_obs_dim, hidden_dims=None,
                 activation=nn.Tanh):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        input_dim = global_obs_dim

        self.net = nn.Sequential()
        for in_features, out_features in zip([input_dim, *hidden_dims], hidden_dims + [1]):
            self.net.append(nn.Linear(in_features, out_features, bias=True))
            if out_features != 1:
                self.net.append(activation())

        gain = 1.0
        for layer in reversed(self.net):
            if isinstance(layer, nn.Linear):
                orthogonal_init(layer, gain=gain)
                gain = 1.0

    def forward(self, global_obs):
        return self.net(global_obs)
