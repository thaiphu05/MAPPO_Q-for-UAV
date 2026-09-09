import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from model.models import DiscreteActor
from model.utils import orthogonal_init


class MAPPOActor(nn.Module):
    def __init__(self, n_agents, input_dim, output_dim, n_wires, n_blocks, ini_method,
                 use_quafu=False, use_quafu_simulator=True, quantum_device='Dongling'):
        super().__init__()
        self.n_agents = n_agents
        self.output_dim = output_dim
        self.actors = nn.ModuleList([
            DiscreteActor(
                n_wires=n_wires, n_blocks=n_blocks,
                input_dim=input_dim, output_dim=output_dim,
                ini_method=ini_method, is_critic=False,
                use_quafu=use_quafu, use_quafu_simulator=use_quafu_simulator,
                quantum_device=quantum_device,
            )
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
