import time
import os
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, SubsetRandomSampler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import numpy as np
import logging
import torch.nn.functional as F

from model.mappo_models import CentralizedCritic
from model.utils import gen_seeds, make_env

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_env_direct(args):
    if args.env_name == 'UAV_Environment':
        from UAV_env import UAV_Environment
        return UAV_Environment()
    if args.env_name == 'multiUAV':
        from multiUAV_env import multiUAV
        return multiUAV()
    return gym.make(args.env_name)


def actions_to_onehot(actions, n_agents, action_dim):
    onehot = F.one_hot(actions.long(), num_classes=action_dim)
    return onehot.reshape(*onehot.shape[:-2], n_agents * action_dim).float()


class MAPPOTrainer:
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.n_agents = args.n_agents
        self.agent_obs_dim = args.agent_obs_dim
        self.action_dim = 5
        self.global_obs_dim = args.global_obs_dim
        self.grid_num = args.grid_num

        hidden_dims = getattr(args, 'critic_hidden', [256, 128])

        use_quantum = getattr(args, 'use_quantum', True)
        if use_quantum:
            from model.mappo_models import MAPPOActor as ActorClass
            self.actor = ActorClass(
                n_agents=self.n_agents,
                input_dim=self.agent_obs_dim,
                output_dim=self.action_dim,
                n_wires=args.n_wires,
                n_blocks=args.n_blocks,
                ini_method=args.ini_method,
            ).to(DEVICE)
        else:
            from model.mappo_models_normal import MAPPOActor as ActorClass
            self.actor = ActorClass(
                n_agents=self.n_agents,
                input_dim=self.agent_obs_dim,
                output_dim=self.action_dim,
                hidden_dims=getattr(args, 'actor_hidden', [1024, 512]),
            ).to(DEVICE)

        self.critic = CentralizedCritic(
            global_obs_dim=self.global_obs_dim,
            hidden_dims=hidden_dims,
        ).to(DEVICE)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)

        grid2 = self.grid_num ** 2
        heatmap_offset = 2 * self.n_agents
        self.agent_slices = []
        for i in range(self.n_agents):
            ps = 2 * i
            hs = heatmap_offset + i * grid2
            us = heatmap_offset + self.n_agents * grid2
            self.agent_slices.append((ps, ps + 2, hs, hs + grid2, us, us + grid2))

    def _extract_agent_obs(self, global_obs):
        agent_obs_list = []
        for i in range(self.n_agents):
            ps, pe, hs, he, us, ue = self.agent_slices[i]
            agent_obs_list.append(
                torch.cat([global_obs[..., ps:pe], global_obs[..., hs:he], global_obs[..., us:ue]], dim=-1)
            )
        return torch.stack(agent_obs_list, dim=-2)

    @torch.no_grad()
    def interact_with_env(self, s):
        s = torch.tensor(s, dtype=torch.float).to(DEVICE)
        if s.dim() == 1:
            s = s.unsqueeze(0)

        agent_obs = self._extract_agent_obs(s)
        actions, log_probs = self.actor.get_actions(agent_obs)
        v = self.critic(s)

        return actions.cpu().numpy(), log_probs.cpu().numpy(), v

    def get_value(self, s):
        return self.critic(s)

    def interact_with_new_policy(self, s, a):
        agent_obs = self._extract_agent_obs(s)
        log_probs, entropies = self.actor.evaluate_actions(agent_obs, a)
        v_s = self.critic(s)
        return log_probs, entropies, v_s

    def encode_base5(self, actions_np):
        result = np.zeros(actions_np.shape[:-1], dtype=np.int64)
        for i in range(self.n_agents):
            result += actions_np[..., i] * (5 ** i)
        return result


def trainer(args):
    ref_env = make_env_direct(args)
    args.n_agents = ref_env.get_num_agents()
    args.agent_obs_dim = ref_env.get_agent_obs_dim()
    args.global_obs_dim = ref_env.observation_space.shape[0]
    args.grid_num = ref_env.grid_num
    args.max_episode_steps = getattr(ref_env, '_max_episode_steps', getattr(ref_env, 'max_step', 1000))
    del ref_env

    if not hasattr(args, 'critic_hidden'):
        args.critic_hidden = [256, 128]

    logger.info(f"Environment: {args.env_name}")
    logger.info(f"Number of agents: {args.n_agents}")
    logger.info(f"Agent obs dim: {args.agent_obs_dim}")
    logger.info(f"Global obs dim: {args.global_obs_dim}")
    logger.info(f"Action dim (per agent): 5")
    logger.info(f"Max episode steps: {args.max_episode_steps}")
    logger.info(f"Critic hidden layers: {args.critic_hidden}")
    logger.info(f"Model type: {'Quantum' if getattr(args, 'use_quantum', True) else 'Normal (MLP)'}")

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    model_tag = 'Quantum' if getattr(args, 'use_quantum', True) else 'Normal'
    writer = SummaryWriter(log_dir=f'./runs/env_{args.env_name}_MAPPO_{model_tag}_{timestamp}')

    seeds = gen_seeds(args)
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args, seed=seeds[i], is_continuous=args.is_continuous) for i in range(args.num_envs)]
    )

    if args.normalize_state:
        envs = gym.wrappers.NormalizeObservation(envs)
        envs = gym.wrappers.TransformObservation(envs, lambda obs: np.clip(obs, -10, 10))
    if args.normalize_reward:
        envs = gym.wrappers.NormalizeReward(envs, gamma=args.gamma)
        envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

    agent = MAPPOTrainer(args)
    evaluate_rewards = []

    s, _ = envs.reset(seed=seeds)

    batch_step = int(args.batch_size // args.num_envs)
    all_step = int(args.max_train_steps // args.batch_size)

    total_steps = 0
    for tran_step in tqdm(range(all_step)):
        b_global_obs = torch.zeros((batch_step, args.num_envs, args.global_obs_dim)).to(DEVICE)
        b_a = torch.zeros((batch_step, args.num_envs, args.n_agents)).long().to(DEVICE)
        b_a_log_prob = torch.zeros((batch_step, args.num_envs, args.n_agents)).to(DEVICE)
        b_agent_r = torch.zeros((batch_step, args.num_envs, args.n_agents)).to(DEVICE)
        b_satisfied = torch.zeros((batch_step, args.num_envs)).to(DEVICE)
        b_r = torch.zeros((batch_step, args.num_envs)).to(DEVICE)
        b_vs = torch.zeros((batch_step, args.num_envs)).to(DEVICE)
        b_done = torch.zeros((batch_step, args.num_envs)).to(DEVICE)

        for step in range(batch_step):
            total_steps += args.num_envs
            b_global_obs[step] = torch.tensor(s).to(DEVICE)
            a, a_log_prob, v = agent.interact_with_env(s)

            base5_action = agent.encode_base5(a)
            s_, r, terminated, truncated, info = envs.step(base5_action)

            # Extract per-agent rewards from info dict
            per_agent_r = np.zeros((args.num_envs, args.n_agents), dtype=np.float32)
            if 'agent_rewards' in info:
                agent_rew_raw = info['agent_rewards']
                agent_rew_raw = np.asarray(agent_rew_raw)
                if agent_rew_raw.ndim == 1:
                    per_agent_r = np.stack([np.asarray(x, dtype=np.float32) for x in agent_rew_raw])
                else:
                    per_agent_r = np.asarray(agent_rew_raw, dtype=np.float32)
            # Override with terminal envs' final step per-agent rewards
            if 'final_info' in info:
                final_mask = info.get('_final_info', np.zeros(args.num_envs, dtype=bool))
                for env_idx in np.where(final_mask)[0]:
                    if info['final_info'][env_idx] is not None and 'agent_rewards' in info['final_info'][env_idx]:
                        fa_rew = np.asarray(info['final_info'][env_idx]['agent_rewards'], dtype=np.float32)
                        per_agent_r[env_idx] = fa_rew

            # Extract satisfied_total from info dict
            satisfied_total = np.zeros(args.num_envs, dtype=np.float32)
            if 'satisfied_total' in info:
                sat_raw = np.asarray(info['satisfied_total'], dtype=np.float32)
                if sat_raw.ndim == 0:
                    satisfied_total[:] = sat_raw
                else:
                    satisfied_total = sat_raw
            if 'final_info' in info:
                final_mask = info.get('_final_info', np.zeros(args.num_envs, dtype=bool))
                for env_idx in np.where(final_mask)[0]:
                    if info['final_info'][env_idx] is not None and 'satisfied_total' in info['final_info'][env_idx]:
                        satisfied_total[env_idx] = np.float32(info['final_info'][env_idx]['satisfied_total'])

            done = np.logical_or(terminated, truncated)

            if info:
                if "final_info" in info:
                    final_mask = info.get("_final_info")
                    for env_idx in np.where(final_mask)[0]:
                        element = info["final_info"][env_idx]
                        if element is None:
                            continue
                        length = element.get("episode").get("l")[0]
                        writer.add_scalar("charts/episodic_return", element.get("episode").get("r")[0], total_steps)
                        writer.add_scalar("charts/episodic_length", length, total_steps)
                        writer.add_scalar("charts/final_step_reward", r[env_idx], total_steps)
                        evaluate_rewards.append(element.get("episode").get("r")[0])

            if truncated.any():
                final_observation = info.get("final_observation")
                for idx in np.where(truncated)[0]:
                    if final_observation[idx] is None:
                        continue
                    final_obs_t = torch.tensor(final_observation[idx], dtype=torch.float).to(DEVICE)
                    if final_obs_t.dim() == 1:
                        final_obs_t = final_obs_t.unsqueeze(0)
                    with torch.no_grad():
                        final_v = agent.get_value(final_obs_t)
                    r[idx] += agent.gamma * final_v.item()
                    per_agent_r[idx] += agent.gamma * final_v.item()

            b_a[step] = torch.tensor(a).long().to(DEVICE)
            b_a_log_prob[step] = torch.tensor(a_log_prob).to(DEVICE)
            b_agent_r[step] = torch.tensor(per_agent_r).to(DEVICE)
            b_satisfied[step] = torch.tensor(satisfied_total).to(DEVICE)
            b_r[step] = torch.tensor(r).to(DEVICE)
            b_vs[step] = v.reshape(-1)
            b_done[step] = torch.tensor(done).to(DEVICE)

            s = s_

        s_t = torch.tensor(s, dtype=torch.float).to(DEVICE)
        with torch.no_grad():
            v_next = agent.get_value(s_t)

        b_vs_ = torch.zeros_like(b_vs).to(DEVICE)
        b_vs_[:-1] = b_vs[1:]
        b_vs_[-1] = v_next.reshape(-1)

        # Team reward advantages (for critic)
        b_adv = torch.zeros_like(b_r).to(DEVICE)
        b_gae = torch.tensor(0.).to(DEVICE)
        for t in reversed(range(len(b_r))):
            delta = b_r[t] + agent.gamma * (1.0 - b_done[t]) * b_vs_[t] - b_vs[t]
            b_adv[t] = b_gae = delta + agent.gamma * agent.lamda * b_gae * (1.0 - b_done[t])
        b_v_target = b_adv + b_vs

        # Per-agent advantages (for actor)
        b_adv_agent = torch.zeros_like(b_agent_r).to(DEVICE)
        b_gae_agent = torch.zeros(args.num_envs, args.n_agents).to(DEVICE)
        for t in reversed(range(len(b_agent_r))):
            done_t = (1.0 - b_done[t]).unsqueeze(-1)
            delta = b_agent_r[t] + agent.gamma * done_t * b_vs_[t].unsqueeze(-1) - b_vs[t].unsqueeze(-1)
            b_adv_agent[t] = b_gae_agent = delta + agent.gamma * agent.lamda * done_t * b_gae_agent
        b_adv_agent = b_adv_agent.reshape(-1, args.n_agents)

        b_global_obs = b_global_obs.reshape(-1, args.global_obs_dim)
        b_a = b_a.reshape(-1, args.n_agents)
        b_a_log_prob = b_a_log_prob.reshape(-1, args.n_agents)
        b_adv = b_adv.reshape(-1, 1)
        b_v_target = b_v_target.reshape(-1, 1)

        clipfracs = []

        for _ in range(agent.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(agent.batch_size)), agent.mini_batch_size, False):
                idx = index

                global_obs_mb = b_global_obs[idx]
                v_s = agent.critic(global_obs_mb)
                critic_loss = nn.functional.mse_loss(b_v_target[idx], v_s)

                agent_obs = agent._extract_agent_obs(global_obs_mb)
                log_probs_now, entropies = agent.actor.evaluate_actions(agent_obs, b_a[idx])
                logratio = log_probs_now - b_a_log_prob[idx]
                ratios = torch.exp(logratio)

                mb_adv = b_adv_agent[idx]
                mb_adv = ((mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8))

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratios - 1) - logratio).mean()
                    clipfracs += [((ratios - 1.0).abs() > args.epsilon).float().mean().item()]

                surr1 = ratios * mb_adv
                surr2 = torch.clamp(ratios, 1 - agent.epsilon, 1 + agent.epsilon) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = entropies.mean()

                loss = actor_loss - agent.entropy_coef * entropy_loss + 0.5 * critic_loss

                agent.optimizer_actor.zero_grad()
                agent.optimizer_critic.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
                agent.optimizer_actor.step()
                agent.optimizer_critic.step()

        with torch.no_grad():
            v_pred = agent.critic(b_global_obs)
            v_explained = 1 - (b_v_target - v_pred).pow(2).sum() / (b_v_target - b_v_target.mean()).pow(2).sum()

        writer.add_scalar("losses/critic_loss", critic_loss.item(), total_steps)
        writer.add_scalar("losses/actor_loss", actor_loss.item(), total_steps)
        writer.add_scalar("losses/entropy", entropy_loss.item(), total_steps)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), total_steps)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), total_steps)
        writer.add_scalar("losses/v_explained", v_explained.item(), total_steps)
        writer.add_scalar("charts/avg_reward", b_r.mean().item(), total_steps)
        writer.add_scalar("charts/avg_satisfied_users", b_satisfied.mean().item(), total_steps)

        if args.lr_decay:
            lr_a_now = agent.lr_a * (1 - tran_step / all_step)
            lr_c_now = agent.lr_c * (1 - tran_step / all_step)
            for p in agent.optimizer_actor.param_groups:
                p['lr'] = lr_a_now
            for p in agent.optimizer_critic.param_groups:
                p['lr'] = lr_c_now
        if args.clip_decay:
            agent.epsilon = args.epsilon * (1 - tran_step / all_step)

    model_tag = 'Quantum' if getattr(args, 'use_quantum', True) else 'Normal'
    folder_path = f'./weights/env_{args.env_name}_MAPPO{model_tag}_{timestamp}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    torch.save(agent.actor.state_dict(), f"{folder_path}MAPPO_actor.pt")
    torch.save(agent.critic.state_dict(), f"{folder_path}MAPPO_critic.pt")


import yaml
import argparse


def setup_mappo_training(config_file_path):
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    batch_size = config['n_steps'] * config['num_envs']

    ini_method_list = [['NOT', 'I'], ['NOT', 'O'], ['NOT', 'X'],
                       ['I', 'I'], ['I', 'O'], ['I', 'X'],
                       ['O', 'I'], ['O', 'O'], ['O', 'X'],
                       ['X', 'I'], ['X', 'O'], ['X', 'X']]
    ini_method = ini_method_list[config.get('ini_method', 0)]

    args = {
        'env_name': config['env_name'],
        'n_steps': config['n_steps'],
        'mini_batch_size': config['mini_batch_size'],
        'max_train_steps': config['max_train_steps'],
        'lr_a': config['lr_a'],
        'lr_c': config['lr_c'],
        'gamma': config['gamma'],
        'lamda': config['lamda'],
        'epsilon': config['epsilon'],
        'K_epochs': config['K_epochs'],
        'entropy_coef': config['entropy_coef'],
        'num_envs': config['num_envs'],
        'normalize_state': config['normalize_state'],
        'normalize_reward': config['normalize_reward'],
        'clip_decay': config['clip_decay'],
        'lr_decay': config['lr_decay'],
        'ini_method': ini_method,
        'seed': config['seed'],
        'n_blocks': config.get('n_blocks', 1),
        'n_wires': config.get('n_wires', 4),
        'batch_size': batch_size,
        'is_continuous': config.get('is_continuous', False),
        'critic_hidden': config.get('critic_hidden', [256, 128]),
        'actor_hidden': config.get('actor_hidden', [1024, 512]),
        'n_agents': config.get('n_agents', 3 if config['env_name'] == 'multiUAV' else 1),
        'use_quantum': config.get('use_quantum', True),
    }

    return argparse.Namespace(**args)
