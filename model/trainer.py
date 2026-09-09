import time
import os
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, SubsetRandomSampler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import logging

from model.models import DiscreteActor, Critic, ContinuousActor
from model.utils import gen_seeds, make_env

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PPO:
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
        self.is_continuous = args.is_continuous
        if self.is_continuous:
            self.actor = ContinuousActor(n_wires=args.n_wires,
                                         n_blocks=args.n_blocks,
                                         input_dim=args.state_dim,
                                         output_dim=args.action_dim,
                                         ini_method=args.ini_method).to(DEVICE)
        else:
            self.actor = DiscreteActor(n_wires=args.n_wires,
                                       n_blocks=args.n_blocks,
                                       input_dim=args.state_dim,
                                       output_dim=args.action_dim,
                                       ini_method=args.ini_method).to(DEVICE)
        self.critic = Critic(args.state_dim, [64, 64], nn.Tanh).to(DEVICE)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)

    def interact_with_env(self, s):
        s = torch.tensor(s, dtype=torch.float).to(DEVICE)
        if len(s.shape) == 1:
            s = s.unsqueeze(0)

        with torch.no_grad():
            if self.is_continuous:
                dist = self.actor.get_dist(s)
                a = dist.sample()
                a_log_prob = dist.log_prob(a).sum(1)
            else:
                dist = Categorical(probs=self.actor(s))
                a = dist.sample()
                a_log_prob = dist.log_prob(a)

            v = self.critic(s)

        return a.cpu().numpy(), a_log_prob.cpu().numpy(), v

    def interact_with_new_policy(self, s, a):
        if self.is_continuous:
            dist = self.actor.get_dist(s)
            a_log_prob_now = dist.log_prob(a).sum(1).view(-1, 1)
            dist_entropy = dist.entropy().sum(1).view(-1, 1)
        else:
            dist = Categorical(probs=self.actor(s))
            a_log_prob_now = dist.log_prob(a).view(-1, 1)
            dist_entropy = dist.entropy().view(-1, 1)

        v_s = self.critic(s)
        return a_log_prob_now, dist_entropy, v_s


def make_env_direct(args):
    if args.env_name == 'UAV_Environment':
        from UAV_env import UAV_Environment
        return UAV_Environment()
    if args.env_name == 'multiUAV':
        from multiUAV_env import multiUAV
        return multiUAV()
    return gym.make(args.env_name)

def trainer(args):
    if args.is_continuous:
        if args.env_name == "LunarLander-v2":
            env = gym.make("LunarLander-v2", continuous=True)
        else:
            env = make_env_direct(args)

        args.state_dim = env.observation_space.shape[0]
        args.action_dim = env.action_space.shape[0]
    else:
        env = make_env_direct(args)

        args.state_dim = env.observation_space.shape[0]
        args.action_dim = env.action_space.n

    args.max_episode_steps = getattr(env, '_max_episode_steps', getattr(env, 'max_step', 1000))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    del env

    logger.info(f"Environment: {args.env_name}")
    logger.info(f"State dimension: {args.state_dim}")
    logger.info(f"Action dimension: {args.action_dim}")
    logger.info(f"Max episode steps: {args.max_episode_steps}")

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    writer = SummaryWriter(log_dir=f'./runs/env_{args.env_name}_{timestamp}')

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

    agent = PPO(args)
    evaluate_rewards = []

    s, _ = envs.reset(seed=seeds)

    batch_step = int(args.batch_size // args.num_envs)
    all_step = int(args.max_train_steps // args.batch_size)

    total_steps = 0
    for tran_step in tqdm(range(all_step)):

        b_s = torch.zeros((batch_step, args.num_envs) + envs.single_observation_space.shape).to(DEVICE)
        b_a = torch.zeros((batch_step, args.num_envs) + envs.single_action_space.shape).long().to(DEVICE)
        b_a_log_prob = torch.zeros((batch_step, args.num_envs)).to(DEVICE)
        b_r = torch.zeros((batch_step, args.num_envs)).to(DEVICE)
        b_vs = torch.zeros((batch_step, args.num_envs)).to(DEVICE)
        b_done = torch.zeros((batch_step, args.num_envs)).to(DEVICE)

        for step in range(batch_step):
            total_steps += args.num_envs
            a, a_log_prob, v = agent.interact_with_env(s)

            if args.is_continuous:
                action = envs.action_space.low + (a - np.zeros_like(a)) * (
                            envs.action_space.high - envs.action_space.low)
                s_, r, terminated, truncated, info = envs.step(action)
            else:
                s_, r, terminated, truncated, info = envs.step(a)

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
                    _, _, final_v = agent.interact_with_env(final_observation[idx])
                    r[idx] += agent.gamma * final_v.item()

            b_s[step] = torch.tensor(s).to(DEVICE)
            b_a[step] = torch.tensor(a).long().to(DEVICE)
            b_a_log_prob[step] = torch.tensor(a_log_prob).to(DEVICE)
            b_r[step] = torch.tensor(r).to(DEVICE)
            b_vs[step] = v.reshape(-1)
            b_done[step] = torch.tensor(done).to(DEVICE)

            s = s_

        b_vs_ = torch.zeros_like(b_vs).to(DEVICE)
        b_vs_[:-1] = b_vs[1:]
        _, _, v = agent.interact_with_env(s)
        b_vs_[-1] = v.reshape(-1)

        b_adv = torch.zeros_like(b_r).to(DEVICE)
        b_gae = torch.tensor(0.).to(DEVICE)
        for t in reversed(range(len(b_r))):
            delta = b_r[t] + agent.gamma * (1.0 - b_done[t]) * b_vs_[t] - b_vs[t]
            b_adv[t] = b_gae = delta + agent.gamma * agent.lamda * b_gae * (1.0 - b_done[t])
        b_v_target = b_adv + b_vs

        b_s = b_s.reshape((-1,) + envs.single_observation_space.shape)
        b_a = b_a.reshape((-1,) + envs.single_action_space.shape)
        b_a_log_prob = b_a_log_prob.reshape(-1, 1)

        b_adv = b_adv.reshape(-1, 1)
        b_v_target = b_v_target.reshape(-1, 1)

        clipfracs = []

        for _ in range(agent.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(agent.batch_size)), agent.mini_batch_size, False):
                a_log_prob_now, dist_entropy, v_s = agent.interact_with_new_policy(b_s[index], b_a[index])
                logratio = a_log_prob_now - b_a_log_prob[index]
                ratios = torch.exp(logratio)

                mb_adv = b_adv[index]
                mb_adv = ((mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8))

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratios - 1) - logratio).mean()
                    clipfracs += [((ratios - 1.0).abs() > args.epsilon).float().mean().item()]

                surr1 = ratios * mb_adv
                surr2 = torch.clamp(ratios, 1 - agent.epsilon, 1 + agent.epsilon) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = dist_entropy.mean()

                critic_loss = nn.functional.mse_loss(b_v_target[index], v_s)

                loss = actor_loss - agent.entropy_coef * entropy_loss + 0.5 * critic_loss

                agent.optimizer_actor.zero_grad()
                agent.optimizer_critic.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
                agent.optimizer_actor.step()
                agent.optimizer_critic.step()

        if args.lr_decay:
            lr_a_now = agent.lr_a * (1 - tran_step / all_step)
            lr_c_now = agent.lr_c * (1 - tran_step / all_step)
            for p in agent.optimizer_actor.param_groups:
                p['lr'] = lr_a_now
            for p in agent.optimizer_critic.param_groups:
                p['lr'] = lr_c_now
        if args.clip_decay:
            agent.epsilon = args.epsilon * (1 - tran_step / all_step)

    folder_path = f'./weights/env_{args.env_name}_{timestamp}/'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    torch.save(agent.actor.state_dict(), f"./weights/env_{args.env_name}_{timestamp}/PPO_actor.pt")
    torch.save(agent.critic.state_dict(), f"./weights/env_{args.env_name}_{timestamp}/PPO_critic.pt")

    if args.normalize_state:
        mean = envs.obs_rms.mean
        variance = envs.obs_rms.var
        with open(f'./weights/env_{args.env_name}_{timestamp}/PPO_norm_stats.txt', 'w') as f:
            f.write(f'Mean: {mean.tolist()}\n')
            f.write(f'Variance: {variance.tolist()}\n')
