import argparse

import torch
import numpy as np
import gymnasium as gym

def orthogonal_init(layer, gain=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, gain)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def xavier_init(layer):
    torch.nn.init.xavier_uniform_(layer.weight)
    torch.nn.init.constant_(layer.bias, 0.0)
    return layer

def ones_init(layer):
    torch.nn.init.ones_(layer.weight)
    torch.nn.init.constant_(layer.bias, 0.0)
    return layer

INIT_METHOD = {
    'O': orthogonal_init,
    'X': xavier_init,
    'I': ones_init,
}

def calculate_all_Z_expectations(counts):
    num_qubits = len(next(iter(counts)))
    total_counts = sum(counts.values())
    zero_nums = np.zeros(num_qubits)
    one_nums = np.zeros(num_qubits)
    for bitstring, count in counts.items():
        for bit_index, bit in enumerate(bitstring):
            if '0' == bit:
                zero_nums[bit_index] += count
            else:
                one_nums[bit_index] += count
    P0 = zero_nums / total_counts
    P1 = one_nums / total_counts
    return P0 - P1

def gen_task(circuit, device):
    task = {
        'chip': device,
        'name': 'MyJob',
        'circuit': circuit,
        'compile': True,
        'correct': False
    }
    return task

def height(xs):
    return np.sin(3 * xs) * 0.45 + 0.55


def make_env(args, seed, is_continuous=False):
    def thunk():
        if args.env_name == 'UAV_Environment':
            from UAV_env import UAV_Environment
            env = UAV_Environment()
        elif args.env_name == 'multiUAV':
            from multiUAV_env import multiUAV
            env = multiUAV()
        elif args.env_name == "LunarLander-v2" and is_continuous:
            env = gym.make("LunarLander-v2", continuous=True)
        else:
            env = gym.make(args.env_name)

        if args.env_name == 'MountainCar-v0':
            env = gym.wrappers.TransformReward(env, lambda r: r + height(getattr(env.unwrapped, 'state', [0, 0])[0]))
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def gen_seeds(args):
    a = args.seed
    seeds = []
    for _ in range(args.num_envs):
        seeds.append(a)
        a = a + 1
    return seeds

import yaml

def load_config_from_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_training(config_file_path):
    config = load_config_from_yaml(config_file_path)

    env_name = config['env_name']
    n_steps = config['n_steps']
    mini_batch_size = config['mini_batch_size']
    max_train_steps = config['max_train_steps']
    lr_a = config['lr_a']
    lr_c = config['lr_c']
    gamma = config['gamma']
    lamda = config['lamda']
    epsilon = config['epsilon']
    K_epochs = config['K_epochs']
    entropy_coef = config['entropy_coef']
    num_envs = config['num_envs']
    normalize_state = config['normalize_state']
    normalize_reward = config['normalize_reward']
    clip_decay = config['clip_decay']
    lr_decay = config['lr_decay']
    ini_method_index = config['ini_method']
    seed = config['seed']
    n_blocks = config['n_blocks']
    n_wires = config['n_wires']

    batch_size = n_steps * num_envs

    ini_method_list = [['NOT', 'I'], ['NOT', 'O'], ['NOT', 'X'],
                       ['I', 'I'], ['I', 'O'], ['I', 'X'],
                       ['O', 'I'], ['O', 'O'], ['O', 'X'],
                       ['X', 'I'], ['X', 'O'], ['X', 'X']]
    ini_method = ini_method_list[ini_method_index]

    is_continuous = config['is_continuous']

    args = {
        'env_name': env_name,
        'n_steps': n_steps,
        'mini_batch_size': mini_batch_size,
        'max_train_steps': max_train_steps,
        'lr_a': lr_a,
        'lr_c': lr_c,
        'gamma': gamma,
        'lamda': lamda,
        'epsilon': epsilon,
        'K_epochs': K_epochs,
        'entropy_coef': entropy_coef,
        'num_envs': num_envs,
        'normalize_state': normalize_state,
        'normalize_reward': normalize_reward,
        'clip_decay': clip_decay,
        'lr_decay': lr_decay,
        'ini_method': ini_method,
        'seed': seed,
        'n_blocks': n_blocks,
        'n_wires': n_wires,
        'batch_size': batch_size,
        'is_continuous': is_continuous,
    }

    args = argparse.Namespace(**args)

    return args

def get_config_path(file_name: str) -> str:
    return f"config/{file_name}.yaml"

