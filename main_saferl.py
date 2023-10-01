# env
from safe_rl_algos.utils.vectorize import ConAsyncVectorEnv
from safe_rl_algos.tasks.safety_gym import SafetyGymEnv

# algorithm
from safe_rl_algos.algos import algo_dict

# utils
from safe_rl_algos.utils import backupFiles, setSeed, cprint
from safe_rl_algos.utils.slackbot import Slackbot
from safe_rl_algos.utils.logger import Logger

# base
from ruamel.yaml import YAML
from copy import deepcopy
import numpy as np
import argparse
import pickle
import random
import torch
import wandb
import time
import os

def getParser():
    parser = argparse.ArgumentParser(description='RL')
    # common
    parser.add_argument('--wandb', action='store_true', help='use wandb?')
    parser.add_argument('--slack', action='store_true', help='use slack?')
    parser.add_argument('--test', action='store_true', help='test or train?')
    parser.add_argument('--device', type=str, default='gpu', help='gpu or cpu.')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index.')
    parser.add_argument('--model_num', type=int, default=0, help='num model.')
    parser.add_argument('--save_freq', type=int, default=int(1e6), help='# of time steps for save.')
    parser.add_argument('--wandb_freq', type=int, default=int(1e3), help='# of time steps for wandb logging.')
    parser.add_argument('--slack_freq', type=int, default=int(2.5e6), help='# of time steps for slack message.')
    parser.add_argument('--seed', type=int, default=1, help='seed number.')
    parser.add_argument('--task_cfg_path', type=str, help='cfg.yaml file location for task.')
    parser.add_argument('--algo_cfg_path', type=str, help='cfg.yaml file location for algorithm.')
    parser.add_argument('--project_name', type=str, default="[RL] Gymnasium", help='wandb project name.')
    parser.add_argument('--comment', type=str, default=None, help='wandb comment saved in run name.')
    return parser

def train(args, task_cfg, algo_cfg):
    # set seed
    setSeed(args.seed)

    # backup configurations
    backup_file_list = list(task_cfg['backup_files']) + list(algo_cfg['backup_files'])
    backupFiles(f"{args.save_dir}/backup", backup_file_list)

    # set arguments
    args.n_envs = task_cfg['n_envs']
    args.n_steps = algo_cfg['n_steps']
    args.n_total_steps = task_cfg['n_total_steps']
    args.max_episode_len = task_cfg['max_episode_len']

    # create environments
    if task_cfg['make_args']:
        env_id = lambda: SafetyGymEnv(args.task_name, **task_cfg['make_args'])
    else:
        env_id = lambda: SafetyGymEnv(args.task_name)
    vec_env = ConAsyncVectorEnv([env_id for _ in range(args.n_envs)])
    args.obs_dim = vec_env.single_observation_space.shape[0]
    args.action_dim = vec_env.single_action_space.shape[0]
    args.cost_dim = vec_env.single_cost_space.shape[0]
    args.action_bound_min = vec_env.single_action_space.low
    args.action_bound_max = vec_env.single_action_space.high
    args.cost_names = task_cfg["costs"]
    assert len(args.cost_names) == args.cost_dim

    # declare agent
    agent_args = deepcopy(args)
    for key in algo_cfg.keys():
        agent_args.__dict__[key] = algo_cfg[key]
    agent = algo_dict[args.algo_name.lower()](agent_args)
    initial_step = agent.load(args.model_num)

    # wandb
    if args.wandb:
        wandb.init(project=args.project_name, config=args)
        if args.comment is not None:
            wandb.run.name = f"{args.name}/{args.comment}"
        else:
            wandb.run.name = f"{args.name}"

    # slackbot
    if args.slack:
        slackbot = Slackbot()

    # logger
    log_name_list = deepcopy(agent_args.logging['cost_indep'])
    for log_name in agent_args.logging['cost_dep']:
        log_name_list += [f"{log_name}_{cost_name}" for cost_name in args.cost_names]
    logger = Logger(log_name_list, f"{args.save_dir}/logs")

    # set train parameters
    reward_sums = np.zeros(args.n_envs)
    cost_sums = np.zeros((args.n_envs, args.cost_dim))
    env_cnts = np.zeros(args.n_envs)
    total_step = initial_step
    wandb_step = initial_step
    slack_step = initial_step
    save_step = initial_step

    # initialize environments
    observations, infos = vec_env.reset()

    # start training
    for _ in range(int(initial_step/args.n_steps), int(args.n_total_steps/args.n_steps)):
        start_time = time.time()

        for _ in range(int(args.n_steps/args.n_envs)):
            env_cnts += 1
            total_step += args.n_envs

            # ======= collect trajectories & training ======= #
            actions = agent.getAction(observations, False)
            observations, rewards, terminates, truncates, infos = vec_env.step(actions)
            costs = rewards[..., 1:]
            rewards = rewards[..., 0]

            reward_sums += rewards
            cost_sums += costs
            temp_fails = []
            temp_dones = []
            temp_observations = []

            for env_idx in range(args.n_envs):
                fail = (not truncates[env_idx]) and terminates[env_idx]
                done = terminates[env_idx] or truncates[env_idx]
                temp_observations.append(
                    infos['final_observation'][env_idx] 
                    if done else observations[env_idx])
                temp_fails.append(fail)
                temp_dones.append(done)

                if done:
                    eplen = env_cnts[env_idx]
                    if 'eplen' in logger.log_name_list: 
                        logger.write('eplen', [eplen, eplen])
                    if 'reward_sum' in logger.log_name_list:
                        logger.write('reward_sum', [eplen, reward_sums[env_idx]])
                    for cost_idx in range(args.cost_dim):
                        log_name = f'cost_sum_{args.cost_names[cost_idx]}'
                        if log_name in logger.log_name_list: 
                            logger.write(log_name, [eplen, cost_sums[env_idx, cost_idx]])
                    reward_sums[env_idx] = 0
                    cost_sums[env_idx, :] = 0
                    env_cnts[env_idx] = 0

            temp_dones = np.array(temp_dones)
            temp_fails = np.array(temp_fails)
            temp_observations = np.array(temp_observations)
            agent.step(rewards, costs, temp_dones, temp_fails, temp_observations)
            # =============================================== #

            # wandb logging
            if total_step - wandb_step >= args.wandb_freq and args.wandb:
                wandb_step += args.wandb_freq
                log_data = {"step": total_step}
                print_len_episode = max(int(args.wandb_freq/args.max_episode_len), args.n_envs)
                print_len_step = max(int(args.wandb_freq/args.n_steps), args.n_envs)
                for cost_idx, cost_name in enumerate(args.cost_names):
                    for log_name in agent_args.logging['cost_dep']:
                        if 'sum' in log_name:                        
                            log_data[f'{log_name}/{cost_name}'] = logger.get_avg(f'{log_name}_{cost_name}', print_len_episode)
                        else:
                            log_data[f'{log_name}/{cost_name}'] = logger.get_avg(f'{log_name}_{cost_name}', print_len_step)
                for log_name in agent_args.logging['cost_indep']:
                    if log_name in ['eplen', 'reward_sum']:
                        log_data[f"metric/{log_name}"] = logger.get_avg(log_name, print_len_episode)
                    else:
                        log_data[f"metric/{log_name}"] = logger.get_avg(log_name, print_len_step)
                wandb.log(log_data)
                print(log_data)

            # send slack message
            if total_step - slack_step >= args.slack_freq and args.slack:
                slackbot.sendMsg(f"{args.project_name}\nname: {wandb.run.name}\nsteps: {total_step}\nlog: {log_data}")
                slack_step += args.slack_freq

            # save
            if total_step - save_step >= args.save_freq:
                save_step += args.save_freq
                agent.save(total_step)
                logger.save()

        # train
        if agent.readyToTrain():
            train_results = agent.train()
            for log_name in agent_args.logging['cost_indep']:
                if log_name in ['fps', 'eplen', 'reward_sum']: continue
                logger.write(log_name, [args.n_steps, train_results[log_name]])
            for cost_idx, cost_name in enumerate(args.cost_names):
                for log_name in agent_args.logging['cost_dep']:
                    if log_name in ['cost_sum']: continue
                    logger.write(f"{log_name}_{cost_name}", [args.n_steps, train_results[log_name][cost_idx]])

        # calculate FPS
        end_time = time.time()
        fps = args.n_steps/(end_time - start_time)
        if 'fps' in agent_args.logging['cost_indep']:
            logger.write('fps', [args.n_steps, fps])

    # final save
    agent.save(total_step)
    logger.save()

    # terminate
    vec_env.close()

def test(args, task_cfg, algo_cfg):
    pass

if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()

    # ==== processing args ==== #
    # load configuration file
    with open(args.task_cfg_path, 'r') as f:
        task_cfg = YAML().load(f)
    args.task_name = task_cfg['name']
    with open(args.algo_cfg_path, 'r') as f:
        algo_cfg = YAML().load(f)
    args.algo_name = algo_cfg['name']
    args.name = f"{(args.task_name.lower())}_{(args.algo_name.lower())}"
    # save_dir
    args.save_dir = f"results/{args.name}/seed_{args.seed}"
    # device
    if torch.cuda.is_available() and args.device == 'gpu':
        device = torch.device(f'cuda:{args.gpu_idx}')
        cprint('[torch] cuda is used.', bold=True, color='cyan')
    else:
        device = torch.device('cpu')
        cprint('[torch] cpu is used.', bold=True, color='cyan')
    args.device = device
    # ========================= #

    if args.test:
        test(args, task_cfg, algo_cfg)
    else:
        train(args, task_cfg, algo_cfg)
