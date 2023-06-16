# env
from stable_baselines3.common.env_util import make_vec_env
from utils.vectorize import CustomSubprocVecEnv
import tasks.register

# algorithm
from algos.offtrc import Agent as OffTRC
from algos.wcsac import Agent as WCSAC
from algos.ipo import Agent as IPO
algo_dict = {'offtrc': OffTRC, 'wcsac': WCSAC, 'ipo': IPO}

# utils
from utils import backupFiles, setSeed, cprint
from utils.slackbot import Slackbot
from utils.logger import Logger

# base
from ruamel.yaml import YAML
from copy import deepcopy
import numpy as np
import argparse
import torch
import wandb
import time
import gym

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
    parser.add_argument('--wandb_freq', type=int, default=int(1e3), help='# of time steps for save.')
    parser.add_argument('--slack_freq', type=int, default=int(2.5e6), help='# of time steps for slack message.')
    parser.add_argument('--total_steps', type=int, default=int(5e6), help='total training steps.')
    parser.add_argument('--seed', type=int, default=1, help='seed number.')
    parser.add_argument('--task_cfg_path', type=str, help='cfg.yaml file location for task.')
    parser.add_argument('--algo_cfg_path', type=str, help='cfg.yaml file location for algorithm.')
    parser.add_argument('--comment', type=str, default=None, help='wandb comment saved in run name.')
    return parser

def train(args, task_cfg, algo_cfg):
    # set seed
    setSeed(args.seed)

    # backup configurations
    backup_file_list = list(task_cfg['backup_files']) + list(algo_cfg['backup_files'])
    backupFiles(f"{args.save_dir}/backup", backup_file_list)

    # create environment from the configuration file
    args.max_episode_len = task_cfg['max_episode_len']
    args.n_envs = task_cfg['n_envs']
    args.n_steps = algo_cfg['n_steps']
    env_id = lambda: gym.make(args.task_name)
    vec_env = make_vec_env(
        env_id=env_id, n_envs=args.n_envs, seed=args.seed,
        vec_env_cls=CustomSubprocVecEnv,
        vec_env_kwargs={'start_method':'spawn'},
    )
    args.obs_dim = vec_env.observation_space.shape[0]
    args.action_dim = vec_env.action_space.shape[0]
    args.action_bound_min = vec_env.action_space.low
    args.action_bound_max = vec_env.action_space.high

    # declare agent
    agent_args = deepcopy(args)
    for key in algo_cfg.keys():
        agent_args.__dict__[key] = algo_cfg[key]
    agent = algo_dict[args.algo_name.lower()](agent_args)
    initial_step = agent.load(args.model_num)
    if initial_step != 0: # load state normalization scale.
        vec_env.loadScaling(args.save_dir, args.model_num)

    # wandb
    if args.wandb:
        project_name = '[Safe RL]'
        wandb.init(project=project_name, config=args)
        if args.comment is not None:
            wandb.run.name = f"{args.name}/{args.comment}"
        else:
            wandb.run.name = f"{args.name}"

    # slackbot
    if args.slack:
        slackbot = Slackbot()

    # logger
    log_name_list = []
    for key in agent_args.logging.keys():
        log_name_list += agent_args.logging[key]
    logger = Logger(log_name_list, f"{args.save_dir}/logs")

    # train
    reward_sums = np.zeros(args.n_envs)
    cost_sums = np.zeros(args.n_envs)
    env_cnts = np.zeros(args.n_envs)
    total_step = initial_step
    wandb_step = initial_step
    slack_step = initial_step
    save_step = initial_step
    initial_epoch = int(initial_step/args.n_steps)
    n_total_epochs = int(args.total_steps/args.n_steps)
    observations = vec_env.reset()

    for _ in range(initial_epoch, n_total_epochs):
        start_time = time.time()

        for _ in range(int(args.n_steps/args.n_envs)):
            env_cnts += 1
            total_step += args.n_envs

            # ======= collect trajectories & training ======= #
            with torch.no_grad():
                obs_tensor = torch.tensor(observations, device=args.device, dtype=torch.float32)
                action_tensor = agent.getAction(obs_tensor, deterministic=False)
                actions = action_tensor.detach().cpu().numpy()

            observations, rewards, dones, infos = vec_env.step(actions)
            costs = np.array([info['cost'] for info in infos])
            reward_sums += rewards
            cost_sums += costs
            fails = np.zeros(args.n_envs)
            next_observations = np.zeros_like(observations)

            for env_idx in range(args.n_envs):
                next_observations[env_idx] = deepcopy(infos[env_idx]['terminal_observation'] if dones[env_idx] else observations[env_idx])
                fails[env_idx] = env_cnts[env_idx] < args.max_episode_len if dones[env_idx] else False
                dones[env_idx] = True if env_cnts[env_idx] >= args.max_episode_len else dones[env_idx]

                if dones[env_idx]:
                    # logging
                    eplen = env_cnts[env_idx]
                    log_name = 'reward_sum'
                    if log_name in logger.log_name_list:
                        logger.write(log_name, [eplen, reward_sums[env_idx]])
                    log_name = 'cost_sum'
                    if log_name in logger.log_name_list:
                        logger.write(log_name, [eplen, cost_sums[env_idx]])
                    log_name = 'eplen'
                    if log_name in logger.log_name_list:
                        logger.write(log_name, [eplen, env_cnts[env_idx]])
                    reward_sums[env_idx] = 0
                    cost_sums[env_idx] = 0
                    env_cnts[env_idx] = 0

            agent.step(rewards, costs, dones, fails, next_observations)
            # =============================================== #

            # wandb logging
            if total_step - wandb_step >= args.wandb_freq and args.wandb:
                wandb_step += args.wandb_freq
                log_data = {"step": total_step}
                print_len = max(int(args.wandb_freq/args.max_episode_len), args.n_envs)
                for key in agent_args.logging.keys():
                    for log_name in agent_args.logging[key]:
                        log_data[f'{key}/{log_name}'] = logger.get_avg(log_name, print_len)
                wandb.log(log_data)
                print(log_data)

            # send slack message
            if total_step - slack_step >= args.slack_freq and args.slack:
                slackbot.sendMsg(f"{project_name}\nname: {wandb.run.name}\nsteps: {total_step}\nlog: {log_data}")
                slack_step += args.slack_freq

            # save
            if total_step - save_step >= args.save_freq:
                save_step += args.save_freq
                vec_env.saveScaling(args.save_dir, total_step)
                agent.save(total_step)
                logger.save()

        # train
        train_results = agent.train()
        for log_name in agent_args.logging['metric']:
            if log_name in ['fps']: continue
            logger.write(log_name, [args.n_steps, train_results[log_name]])
        for log_name in agent_args.logging['train']:
            logger.write(log_name, [args.n_steps, train_results[log_name]])

        # calculate FPS
        end_time = time.time()
        fps = args.n_steps/(end_time - start_time)
        if 'fps' in agent_args.logging['metric']:
            logger.write('fps', [args.n_steps, fps])

    # final save
    vec_env.saveScaling(args.save_dir, total_step)
    agent.save(total_step)
    logger.save()


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
