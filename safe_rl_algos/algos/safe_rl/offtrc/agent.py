from algos.common.actor_gaussian import ActorGaussian
from algos.common.critic_base import CriticS
from algos.common.agent_base import AgentBase

from algos.offtrc.storage import ReplayBuffer
from algos.offtrc.optimizer import ConTROptimizer
from utils import cprint

from scipy.stats import norm
from typing import Tuple
import numpy as np
import torch
import os

EPS = 1e-8

class Agent(AgentBase):
    def __init__(self, args):
        # base
        self.device = args.device
        self.name = args.name
        self.checkpoint_dir=f'{args.save_dir}/checkpoint'

        # for env
        self.discount_factor = args.discount_factor
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.action_bound_min = args.action_bound_min
        self.action_bound_max = args.action_bound_max
        self.n_envs = args.n_envs
        self.n_steps = args.n_steps
        self.n_update_steps = args.n_update_steps

        # for RL
        self.critic_lr = args.critic_lr
        self.n_critic_iters = args.n_critic_iters
        self.max_grad_norm = args.max_grad_norm
        self.gae_coeff = args.gae_coeff
        self.len_replay_buffer = args.len_replay_buffer
        self.model_cfg = args.model

        # for model
        self.actor = ActorGaussian(
            self.device, self.obs_dim, self.action_dim, self.action_bound_min, 
            self.action_bound_max, self.model_cfg['actor']).to(self.device)
        self.reward_critic = CriticS(self.device, self.obs_dim, self.model_cfg['reward_critic']).to(self.device)
        self.cost_critic = CriticS(self.device, self.obs_dim, self.model_cfg['cost_critic']).to(self.device)
        self.cost_std_critic = CriticS(self.device, self.obs_dim, self.model_cfg['cost_std_critic']).to(self.device)

        # for replay buffer
        self.replay_buffer = ReplayBuffer(
            self.obs_dim, self.action_dim, self.len_replay_buffer, self.discount_factor, 
            self.gae_coeff, self.n_envs, self.device, self.n_steps, self.n_update_steps)

        # for constraint
        self.con_alpha = args.con_alpha
        self.con_sigma = norm.pdf(norm.ppf(self.con_alpha))/self.con_alpha
        self.con_threshold = args.con_threshold/(1.0 - self.discount_factor)

        # for actor optimizer
        self.damping_coeff = args.damping_coeff
        self.num_conjugate = args.num_conjugate
        self.line_decay = args.line_decay
        self.max_kl = args.max_kl
        self.actor_optimizer = ConTROptimizer(
            self.actor, self.damping_coeff, self.num_conjugate, self.line_decay, 
            self.max_kl, self.con_threshold, self.device)

        # for critic optimizers
        self.reward_critic_optimizer = torch.optim.Adam(self.reward_critic.parameters(), lr=self.critic_lr)
        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=self.critic_lr)
        self.cost_std_critic_optimizer = torch.optim.Adam(self.cost_std_critic.parameters(), lr=self.critic_lr)


    """ public functions
    """
    def getAction(self, state:torch.Tensor, deterministic:bool) -> torch.Tensor:
        action_shape = state.shape[:-1] + (self.action_dim,)
        ε = torch.randn(action_shape, device=self.device)
        self.actor.updateActionDist(state, ε)

        norm_action, unnorm_action = self.actor.sample(deterministic)
        log_prob = self.actor.getLogProb()
        mean, std = self.actor.getMeanStd()

        self.state = state.detach().cpu().numpy()
        self.action = norm_action.detach().cpu().numpy()
        self.log_prob = log_prob.detach().cpu().numpy()
        self.mean = mean.detach().cpu().numpy()
        self.std = std.detach().cpu().numpy()
        return unnorm_action

    def step(self, rewards, costs, dones, fails, next_states):
        self.replay_buffer.addTransition(self.state, self.action, self.mean, self.std, self.log_prob, \
                            rewards, costs, dones, fails, next_states)

    def train(self):
        # get batches
        with torch.no_grad():
            states_tensor, actions_tensor, mu_means_tensor, mu_stds_tensor, reward_targets_tensor, cost_targets_tensor, cost_std_targets_tensor, \
                reward_gaes_tensor, cost_gaes_tensor, cost_var_gaes_tensor, cost_mean, cost_var_mean = \
                    self.replay_buffer.getBatches(self.actor, self.reward_critic, self.cost_critic, self.cost_std_critic)

        # ================== Critic Update ================== #
        for _ in range(self.n_critic_iters):
            reward_critic_loss = self.reward_critic.getLoss(states_tensor, reward_targets_tensor)
            self.reward_critic_optimizer.zero_grad()
            reward_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward_critic.parameters(), self.max_grad_norm)
            self.reward_critic_optimizer.step()

            cost_critic_loss = self.cost_critic.getLoss(states_tensor, cost_targets_tensor)
            self.cost_critic_optimizer.zero_grad()
            cost_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.max_grad_norm)
            self.cost_critic_optimizer.step()

            cost_std_critic_loss = self.cost_std_critic.getLoss(states_tensor, cost_std_targets_tensor)
            self.cost_std_critic_optimizer.zero_grad()
            cost_std_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.cost_std_critic.parameters(), self.max_grad_norm)
            self.cost_std_critic_optimizer.step()
        # ================================================== #

        # ================= Policy Update ================= #
        # backup old policy
        with torch.no_grad():
            ε = torch.normal(mean=torch.zeros_like(actions_tensor), std=torch.ones_like(actions_tensor))
            self.actor.updateActionDist(states_tensor, ε)
            entropy = self.actor.getEntropy()
            old_action_dists = self.actor.getDist()
            mu_action_dists = torch.distributions.Normal(mu_means_tensor, mu_stds_tensor)
            mu_log_probs_tensor = torch.sum(mu_action_dists.log_prob(actions_tensor), dim=-1)
            old_log_probs_tensor = torch.sum(old_action_dists.log_prob(actions_tensor), dim=-1)
            old_prob_ratios = torch.clamp(torch.exp(old_log_probs_tensor - mu_log_probs_tensor), 0.0, 1.0)
            mu_kl = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(mu_action_dists, old_action_dists), dim=-1)).item()

        def get_obj_con_kl():
            self.actor.updateActionDist(states_tensor, ε)
            cur_action_dists = self.actor.getDist()

            kl = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(old_action_dists, cur_action_dists), dim=-1))

            cur_log_probs = torch.sum(cur_action_dists.log_prob(actions_tensor), dim=-1)
            old_log_probs = torch.sum(old_action_dists.log_prob(actions_tensor), dim=-1)
            prob_ratios = torch.exp(cur_log_probs - old_log_probs)
            reward_gaes_mean = torch.mean(reward_gaes_tensor*old_prob_ratios)
            reward_gaes_std = torch.std(reward_gaes_tensor*old_prob_ratios)
            objective = torch.mean(prob_ratios*(reward_gaes_tensor*old_prob_ratios - reward_gaes_mean)/(reward_gaes_std + EPS))

            cost_gaes_mean = torch.mean(cost_gaes_tensor*old_prob_ratios)
            cost_var_gaes_mean = torch.mean(cost_var_gaes_tensor*old_prob_ratios)
            approx_cost_mean = cost_mean + (1.0/(1.0 - self.discount_factor))*torch.mean(prob_ratios*(cost_gaes_tensor*old_prob_ratios - cost_gaes_mean))
            approx_cost_var = cost_var_mean + (1.0/(1.0 - self.discount_factor**2))*torch.mean(prob_ratios*(cost_var_gaes_tensor*old_prob_ratios - cost_var_gaes_mean))
            constraint = approx_cost_mean + self.con_sigma*torch.sqrt(torch.clamp(approx_cost_var, EPS, np.inf))
            return objective, constraint, kl

        objective, constraint, kl, max_kl, beta, optim_case = self.actor_optimizer.step(get_obj_con_kl, mu_kl)
        # ================================================= #

        # return
        train_results = {
            'objective': objective,
            'constraint': constraint,
            'reward_critic_loss': reward_critic_loss.item(),
            'cost_critic_loss': cost_critic_loss.item(),
            'cost_std_critic_loss': cost_std_critic_loss.item(),
            'entropy': entropy.item(),
            'kl': kl,
            'max_kl': max_kl,
            'beta': beta,
            'optim_case': optim_case,
        }
        return train_results

    def save(self, model_num):
        save_dict = {
            'actor': self.actor.state_dict(),
            'reward_critic': self.reward_critic.state_dict(),
            'cost_critic': self.cost_critic.state_dict(),
            'cost_std_critic': self.cost_std_critic.state_dict(),
            'reward_critic_optimizer': self.reward_critic_optimizer.state_dict(),            
            'cost_critic_optimizer': self.cost_critic_optimizer.state_dict(),            
            'cost_std_critic_optimizer': self.cost_std_critic_optimizer.state_dict(),            
        }
        torch.save(save_dict, f"{self.checkpoint_dir}/model_{model_num}.pt")
        cprint(f'[{self.name}] save success.', bold=True, color="blue")

    def load(self, model_num):
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = f"{self.checkpoint_dir}/model_{model_num}.pt"
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            self.reward_critic.load_state_dict(checkpoint['reward_critic'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic'])
            self.cost_std_critic.load_state_dict(checkpoint['cost_std_critic'])
            self.actor.load_state_dict(checkpoint['actor'])
            self.reward_critic_optimizer.load_state_dict(checkpoint['reward_critic_optimizer'])
            self.cost_critic_optimizer.load_state_dict(checkpoint['cost_critic_optimizer'])
            self.cost_std_critic_optimizer.load_state_dict(checkpoint['cost_std_critic_optimizer'])
            cprint(f'[{self.name}] load success.', bold=True, color="blue")
            return int(model_num)
        else:
            self.actor.initialize()
            self.reward_critic.initialize()
            self.cost_critic.initialize()
            self.cost_std_critic.initialize()
            cprint(f'[{self.name}] load fail.', bold=True, color="red")
            return 0
