from algos.offtrc_dirichlet.storage import ReplayBuffer
from algos.offtrc_dirichlet.critic import Critic
from algos.offtrc_dirichlet.actor import Actor
from algos.offtrc.optimizer import TROptimizer, ConTROptimizer
from algos.common.agent_base import AgentBase
from utils import cprint

from scipy.stats import norm
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
        self.n_assets = args.n_assets
        self.feature_dim = args.feature_dim
        self.n_steps = args.n_steps
        self.n_update_steps = args.n_update_steps

        # for RL
        self.critic_lr = args.critic_lr
        self.n_critic_iters = args.n_critic_iters
        self.max_grad_norm = args.max_grad_norm
        self.gae_coeff = args.gae_coeff
        self.len_replay_buffer = args.len_replay_buffer
        self.model_cfg = args.model
        self.ent_coeff = args.ent_coeff

        # for trust region
        self.damping_coeff = args.damping_coeff
        self.num_conjugate = args.num_conjugate
        self.line_decay = args.line_decay
        self.max_kl = args.max_kl

        # for model
        self.actor = Actor(
            self.device, self.n_assets, self.feature_dim, self.model_cfg['actor']
        ).to(self.device)
        self.reward_critic = Critic(
            self.device, self.n_assets, self.feature_dim, self.model_cfg['reward_critic']
        ).to(self.device)
        self.cost_critic = Critic(
            self.device, self.n_assets, self.feature_dim, self.model_cfg['cost_critic']
        ).to(self.device)
        self.cost_std_critic = Critic(
            self.device, self.n_assets, self.feature_dim, self.model_cfg['cost_std_critic']
        ).to(self.device)

        # for replay buffer
        self.replay_buffer = ReplayBuffer(
            self.device, self.len_replay_buffer, self.discount_factor, 
            self.gae_coeff, self.n_steps, self.n_update_steps)

        # for constraint
        # self.con_alpha = args.con_alpha
        # self.con_sigma = norm.pdf(norm.ppf(self.con_alpha))/self.con_alpha
        # self.con_threshold = args.con_threshold/(1.0 - self.discount_factor)

        # for actor optimizer
        self.damping_coeff = args.damping_coeff
        self.num_conjugate = args.num_conjugate
        self.line_decay = args.line_decay
        self.max_kl = args.max_kl
        self.actor_optimizer = TROptimizer(
            self.actor, self.damping_coeff, self.num_conjugate, self.line_decay, 
            self.max_kl, self.device)

        # for critic optimizers
        self.reward_critic_optimizer = torch.optim.Adam(
            self.reward_critic.parameters(), lr=self.critic_lr)
        self.cost_critic_optimizer = torch.optim.Adam(
            self.cost_critic.parameters(), lr=self.critic_lr)
        self.cost_std_critic_optimizer = torch.optim.Adam(
            self.cost_std_critic.parameters(), lr=self.critic_lr)

    """ public functions
    """
    def getAction(self, state:torch.Tensor) -> torch.Tensor:
        self.actor.updateActionDist(state)

        action = self.actor.sample()
        log_prob = self.actor.getLogProb()
        concentraion = self.actor.getConcentration()

        self.state = state.detach().cpu().numpy()
        self.action = action.detach().cpu().numpy()
        self.log_prob = log_prob.detach().cpu().numpy()
        self.concentraion = concentraion.detach().cpu().numpy()
        return action

    def step(self, reward, cost, done, next_state):
        self.replay_buffer.addTransition(
            self.state, self.action, self.concentraion, 
            self.log_prob, reward, cost, done, next_state)

    def train(self):
        # get batches
        with torch.no_grad():
            states_tensor, actions_tensor, concentrations_tensor, reward_targets_tensor, \
            cost_targets_tensor, cost_std_targets_tensor, reward_gaes_tensor, cost_gaes_tensor, \
            cost_var_gaes_tensor, cost_mean, cost_var_mean = self.replay_buffer.getBatches(
                self.actor, self.reward_critic, self.cost_critic, self.cost_std_critic)

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
            self.actor.updateActionDist(states_tensor)
            entropy = self.actor.getEntropy()
            old_action_dists = self.actor.getDist()
            mu_action_dists = torch.distributions.Dirichlet(concentrations_tensor)
            mu_log_probs_tensor = mu_action_dists.log_prob(actions_tensor)
            old_log_probs_tensor = old_action_dists.log_prob(actions_tensor)
            old_prob_ratios = torch.clamp(torch.exp(old_log_probs_tensor - mu_log_probs_tensor), 0.0, 1.0)
            mu_kl = torch.mean(torch.distributions.kl.kl_divergence(mu_action_dists, old_action_dists)).item()

        # def get_obj_con_kl():
        def get_obj_kl():
            self.actor.updateActionDist(states_tensor)
            cur_action_dists = self.actor.getDist()
            entropy = self.actor.getEntropy()

            kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, cur_action_dists))

            cur_log_probs = cur_action_dists.log_prob(actions_tensor)
            old_log_probs = old_action_dists.log_prob(actions_tensor)
            prob_ratios = torch.exp(cur_log_probs - old_log_probs)
            reward_gaes_mean = torch.mean(reward_gaes_tensor*old_prob_ratios)
            reward_gaes_std = torch.std(reward_gaes_tensor*old_prob_ratios)
            objective = torch.mean(prob_ratios*(reward_gaes_tensor*old_prob_ratios - reward_gaes_mean)/(reward_gaes_std + EPS)) + self.ent_coeff*entropy

            # cost_gaes_mean = torch.mean(cost_gaes_tensor*old_prob_ratios)
            # cost_var_gaes_mean = torch.mean(cost_var_gaes_tensor*old_prob_ratios)
            # approx_cost_mean = cost_mean + (1.0/(1.0 - self.discount_factor))*torch.mean(prob_ratios*(cost_gaes_tensor*old_prob_ratios - cost_gaes_mean))
            # approx_cost_var = cost_var_mean + (1.0/(1.0 - self.discount_factor**2))*torch.mean(prob_ratios*(cost_var_gaes_tensor*old_prob_ratios - cost_var_gaes_mean))
            # constraint = approx_cost_mean + self.con_sigma*torch.sqrt(torch.clamp(approx_cost_var, EPS, np.inf))
            # return objective, constraint, kl
            return objective, kl

        objective, kl, max_kl, beta = self.actor_optimizer.step(get_obj_kl, mu_kl)
        # ================================================= #

        # return
        train_results = {
            'objective': objective,
            'reward_critic_loss': reward_critic_loss.item(),
            'cost_critic_loss': cost_critic_loss.item(),
            'cost_std_critic_loss': cost_std_critic_loss.item(),
            'entropy': entropy.item(),
            'kl': kl,
            'total_kl': kl + np.sqrt(mu_kl*(self.max_kl + 0.25*mu_kl)) - 0.5*mu_kl,
            'max_kl': max_kl,
            'beta': beta,
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
