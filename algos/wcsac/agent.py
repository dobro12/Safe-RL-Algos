from algos.common.actor_squash import ActorSquash
from algos.common.critic_base import CriticSA
from algos.common.agent_base import AgentBase

from algos.wcsac.storage import ReplayBuffer
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

        # for RL
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.batch_size = args.batch_size
        self.n_update_iters = args.n_update_iters
        self.max_grad_norm = args.max_grad_norm
        self.len_replay_buffer = args.len_replay_buffer
        self.soft_update_ratio = args.soft_update_ratio
        self.model_cfg = args.model

        # for model
        self.actor = ActorSquash(
            self.device, self.obs_dim, self.action_dim, self.action_bound_min, 
            self.action_bound_max, self.model_cfg['actor']).to(self.device)
        self.reward_critic1 = CriticSA(
            self.device, self.obs_dim, self.action_dim, self.model_cfg['reward_critic']).to(self.device)
        self.reward_critic2 = CriticSA(
            self.device, self.obs_dim, self.action_dim, self.model_cfg['reward_critic']).to(self.device)
        self.reward_critic_target1 = CriticSA(
            self.device, self.obs_dim, self.action_dim, self.model_cfg['reward_critic']).to(self.device)
        self.reward_critic_target2 = CriticSA(
            self.device, self.obs_dim, self.action_dim, self.model_cfg['reward_critic']).to(self.device)
        self.cost_critic = CriticSA(
            self.device, self.obs_dim, self.action_dim, self.model_cfg['cost_critic']).to(self.device)
        self.cost_critic_target = CriticSA(
            self.device, self.obs_dim, self.action_dim, self.model_cfg['cost_critic']).to(self.device)
        self.cost_std_critic = CriticSA(
            self.device, self.obs_dim, self.action_dim, self.model_cfg['cost_std_critic']).to(self.device)
        self.cost_std_critic_target = CriticSA(
            self.device, self.obs_dim, self.action_dim, self.model_cfg['cost_std_critic']).to(self.device)

        # for entropy
        self.target_entropy = self.action_dim*args.entropy_threshold
        self.soft_entropy_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.getEntropyAlpha = lambda: torch.nn.functional.softplus(self.soft_entropy_alpha)
        self.ent_alpha_lr = args.ent_alpha_lr

        # for replay buffer
        self.replay_buffer = ReplayBuffer(self.len_replay_buffer, self.batch_size, self.device)

        # for constraint
        self.con_damp = args.con_damp
        self.con_alpha = args.con_alpha
        self.con_sigma = norm.pdf(norm.ppf(self.con_alpha))/self.con_alpha
        self.con_threshold = args.con_threshold/(1.0 - self.discount_factor)
        self.soft_con_beta = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.getConBeta = lambda: torch.nn.functional.softplus(self.soft_con_beta)
        self.con_beta_lr = args.con_beta_lr

        # for optimizers
        self.actor_params = list(self.actor.parameters())
        self.reward_critic_params = list(self.reward_critic1.parameters()) + list(self.reward_critic2.parameters())
        self.cost_critic_params = list(self.cost_critic.parameters())
        self.cost_std_critic_params = list(self.cost_std_critic.parameters())
        self.actor_optimizer = torch.optim.Adam(self.actor_params, lr=self.actor_lr)
        self.reward_critic_optimizer = torch.optim.Adam(self.reward_critic_params, lr=self.critic_lr)
        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic_params, lr=self.critic_lr)
        self.cost_std_critic_optimizer = torch.optim.Adam(self.cost_std_critic_params, lr=self.critic_lr)
        self.entropy_alpha_optimizer = torch.optim.Adam([self.soft_entropy_alpha], lr=self.ent_alpha_lr)
        self.con_beta_optimizer = torch.optim.Adam([self.soft_con_beta], lr=self.con_beta_lr)

    """ public functions
    """
    def getAction(self, state:torch.Tensor, deterministic:bool) -> torch.Tensor:
        action_shape = state.shape[:-1] + (self.action_dim,)
        ε = torch.randn(action_shape, device=self.device)
        self.actor.updateActionDist(state, ε)
        norm_action, unnorm_action = self.actor.sample(deterministic)
        self.state = state.detach().cpu().numpy()
        self.action = norm_action.detach().cpu().numpy()
        return unnorm_action

    def step(self, rewards, costs, dones, fails, next_states):
        self.replay_buffer.addTransition(self.state, self.action, rewards, costs, dones, fails, next_states)

    def train(self):
        if self.replay_buffer.getLen() < self.batch_size:
            results = {
                'reward_critic_loss': 0.0,
                'cost_critic_loss': 0.0,
                'cost_std_critic_loss': 0.0,
                'policy_loss': 0.0,
                'constraint': 0.0,
                'entropy': 0.0,
                'ent_alpha': 0.0,
                'con_beta': 0.0,
            }
        else:
            for _ in range(self.n_update_iters):
                reward_critic_loss, cost_critic_loss, cost_std_critic_loss, policy_loss, \
                    constraint_tensor, entropy, entropy_alpha_tensor, con_beta_tensor = self._train()
            results = {
                'reward_critic_loss': reward_critic_loss.item(),
                'cost_critic_loss': cost_critic_loss.item(),
                'cost_std_critic_loss': cost_std_critic_loss.item(),
                'policy_loss': policy_loss.item(),
                'constraint': constraint_tensor.item(),
                'entropy': entropy.item(),
                'ent_alpha': entropy_alpha_tensor.item(),
                'con_beta': con_beta_tensor.item(),
            }
        return results

    def save(self, model_num):
        save_dict = {
            'actor': self.actor.state_dict(),
            'reward_critic1': self.reward_critic1.state_dict(),
            'reward_critic2': self.reward_critic2.state_dict(),
            'reward_critic_target1': self.reward_critic_target1.state_dict(),
            'reward_critic_target2': self.reward_critic_target2.state_dict(),
            'cost_critic': self.cost_critic.state_dict(),
            'cost_critic_target': self.cost_critic_target.state_dict(),
            'cost_std_critic': self.cost_std_critic.state_dict(),
            'cost_std_critic_target': self.cost_std_critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'reward_critic_optimizer': self.reward_critic_optimizer.state_dict(),            
            'cost_critic_optimizer': self.cost_critic_optimizer.state_dict(),            
            'cost_std_critic_optimizer': self.cost_std_critic_optimizer.state_dict(),
            'soft_entropy_alpha': self.soft_entropy_alpha.data,
            'soft_con_beta': self.soft_con_beta.data,
            'entropy_alpha_optimizer': self.entropy_alpha_optimizer.state_dict(),
            'con_beta_optimizer': self.con_beta_optimizer.state_dict(),
        }
        torch.save(save_dict, f"{self.checkpoint_dir}/model_{model_num}.pt")
        cprint(f'[{self.name}] save success.', bold=True, color="blue")

    def load(self, model_num):
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = f"{self.checkpoint_dir}/model_{model_num}.pt"
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            self.actor.load_state_dict(checkpoint['actor'])
            self.reward_critic1.load_state_dict(checkpoint['reward_critic1'])
            self.reward_critic2.load_state_dict(checkpoint['reward_critic2'])
            self.reward_critic_target1.load_state_dict(checkpoint['reward_critic_target1'])
            self.reward_critic_target2.load_state_dict(checkpoint['reward_critic_target2'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target'])
            self.cost_std_critic.load_state_dict(checkpoint['cost_std_critic'])
            self.cost_std_critic_target.load_state_dict(checkpoint['cost_std_critic_target'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.reward_critic_optimizer.load_state_dict(checkpoint['reward_critic_optimizer'])
            self.cost_critic_optimizer.load_state_dict(checkpoint['cost_critic_optimizer'])
            self.cost_std_critic_optimizer.load_state_dict(checkpoint['cost_std_critic_optimizer'])
            self.soft_entropy_alpha.data = checkpoint['soft_entropy_alpha']
            self.soft_con_beta.data = checkpoint['soft_con_beta']
            self.entropy_alpha_optimizer.load_state_dict(checkpoint['entropy_alpha_optimizer'])
            self.con_beta_optimizer.load_state_dict(checkpoint['con_beta_optimizer'])
            cprint(f'[{self.name}] load success.', bold=True, color="blue")
            return int(model_num)
        else:
            self.actor.initialize()
            self.reward_critic1.initialize()
            self.reward_critic2.initialize()
            self.cost_critic.initialize()
            self.cost_std_critic.initialize()
            self._softUpdate(self.reward_critic_target1, self.reward_critic1, 0.0)
            self._softUpdate(self.reward_critic_target2, self.reward_critic2, 0.0)
            self._softUpdate(self.cost_critic_target, self.cost_critic, 0.0)
            self._softUpdate(self.cost_std_critic_target, self.cost_std_critic, 0.0)
            cprint(f'[{self.name}] load fail.', bold=True, color="red")
            return 0

    """ private functions
    """
    def _softUpdate(self, target, source, polyak):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * polyak + param.data * (1.0 - polyak))

    def _train(self):
        # get batches
        with torch.no_grad():
            states_tensor, actions_tensor, rewards_tensor, costs_tensor, \
                fails_tensor, next_states_tensor = self.replay_buffer.getBatches()

        # ================== Critic Update ================== #
        # calculate critic targets
        with torch.no_grad():
            entropy_alpha_tensor = self.getEntropyAlpha()
            epsilons_tensor = torch.randn_like(actions_tensor)
            self.actor.updateActionDist(next_states_tensor, epsilons_tensor)
            next_actions_tensor = self.actor.sample(deterministic=False)[0]
            next_log_probs_tensor = self.actor.getLogProb()
            next_reward_values1_tensor = self.reward_critic_target1(next_states_tensor, next_actions_tensor)
            next_reward_values2_tensor = self.reward_critic_target2(next_states_tensor, next_actions_tensor)
            next_cost_values_tensor = self.cost_critic_target(next_states_tensor, next_actions_tensor)
            next_cost_std_values_tensor = self.cost_std_critic_target(next_states_tensor, next_actions_tensor)
            reward_targets_tensor = rewards_tensor + self.discount_factor * (1.0 - fails_tensor) * \
                (torch.min(next_reward_values1_tensor, next_reward_values2_tensor) - entropy_alpha_tensor * next_log_probs_tensor)
            cost_targets_tensor = costs_tensor + self.discount_factor * (1.0 - fails_tensor) * next_cost_values_tensor
            cost_var_targets_tensor = torch.square(costs_tensor + self.discount_factor * (1.0 - fails_tensor) * next_cost_values_tensor) \
                        + torch.square(self.discount_factor * (1.0 - fails_tensor) * next_cost_std_values_tensor) \
                        - torch.square(self.cost_critic(states_tensor, actions_tensor))
            cost_std_targets_tensor = torch.sqrt(torch.clamp(cost_var_targets_tensor, 0.0, np.inf))

        # reward critic update
        reward_critic_loss = self.reward_critic1.getLoss(states_tensor, actions_tensor, reward_targets_tensor) \
                            + self.reward_critic2.getLoss(states_tensor, actions_tensor, reward_targets_tensor)
        self.reward_critic_optimizer.zero_grad()
        reward_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.reward_critic_params, self.max_grad_norm)
        self.reward_critic_optimizer.step()

        # cost critic update
        cost_critic_loss = self.cost_critic.getLoss(states_tensor, actions_tensor, cost_targets_tensor)
        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cost_critic_params, self.max_grad_norm)
        self.cost_critic_optimizer.step()

        # cost std critic update
        cost_std_critic_loss = self.cost_std_critic.getLoss(states_tensor, actions_tensor, cost_std_targets_tensor)
        self.cost_std_critic_optimizer.zero_grad()
        cost_std_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cost_std_critic_params, self.max_grad_norm)
        self.cost_std_critic_optimizer.step()

        # soft update
        self._softUpdate(self.reward_critic_target1, self.reward_critic1, self.soft_update_ratio)
        self._softUpdate(self.reward_critic_target2, self.reward_critic2, self.soft_update_ratio)
        self._softUpdate(self.cost_critic_target, self.cost_critic, self.soft_update_ratio)
        self._softUpdate(self.cost_std_critic_target, self.cost_std_critic, self.soft_update_ratio)
        # ================================================== #

        # ================= Policy Update ================= #
        with torch.no_grad():
            entropy_alpha_tensor = self.getEntropyAlpha()
            con_beta_tensor = self.getConBeta()
            epsilons_tensor = torch.randn_like(actions_tensor)
            constraint_tensor = torch.mean(self.cost_critic(states_tensor, actions_tensor) \
                                    + self.con_sigma*self.cost_std_critic(states_tensor, actions_tensor))

        self.actor.updateActionDist(states_tensor, epsilons_tensor)
        sampled_actions_tensor = self.actor.sample(deterministic=False)[0]
        log_probs_tensor = self.actor.getLogProb()
        min_reward_values_tensor = torch.min(
            self.reward_critic1(states_tensor, sampled_actions_tensor), \
            self.reward_critic2(states_tensor, sampled_actions_tensor))
        damp_tensor = self.con_damp*(self.con_threshold - constraint_tensor)
        policy_loss = torch.mean(entropy_alpha_tensor*log_probs_tensor - min_reward_values_tensor \
                                + (con_beta_tensor - damp_tensor)*(self.cost_critic(states_tensor, sampled_actions_tensor) \
                                + self.con_sigma*self.cost_std_critic(states_tensor, sampled_actions_tensor)))
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm)
        self.actor_optimizer.step()
        # ================================================= #

        # entropy alpha update
        entropy_alpha_tensor = self.getEntropyAlpha()
        entropy = -torch.mean(log_probs_tensor)
        entropy_alpha_loss = torch.mean(entropy_alpha_tensor*(entropy - self.target_entropy).detach())
        self.entropy_alpha_optimizer.zero_grad()
        entropy_alpha_loss.backward()
        self.entropy_alpha_optimizer.step()

        # con beta update
        con_beta_tensor = self.getConBeta()
        con_beta_loss = torch.mean(con_beta_tensor*(self.con_threshold - constraint_tensor).detach())
        self.con_beta_optimizer.zero_grad()
        con_beta_loss.backward()
        self.con_beta_optimizer.step()

        return reward_critic_loss, cost_critic_loss, cost_std_critic_loss, policy_loss, \
            constraint_tensor, entropy, entropy_alpha_tensor, con_beta_tensor