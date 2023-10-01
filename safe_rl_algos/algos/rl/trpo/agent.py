from safe_rl_algos.algos.common.optimizer_base import TROptimizer as ActorOptimizer
from safe_rl_algos.algos.common.actor_gaussian import ActorGaussian as Actor
from safe_rl_algos.algos.common.critic_base import CriticS as Critic
from safe_rl_algos.algos.common.agent_base import AgentBase
from safe_rl_algos.utils import cprint

from .storage import RolloutBuffer

from typing import Tuple
import numpy as np
import torch
import os

EPS = 1e-8

class Agent(AgentBase):
    def __init__(self, args) -> None:
        super().__init__(
            name=args.name,
            device=args.device,
            obs_dim=args.obs_dim,
            action_dim=args.action_dim,
            norm_obs=args.norm_obs,
            norm_reward=args.norm_reward,
        )

        # base
        self.save_dir = args.save_dir
        self.checkpoint_dir=f'{self.save_dir}/checkpoint'
        self.action_bound_min = args.action_bound_min
        self.action_bound_max = args.action_bound_max
        self.n_steps = args.n_steps
        self.n_envs = args.n_envs

        # for RL
        self.discount_factor = args.discount_factor
        self.critic_lr = args.critic_lr
        self.n_critic_iters = args.n_critic_iters
        self.max_grad_norm = args.max_grad_norm
        self.gae_coeff = args.gae_coeff
        self.model_cfg = args.model

        # for trust region
        self.max_kl = args.max_kl
        self.damping_coeff = args.damping_coeff
        self.num_conjugate = args.num_conjugate
        self.line_decay = args.line_decay
        self.kl_tolerance = args.kl_tolerance

        # for model
        self.actor = Actor(
            self.device, self.obs_dim, self.action_dim, self.action_bound_min, 
            self.action_bound_max, self.model_cfg['actor']).to(self.device)
        self.reward_critic = Critic(
            self.device, self.obs_dim, self.model_cfg['reward_critic']).to(self.device)

        # for buffer
        self.replay_buffer = RolloutBuffer(
            self.device, self.obs_dim, self.action_dim, self.discount_factor, 
            self.gae_coeff, self.n_envs, self.n_steps)

        # for optimizers
        self.actor_optimizer = ActorOptimizer(
            self.device, self.actor, self.damping_coeff, self.num_conjugate, 
            self.line_decay, self.max_kl, self.kl_tolerance)
        self.reward_critic_optimizer = torch.optim.Adam(self.reward_critic.parameters(), lr=self.critic_lr)

    ################
    # Public Methods
    ################

    @torch.no_grad()
    def getAction(self, state:np.ndarray, deterministic:bool) -> np.ndarray:
        state_tensor = torch.tensor(self.obs_rms.normalize(state), dtype=torch.float32, device=self.device)
        epsilon_tensor = torch.randn(state_tensor.shape[:-1] + (self.action_dim,), device=self.device)

        self.actor.updateActionDist(state_tensor, epsilon_tensor)
        norm_action_tensor, unnorm_action_tensor = self.actor.sample(deterministic)

        self.state = state.copy()
        self.action = norm_action_tensor.detach().cpu().numpy()
        return unnorm_action_tensor.detach().cpu().numpy()

    def step(self, rewards, dones, fails, next_states):
        self.replay_buffer.addTransition(self.state, self.action, rewards, dones, fails, next_states)

        # update statistics
        if self.norm_obs:
            self.obs_rms.update(self.state)
        if self.norm_reward:
            self.reward_rms.update(rewards)

    def readyToTrain(self):
        return self.replay_buffer.getLen() > 0

    def train(self):
        # get batches
        with torch.no_grad():
            states_tensor, actions_tensor, reward_targets_tensor, reward_gaes_tensor = \
                self.replay_buffer.getBatches(self.obs_rms, self.reward_rms, self.reward_critic)

        # ================== Critic Update ================== #
        for _ in range(self.n_critic_iters):
            reward_critic_loss = self.reward_critic.getLoss(states_tensor, reward_targets_tensor)
            self.reward_critic_optimizer.zero_grad()
            reward_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward_critic.parameters(), self.max_grad_norm)
            self.reward_critic_optimizer.step()
        # ================================================== #

        # ================= Policy Update ================= #
        # backup old policy
        with torch.no_grad():
            reward_gaes_tensor = (reward_gaes_tensor - reward_gaes_tensor.mean())/(reward_gaes_tensor.std() + EPS)
            epsilons_tensor = torch.randn_like(actions_tensor)
            self.actor.updateActionDist(states_tensor, epsilons_tensor)
            old_action_dists = self.actor.getDist()
            old_log_probs_tensor = torch.sum(old_action_dists.log_prob(actions_tensor), dim=-1)
            entropy = self.actor.getEntropy()

        # update policy
        def get_obj_kl():
            self.actor.updateActionDist(states_tensor, epsilons_tensor)
            cur_action_dists = self.actor.getDist()
            cur_log_probs_tensor = torch.sum(cur_action_dists.log_prob(actions_tensor), dim=-1)
            kl = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(old_action_dists, cur_action_dists), dim=-1))
            obj = torch.mean(torch.exp(cur_log_probs_tensor - old_log_probs_tensor)*reward_gaes_tensor)
            return obj, kl
        objective, kl, max_kl, beta = self.actor_optimizer.step(get_obj_kl)
        # ================================================= #

        # return
        train_results = {
            'reward_critic_loss': reward_critic_loss.item(),
            'entropy': entropy.item(),
            'objective': objective,
            'kl': kl,
            'max_kl': max_kl,
            'beta': beta,
        }
        return train_results


    def save(self, model_num):
        # save rms
        self.obs_rms.save(self.save_dir, model_num)
        self.reward_rms.save(self.save_dir, model_num)

        # save network models
        save_dict = {
            'actor': self.actor.state_dict(),
            'reward_critic': self.reward_critic.state_dict(),
            'reward_critic_optimizer': self.reward_critic_optimizer.state_dict(),            
        }
        torch.save(save_dict, f"{self.checkpoint_dir}/model_{model_num}.pt")
        cprint(f'[{self.name}] save success.', bold=True, color="blue")

    def load(self, model_num):
        # load rms
        self.obs_rms.load(self.save_dir, model_num)
        self.reward_rms.load(self.save_dir, model_num)

        # load network models
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = f"{self.checkpoint_dir}/model_{model_num}.pt"
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            self.actor.load_state_dict(checkpoint['actor'])
            self.reward_critic.load_state_dict(checkpoint['reward_critic'])
            self.reward_critic_optimizer.load_state_dict(checkpoint['reward_critic_optimizer'])
            cprint(f'[{self.name}] load success.', bold=True, color="blue")
            return int(model_num)
        else:
            self.actor.initialize()
            self.reward_critic.initialize()
            cprint(f'[{self.name}] load fail.', bold=True, color="red")
            return 0
