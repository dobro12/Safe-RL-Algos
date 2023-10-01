from safe_rl_algos.algos.common.optimizer_base import (
    TROptimizer, flatGrad, EPS
)

from typing import (
    Tuple, Iterator, Callable
)
import numpy as np
import torch

class ActorOptimizer(TROptimizer):
    def __init__(
            self, device:torch.device, 
            actor:torch.nn.Module, 
            damping_coeff:float, 
            num_conjugate:int, 
            line_decay:float, 
            max_kl:float,
            kl_tolerance:float,
            con_taus:torch.Tensor,
            con_thresholds:torch.Tensor) -> None:
        super().__init__(device, actor, damping_coeff, num_conjugate, line_decay, max_kl, kl_tolerance)
        self.con_taus = con_taus
        self.con_thresholds = con_thresholds
        self.cost_dim = con_taus.shape[0]
        self.thresh_tol = 0.1

    #################
    # public function
    #################

    def step(
            self, 
            get_obj_cons_kl:Callable[[], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], 
            mu_kl:float=0.0,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:

        # get objective, constraints, kl
        max_kl = self._getMaxKL(mu_kl)
        objective, constraints, kl = get_obj_cons_kl()
        self._computeKLGrad(kl)

        # check feasibility
        violate_idx = -1
        for con_idx in range(self.cost_dim):
            if constraints[con_idx] + self.thresh_tol > self.con_thresholds[con_idx]:
                violate_idx = con_idx
                break
        if violate_idx != -1:
            # for recovery mode
            objective = -constraints[violate_idx]
        else:
            # for normal mode
            objective = objective + torch.sum(torch.log(self.con_thresholds - constraints)/self.con_taus)

        # calculate search direction
        g_tensor = flatGrad(objective, self.actor.parameters(), retain_graph=True)
        H_inv_g_tensor = self._conjugateGradient(g_tensor)
        approx_g_tensor = self._Hx(H_inv_g_tensor)
        with torch.no_grad():
            g_H_inv_g_tensor = torch.dot(approx_g_tensor, H_inv_g_tensor)
            nu = torch.sqrt(2.0*max_kl/(g_H_inv_g_tensor + EPS))
            delta_theta = nu*H_inv_g_tensor

            # line search
            beta = 1.0
            init_theta = torch.cat([t.view(-1) for t in self.actor.parameters()]).clone().detach()
            init_objective = objective.clone().detach()
            while True:
                theta = beta*delta_theta + init_theta
                self._applyParams(theta)
                objective, constraints, kl = get_obj_cons_kl()
                if violate_idx != -1:
                    objective = -constraints[violate_idx]
                else:
                    objective = objective + torch.sum(torch.log(self.con_thresholds - constraints)/self.con_taus)
                if kl <= self.kl_tolerance*max_kl and objective >= init_objective:
                    break
                beta *= self.line_decay
        return objective, constraints,  kl, max_kl, beta, violate_idx