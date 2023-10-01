import numpy as np
import torch

from typing import (
    Tuple, Iterator, Callable
)

EPS = 1e-8

def flatGrad(
        y:torch.Tensor, 
        x:Iterator[torch.nn.parameter.Parameter], 
        retain_graph:bool=False, 
        create_graph:bool=False
    ) -> torch.Tensor:
    if create_graph:
        retain_graph = True
    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g


class TROptimizer:
    def __init__(
            self, device:torch.device, 
            actor:torch.nn.Module, 
            damping_coeff:float, 
            num_conjugate:int, 
            line_decay:float, 
            max_kl:float,
            kl_tolerance:float) -> None:

        # base
        self.device = device
        self.actor = actor
        self.damping_coeff = damping_coeff
        self.num_conjugate = num_conjugate
        self.line_decay = line_decay
        self.max_kl = max_kl
        self.kl_tolerance = kl_tolerance

        # count number of parameters
        self.n_params = 0
        for param in self.actor.parameters():
            self.n_params += param.shape.numel()

    #################
    # public function
    #################

    def step(
            self, 
            get_obj_kl:Callable[[], Tuple[torch.Tensor, torch.Tensor]], 
            mu_kl:float=0.0,
        ) -> Tuple[float, float, float, float]:

        # for adaptive kl
        max_kl = self._getMaxKL(mu_kl)

        # calculate gradient
        objective, kl = get_obj_kl()
        self._computeKLGrad(kl)
        g_tensor = flatGrad(objective, self.actor.parameters(), retain_graph=True)
        H_inv_g_tensor = self._conjugateGradient(g_tensor)
        approx_g_tensor = self._Hx(H_inv_g_tensor)

        with torch.no_grad():
            # calculate search direction
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
                objective, kl = get_obj_kl()
                if kl <= self.kl_tolerance*max_kl and objective >= init_objective:
                    break
                beta *= self.line_decay
        return objective.item(), kl.item(), max_kl, beta

    ##################
    # private function
    ##################

    def _getMaxKL(self, mu_kl:float=0.0) -> float:
        kl_bonus = np.sqrt(mu_kl*(self.max_kl + 0.25*mu_kl)) - 0.5*mu_kl
        max_kl = np.clip(self.max_kl - kl_bonus, 0.0, np.inf)
        return max_kl

    def _computeKLGrad(self, kl:torch.Tensor) -> None:
        self._flat_grad_kl = flatGrad(kl, self.actor.parameters(), create_graph=True)

    def _applyParams(self, params:torch.Tensor) -> None:
        n = 0
        for p in self.actor.parameters():
            numel = p.numel()
            p.data.copy_(params[n:n + numel].view(p.shape))
            n += numel

    def _Hx(self, x:torch.Tensor) -> torch.Tensor:
        kl_x = torch.dot(self._flat_grad_kl, x.detach())
        H_x = flatGrad(kl_x, self.actor.parameters(), retain_graph=True)
        return H_x + x*self.damping_coeff

    def _conjugateGradient(self, g:torch.Tensor) -> torch.Tensor:
        x = torch.zeros_like(g, device=self.device)
        r = g.clone()
        p = g.clone()
        rs_old = torch.sum(r*r)
        for i in range(self.num_conjugate):
            Ap = self._Hx(p)
            pAp = torch.sum(p*Ap)
            alpha = rs_old/(pAp + EPS)
            x += alpha*p
            r -= alpha*Ap
            rs_new = torch.sum(r*r)
            p = r + (rs_new/(rs_old + EPS))*p
            rs_old = rs_new
        return x
