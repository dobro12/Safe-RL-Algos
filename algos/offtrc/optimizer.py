from copy import deepcopy
import numpy as np
import torch
import time

EPS = 1e-8

def flatGrad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True
    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g


class TROptimizer:
    def __init__(self, actor, damping_coeff, num_conjugate, line_decay, max_kl, device) -> None:
        self.actor = actor
        self.damping_coeff = damping_coeff
        self.num_conjugate = num_conjugate
        self.line_decay = line_decay
        self.max_kl = max_kl
        self.device = device

        # count number of parameters
        self.n_params = 0
        for param in self.actor.parameters():
            self.n_params += param.shape.numel()

    #################
    # public function
    #################

    def step(self, get_obj_kl, mu_kl=0.0):
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
                if kl <= 1.5*max_kl and objective >= init_objective:
                    break
                beta *= self.line_decay
        return objective.item(), kl.item(), max_kl, beta

    ##################
    # private function
    ##################

    def _getMaxKL(self, mu_kl=0.0):
        kl_bonus = np.sqrt(mu_kl*(self.max_kl + 0.25*mu_kl)) - 0.5*mu_kl
        max_kl = np.clip(self.max_kl - kl_bonus, 0.0, np.inf)
        return max_kl

    def _computeKLGrad(self, kl):
        self._flat_grad_kl = flatGrad(kl, self.actor.parameters(), create_graph=True)

    def _applyParams(self, params):
        n = 0
        for p in self.actor.parameters():
            numel = p.numel()
            p.data.copy_(params[n:n + numel].view(p.shape))
            n += numel

    def _Hx(self, x):
        kl_x = torch.dot(self._flat_grad_kl, x)
        H_x = flatGrad(kl_x, self.actor.parameters(), retain_graph=True)
        return H_x + x*self.damping_coeff

    def _conjugateGradient(self, g):
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
    

class ConTROptimizer(TROptimizer):
    def __init__(self, actor, damping_coeff, num_conjugate, line_decay, max_kl, con_threshold, device) -> None:
        super().__init__(actor, damping_coeff, num_conjugate, line_decay, max_kl, device)
        self.con_threshold = con_threshold

    def step(self, get_obj_con_kl, mu_kl=0.0):
        # for adaptive kl
        max_kl = self._getMaxKL(mu_kl)

        # calculate gradient
        objective, constraint, kl = get_obj_con_kl()
        self._computeKLGrad(kl)
        g_tensor = flatGrad(objective, self.actor.parameters(), retain_graph=True)
        b_tensor = flatGrad(-constraint, self.actor.parameters(), retain_graph=True)
        H_inv_g_tensor = self._conjugateGradient(g_tensor)
        H_inv_b_tensor = self._conjugateGradient(b_tensor)
        approx_g_tensor = self._Hx(H_inv_g_tensor)
        approx_b_tensor = self._Hx(H_inv_b_tensor)
        c_value = constraint.item() - self.con_threshold

        with torch.no_grad():
            # ======== solve Lagrangian problem ======== #
            if torch.dot(b_tensor, b_tensor) <= 1e-8 and c_value < 0:
                scalar_q = torch.dot(approx_g_tensor, H_inv_g_tensor)
                optim_case = 4
            else:
                scalar_q = torch.dot(approx_g_tensor, H_inv_g_tensor)
                scalar_r = torch.dot(approx_g_tensor, H_inv_b_tensor)
                scalar_s = torch.dot(approx_b_tensor, H_inv_b_tensor)
                A_value = scalar_q - scalar_r**2/scalar_s
                B_value = 2*max_kl - c_value**2/scalar_s
                if c_value < 0 and B_value <= 0:
                    optim_case = 3
                elif c_value < 0 and B_value > 0:
                    optim_case = 2
                elif c_value >= 0 and B_value > 0:
                    optim_case = 1
                else:
                    optim_case = 0
            if optim_case in [3, 4]:
                lam = torch.sqrt(scalar_q/(2*max_kl))
                nu = 0
            elif optim_case in [1, 2]:
                LA, LB = [0, scalar_r/c_value], [scalar_r/c_value, np.inf]
                LA, LB = (LA, LB) if c_value < 0 else (LB, LA)
                proj = lambda x, L : max(L[0], min(L[1], x))
                lam_a = proj(torch.sqrt(A_value/B_value), LA)
                lam_b = proj(torch.sqrt(scalar_q/(2*max_kl)), LB)
                f_a = lambda lam : -0.5*(A_value/(lam + EPS) + B_value*lam) - scalar_r*c_value/(scalar_s + EPS)
                f_b = lambda lam : -0.5*(scalar_q/(lam + EPS) + 2*max_kl*lam)
                lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
                nu = max(0, lam*c_value - scalar_r)/(scalar_s + EPS)
            else:
                lam = 0
                nu = torch.sqrt(2*max_kl/(scalar_s + EPS))
            # ========================================== #

            # line search
            delta_theta = (1./(lam + EPS))*(H_inv_g_tensor + nu*H_inv_b_tensor) if optim_case > 0 else nu*H_inv_b_tensor
            beta = 1.0
            init_theta = torch.cat([t.view(-1) for t in self.actor.parameters()]).clone().detach()
            init_objective = objective.clone().detach()
            init_constraint = constraint.clone().detach()
            while True:
                theta = beta*delta_theta + init_theta
                self._applyParams(theta)
                objective, constraint, kl = get_obj_con_kl()
                if kl <= max_kl*1.5 and (objective >= init_objective if optim_case > 1 else True) and constraint - init_constraint <= max(-c_value, 0):
                    break
                beta *= self.line_decay

        return objective.item(), constraint.item(), kl.item(), max_kl, beta, optim_case
