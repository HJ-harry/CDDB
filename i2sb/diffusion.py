# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import numpy as np
from tqdm import tqdm
from functools import partial
import torch

from .util import unsqueeze_xdim

from ipdb import set_trace as debug
from i2sb.util import clear_color, clear
import matplotlib.pyplot as plt


def compute_gaussian_product_coef(sigma1, sigma2):
    """ Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
        return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var) """

    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var

class Diffusion():
    def __init__(self, betas, device):

        self.device = device

        # compute analytic std: eq 11
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        # tensorize everything
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = to_torch(betas).to(device)
        self.std_fwd = to_torch(std_fwd).to(device)
        self.std_bwd = to_torch(std_bwd).to(device)
        self.std_sb  = to_torch(std_sb).to(device)
        self.mu_x0 = to_torch(mu_x0).to(device)
        self.mu_x1 = to_torch(mu_x1).to(device)

    def get_std_fwd(self, step, xdim=None):
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def q_sample(self, step, x0, x1, ot_ode=False):
        """ Sample q(x_t | x_0, x_1), i.e. eq 11 """

        assert x0.shape == x1.shape
        batch, *xdim = x0.shape

        mu_x0  = unsqueeze_xdim(self.mu_x0[step],  xdim)
        mu_x1  = unsqueeze_xdim(self.mu_x1[step],  xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)

        xt = mu_x0 * x0 + mu_x1 * x1
        if not ot_ode:
            xt = xt + std_sb * torch.randn_like(xt)
        return xt.detach()


    def p_posterior(self, nprev, n, x_n, x0, ot_ode=False, verbose=False):
        """ Sample p(x_{nprev} | x_n, x_0), i.e. eq 4"""

        assert nprev < n
        std_n     = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not ot_ode and nprev > 0:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)

        if verbose:
            return xt_prev, mu_x0
        else:
            return xt_prev
    
    
    def p_posterior_ddim(self, nprev, n, x_n, x0, pred_eps, eta=1.0):
        """ Posterior sampling for ddim. OT-ODE disabled. """

        assert nprev < n
        std_n     = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        
        c1 = var.sqrt() * eta
        c2 = var.sqrt() * np.sqrt(1 - eta**2)
        
        xt_prev = xt_prev + c1 * torch.randn_like(xt_prev) + c2 * pred_eps

        return xt_prev
    

    def ddpm_sampling(self, steps, pred_x0_fn, x1, x1_pinv, x1_forw, mask=None, ot_ode=False, log_steps=None, verbose=True):
        xt = x1.detach().to(self.device)
        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
        cnt = 0
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            pred_x0 = pred_x0_fn(xt, step)
            xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)

            if mask is not None:
                
                # import matplotlib.pyplot as plt
                
                xt_true = x1
                if not ot_ode:
                    _prev_step = torch.full((xt.shape[0],), prev_step, device=self.device, dtype=torch.long)
                    std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=x1.shape[1:])
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                # if cnt % 10 == 0:
                #     plt.subplot(1,2,1)
                #     plt.imshow(clear_color(xt))
                xt = (1. - mask) * xt_true + mask * xt
                #     plt.subplot(1,2,2)
                #     plt.imshow(clear_color(xt))
                #     plt.show()
                    
                #     import ipdb; ipdb.set_trace()

            cnt += 1
            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)
    
    
    def ddnm_sampling(self, steps, pred_x0_fn, x1, x1_pinv, x1_forw, mask=None, 
                      corrupt_type=None, corrupt_method=None, step_size=1.0,
                      ot_ode=False, log_steps=None, verbose=True):
        xt = x1.detach().to(self.device)

        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"
            pred_x0 = pred_x0_fn(xt, step)
            
            # projection data consistency - useless for inpainting. Might as well drop
            if mask is not None:
                corrupt_x0_pinv, _ = corrupt_method(pred_x0)
            elif "jpeg" in corrupt_type or "sr" in corrupt_type:
                corrupt_x0_pinv, _ = corrupt_method(pred_x0)
            else:
                _, corrupt_x0_pinv = corrupt_method(pred_x0)
            pred_x0 = pred_x0 - corrupt_x0_pinv + x1_pinv
            
            xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)
            
            if mask is not None:
                xt_true = x1
                if not ot_ode:
                    _prev_step = torch.full((xt.shape[0],), prev_step, device=self.device, dtype=torch.long)
                    std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=x1.shape[1:])
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                xt = (1. - mask) * xt_true + mask * xt

            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)
    
    
    def ddpm_dps_sampling(self, steps, pred_x0_fn, x1, x1_pinv, x1_forw, 
                          mask=None, corrupt_type=None, corrupt_method=None, step_size=1.0,
                          ot_ode=False, log_steps=None, verbose=True, results_dir=None):
        xt = x1.detach().to(self.device)

        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]
        
        if results_dir is not None:
            for t in ["x0_before", "x0_after", "x0_diff", "x0_diff_mean"]:
                (results_dir / t).mkdir(exist_ok=True, parents=True)

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            # turn on gradient
            xt.requires_grad_()
            pred_x0 = pred_x0_fn(xt, step)
            
            # DPS
            # for inpainting, corrupt_method returns a tuple
            if mask is not None:
                corrupt_x0_forw, _ = corrupt_method(pred_x0)
            elif "jpeg" in corrupt_type or "sr" in corrupt_type:
                _, corrupt_x0_forw = corrupt_method(pred_x0)
            else:
                corrupt_x0_forw, _ = corrupt_method(pred_x0)
                
                
            residual = corrupt_x0_forw - x1_forw
            residual_norm = torch.linalg.norm(residual) ** 2
            # residual_norm = torch.linalg.norm(residual)
            norm_grad = torch.autograd.grad(outputs=residual_norm, inputs=xt)[0]
            
            xt, mu_x0 = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode, verbose=True)
            if results_dir is not None:
                plt.imsave(str(results_dir / "x0_before" / f"{step}.png"), clear_color(pred_x0))
            # xt = xt - mu_x0 * step_size * norm_grad
            pair_steps.set_postfix({"mu_x0": mu_x0.item()})
            xt = xt - step_size * norm_grad
            pred_x0_correct = pred_x0 - step_size * norm_grad
            if results_dir is not None:
                plt.imsave(str(results_dir / "x0_after" / f"{step}.png"), clear_color(pred_x0_correct), cmap='gray')
                plt.imsave(str(results_dir / "x0_diff" / f"{step}.png"), clear_color(norm_grad), cmap='gray')
                plt.imsave(str(results_dir / "x0_diff_mean" / f"{step}.png"), clear(norm_grad.mean(dim=1).unsqueeze(dim=1)))
            # xt = xt - step_size * norm_grad
            
            if mask is not None:
                xt_true = x1
                if not ot_ode:
                    _prev_step = torch.full((xt.shape[0],), prev_step, device=self.device, dtype=torch.long)
                    std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=x1.shape[1:])
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                xt = (1. - mask) * xt_true + mask * xt

            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)
    
    
    def dds_sampling(self, steps, pred_x0_fn, x1, x1_pinv, x1_forw,
                     mask=None, corrupt_type=None, corrupt_method=None, step_size=1.0,
                     ot_ode=False, log_steps=None, verbose=True, results_dir=None):
        xt = x1.detach().to(self.device)

        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]
        
        if results_dir is not None:
            for t in ["x0_before", "x0_after"]:
                (results_dir / t).mkdir(exist_ok=True, parents=True)

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDS sampling', total=len(steps)-1) if verbose else pair_steps
        cnt = 0
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            # turn on gradient
            xt.requires_grad_()
            pred_x0 = pred_x0_fn(xt, step)
            
            # DPS
            # for inpainting, corrupt_method returns a tuple
            if mask is not None:
                corrupt_x0_forw, _ = corrupt_method(pred_x0)
            elif "jpeg" in corrupt_type or "sr" in corrupt_type:
                _, corrupt_x0_forw = corrupt_method(pred_x0)
            else:
                corrupt_x0_forw, _ = corrupt_method(pred_x0)
            # residual = corrupt_x0 - x1_meas
            residual = corrupt_x0_forw - x1_forw
            residual_norm = torch.linalg.norm(residual) ** 2
            norm_grad = torch.autograd.grad(outputs=residual_norm, inputs=pred_x0)[0]
            
            xt, mu_x0 = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode, verbose=True)
            xt = xt - mu_x0 * step_size * norm_grad
            
            # take multiple gradient steps
            if results_dir is not None:
                if cnt == 5:
                    plt.imsave(str(results_dir / "x0_before" / f"{step}.png"), clear_color(pred_x0))
                    for k in range(5):
                        pred_x0 = pred_x0 - step_size * norm_grad
                        _, corrupt_x0_forw = corrupt_method(pred_x0)
                        residual = corrupt_x0_forw - x1_forw
                        residual_norm = torch.linalg.norm(residual) ** 2
                        norm_grad = torch.autograd.grad(outputs=residual_norm, inputs=pred_x0)[0]
                        plt.imsave(str(results_dir / "x0_after" / f"{step}_{k}.png"), clear_color(pred_x0))
            
            xt.detach_()
            pred_x0.detach_()
            
            if mask is not None:
                xt_true = x1
                if not ot_ode:
                    _prev_step = torch.full((xt.shape[0],), prev_step, device=self.device, dtype=torch.long)
                    std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=x1.shape[1:])
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                xt = (1. - mask) * xt_true + mask * xt

            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())
            cnt += 1

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)
    
    
    def pigdm_sampling(self, steps, pred_x0_fn, x1, x1_pinv, x1_forw,
                       mask=None, corrupt_type=None, corrupt_method=None, step_size=1.0,
                       ot_ode=False, log_steps=None, verbose=True):
        xt = x1.detach().to(self.device)

        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            # turn on gradient
            xt.requires_grad_()
            pred_x0 = pred_x0_fn(xt, step)
            
            # for inpainting, corrupt_method returns a tuple
            if mask is not None:
                corrupt_x0_pinv, _ = corrupt_method(pred_x0)
            elif "jpeg" in corrupt_type or "sr" in corrupt_type:
                corrupt_x0_pinv, _ = corrupt_method(pred_x0)
            else:
                _, corrupt_x0_pinv = corrupt_method(pred_x0)
            
            # pigdm
            mat = x1_pinv - corrupt_x0_pinv
            mat_rs = mat.detach().reshape(mat.shape[0], -1)
            mat_x = (mat_rs * pred_x0.reshape(mat.shape[0], -1)).sum()
            guidance = torch.autograd.grad(outputs=mat_x, inputs=pred_x0)[0]
            
            xt, mu_x0 = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode, verbose=True)
            xt = xt + mu_x0 * step_size * guidance
            
            # free memory
            xt.detach_()
            
            if mask is not None:
                xt_true = x1
                if not ot_ode:
                    _prev_step = torch.full((xt.shape[0],), prev_step, device=self.device, dtype=torch.long)
                    std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=x1.shape[1:])
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                xt = (1. - mask) * xt_true + mask * xt

            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)
    
    
    def ddim_sampling(self, steps, pred_x0_eps_fn, x1, eta=1.0, eps=1.0, mask=None, log_steps=None, verbose=True):
        """
        (pred_x0_fn) for ddim_sampling returns both pred_x0, model_output
        >> pred_x0, pred_eps = pred_x0_eps_fn(xt, step)
        """
        xt = x1.detach().to(self.device)

        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            pred_x0, pred_eps = pred_x0_eps_fn(xt, step)
            xt = self.p_posterior_ddim(prev_step, step, xt, pred_x0, pred_eps, eta=eta)

            if mask is not None:
                xt_true = x1
                _prev_step = torch.full((xt.shape[0],), prev_step, device=self.device, dtype=torch.long)
                std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=x1.shape[1:])
                xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                xt = (1. - mask) * xt_true + mask * xt

            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)
