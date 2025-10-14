# summarizer.py
from typing import Dict, Any
import numpy as np
import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.acquisition.analytic import ExpectedImprovement

def summarize_state(gp: SingleTaskGP, X: Tensor, Y: Tensor, grid_n: int = 20, topk: int = 5) -> Dict[str, Any]:
    with torch.no_grad():
        xs = torch.linspace(0, 1, grid_n, device=X.device, dtype=X.dtype)
        xx, yy = torch.meshgrid(xs, xs, indexing="xy")
        grid = torch.stack([xx, yy], dim=-1).reshape(-1, 2)

        post = gp.posterior(grid.unsqueeze(1))  # q=1
        var = post.variance.reshape(-1)

        best_f = float(Y.max().item())
        EI = ExpectedImprovement(model=gp, best_f=best_f, maximize=True)
        ei_vals = EI(grid.unsqueeze(1)).reshape(-1)

        k1 = min(topk, ei_vals.numel()); k2 = min(topk, var.numel())
        top_ei  = grid[torch.topk(ei_vals, k=k1).indices].detach().cpu().tolist()
        top_var = grid[torch.topk(var,     k=k2).indices].detach().cpu().tolist()
        inc     = X[int(torch.argmax(Y))].detach().cpu().tolist()

        lsc = None
        if hasattr(gp.covar_module, 'base_kernel') and hasattr(gp.covar_module.base_kernel, 'lengthscale'):
            l = gp.covar_module.base_kernel.lengthscale.detach().view(-1).cpu().tolist()
            lsc = l[:2] if len(l) >= 2 else l

        # simple coverage/density
        H, _, _ = np.histogram2d(X[:,0].cpu().numpy(), X[:,1].cpu().numpy(),
                                 bins=grid_n, range=[[0,1],[0,1]])

        return {
            "grid_n": grid_n,
            "top_ei": top_ei,
            "top_var": top_var,
            "incumbent": inc,
            "lengthscales": lsc,
            "density": H.astype(int).tolist(),
            "best_y": best_f,
            "n_obs": int(X.shape[0]),
        }
