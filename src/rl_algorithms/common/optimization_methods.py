import torch

from torch import Tensor
from typing import Callable

def conjugate_gradient(
    FVP: Callable, 
    b: Tensor, 
    max_iter: int = 10, 
    eps: float = 1e-6, 
    device: str = 'cuda'
):
    x = torch.zeros_like(b, device=device)
    res = b - FVP(x)
    d = res.clone()
    error = res.T @ res
    i = 0
    while i < max_iter and error > eps:
        q = FVP(d)
        alpha = error / (res.T @ q)
        x = x + alpha * d
        if i % 50: # dont need it technially since iter < 50 but it was in the algo
            res = b - FVP(x)
        else:
            res = res - alpha * q
        
        old_error = error.clone()
        error = res.T @ res
        beta = error / old_error
        d = res + beta * d
        i += 1
        
    return x

@torch.no_grad()
def backtracking_linesearch_with_kl(
    model, 
    batch: Tensor,
    advantages: Tensor, 
    old_dist: tuple,
    step_dir: Tensor, 
    start: float, 
    initial_loss: float | Tensor, 
    c: float = 0.5, 
    max_iter: int = 10
):
    update = False
    expn = start
    old_loss = initial_loss
    old_weights = model.actor_params.clone()
    for i in range(max_iter):
        new_params = old_weights + expn * step_dir
        model.update_actor_weights(new_params)
        act_loss, new_dist = model.actor_loss(batch.states, batch.actions, advantages, old_dist)
        kl = model.kl_gaussian(batch.states, old_dist).mean()
        if act_loss < old_loss:
            if kl <= model.trust_region:
                update = True
                break
    
        expn *= c
        old_loss = act_loss
        
    if not update:
        model.update_actor_weights(old_weights)
    
    return update