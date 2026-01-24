import torch

from torch import Tensor
from typing import Callable

# mainly from https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf

def conjugate_gradient(
    FVP: Callable, 
    b: Tensor, 
    max_iter: int = 15, 
    eps: float = 1e-6, 
    device: str = 'cuda'
):
    x = 1e-4 * torch.randn_like(b, device=device) # based on sb3 implementation 
    res = b - FVP(x)
    d = res.clone()
    error = torch.dot(res, res)
    i = 0
    while i < max_iter and error > eps:
        q = FVP(d)
        alpha = error / (torch.dot(d, q))
        x = x + alpha * d
        if i % 50 == 0: # dont need it technially since iter < 50 but it was in the algo
            res = b - FVP(x)
        else:
            res = res - alpha * q
        
        old_error = error.clone()
        error = torch.dot(res, res)
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
    c: float = 0.8, 
    max_iter: int = 10
):
    update = False
    expn = start
    old_loss = initial_loss
    old_weights = model.actor_params.clone()
    for i in range(max_iter):
        new_params = old_weights + expn * step_dir
        model.update_actor_weights(new_params)
        act_loss, _ = model.actor_loss(batch.states, batch.actions, advantages, old_dist)
        if model.categorical:
            kl = model.kl_categorical(batch.states, old_dist).mean()
        else:
            kl = model.kl_gaussian(batch.states, old_dist).mean()
        if act_loss > old_loss and kl <= model.trust_region:
            update = True
            break
    
        expn *= c
        
    if not update:
        model.update_actor_weights(old_weights)
    
    return update, kl.item() if update else 0.0, act_loss