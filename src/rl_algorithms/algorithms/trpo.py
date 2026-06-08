import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces.multi_discrete import MultiDiscrete
from torch import Tensor
from torch import distributions as dist
from tqdm import tqdm

from rl_algorithms.common.optimization_methods import conjugate_gradient, backtracking_linesearch_with_kl
from rl_algorithms.common.buffers import RolloutBufferBatch, Runner
from rl_algorithms.common.utils import explained_variance

# some inspiration take from https://github.com/ikostrikov/pytorch-trpo (particularly grad directions etc)

class ActorCont(nn.Module):
    
    def __init__(
        self, 
        obs_shape: int, 
        action_shape: int, 
        act_hidden: list = list([]),
        act: nn.Module = nn.ReLU,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.ffn = nn.Sequential(
            nn.Linear(obs_shape, act_hidden[0]),
            act()
        )
        
        for layer1, layer2 in zip(act_hidden[:-1], act_hidden[1:]):
            self.ffn.extend([
                nn.Linear(layer1, layer2),
                act()
            ])
        self.ffn.append(nn.Linear(act_hidden[-1], action_shape))
        self.actor_std_log = nn.Parameter(torch.zeros(1, action_shape))
    
    def forward(self, states: Tensor):
        means = self.ffn(states)
        return means, torch.exp(self.actor_std_log.expand_as(means))
    
class ActorDisc(nn.Module):
    
    def __init__(
        self, 
        obs_shape: int, 
        action_shape: int, 
        act_hidden: list = list([]),
        act: nn.Module = nn.ReLU,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.ffn = nn.Sequential(
            nn.Linear(obs_shape, act_hidden[0]),
            act()
        )
        
        for layer1, layer2 in zip(act_hidden[:-1], act_hidden[1:]):
            self.ffn.extend([
                nn.Linear(layer1, layer2),
                act()
            ])
        self.ffn.append(nn.Linear(act_hidden[-1], action_shape))
    
    def forward(self, states: Tensor):
        logits = self.ffn(states)
        return logits

class TRPOAgent:
    
    def __init__(
        self, 
        env_id: str,
        lr_critic: float = 1e-3,
        discount: float = 0.99, 
        trust_region: float = 0.01,
        act_hidden: list = list([400, 300]),
        critic_hidden: list = list([400, 300]),
        act: nn.Module = nn.ReLU,
        damping: float = 0.1, 
        device: str = 'cuda', 
        num_critic_updates: int = 10, 
        norm_adv: bool = True, 
        n_envs: int = 5, 
        batch_size: int = 128, 
        n_steps: int = 512,
        *args, 
        **kwargs
    ):
        self.runner = Runner(env_id, device=device, n_envs=n_envs, n_steps=n_steps)
        self.categorical = True if isinstance(self.runner.envs.action_space, (Discrete, MultiDiscrete)) else False
        self.action_shape = self.runner.envs.single_action_space.n if self.categorical else self.runner.action_shape
        self.obs_shape = self.runner.obs_shape
        self.discount = discount
        self.trust_region = trust_region
        self.damping = damping
        self.device = device
        self.critic_steps = num_critic_updates
        self.norm_adv = norm_adv
        self.n_envs = n_envs
        self.batch_size = batch_size
        self.n_steps = n_steps

        if self.categorical:
            self.actor = ActorDisc(self.obs_shape, self.action_shape, act_hidden=act_hidden, act=act).to(device)
        else:
            self.actor = ActorCont(self.obs_shape, self.action_shape, act_hidden=act_hidden, act=act).to(device)
        
        self.critic = nn.Sequential(
            nn.Linear(self.obs_shape, critic_hidden[0]),
            act()
        )
        for layer1, layer2 in zip(critic_hidden[:-1], critic_hidden[1:]):
            self.critic.extend([
                nn.Linear(layer1, layer2),
                act()
            ])
        self.critic.append(nn.Linear(critic_hidden[-1], 1))
        self.critic.to(device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.critic_loss = nn.MSELoss()
         
        
    def __call__(self, states: Tensor):
        if self.categorical:
            logits = self.actor(states)
            return logits
        else:
            means, stds = self.actor(states)
            return means, stds
    
    @torch.no_grad()
    def sample_action(self, states: Tensor | np.ndarray, return_np: bool = False):
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).reshape(-1, self.obs_shape).float()
        if self.categorical:
            logits = self(states)
            act_dist = dist.Categorical(logits=logits)
        else:
            means, stds = self(states)
            act_dist = dist.Normal(loc=means, scale=stds)

        sample = act_dist.sample()
        log_probs = act_dist.log_prob(sample)
        
        return (
            sample.cpu().numpy() if return_np else sample, 
            log_probs.view(-1, 1) if self.categorical else log_probs.sum(dim=-1, keepdim=True),
        )
    
    def update_actor_weights(self, new_params: Tensor):
        offset = 0
        for param in list(self.actor.parameters()):
            n = param.numel()
            new_param = new_params[offset:offset+n]
            param.data.copy_(new_param.reshape(param.shape))
            offset += n 
            
    
    def actor_loss_cont(self, states: Tensor, actions: Tensor, advantages: Tensor, old_log_probs: tuple):
        new_means, new_std = self(states)
        dist_new = dist.Normal(loc=new_means, scale=new_std)
        log_std_new = dist_new.log_prob(actions)

        ratios = torch.exp(log_std_new - old_log_probs).sum(1, keepdim=True)
    
        return (advantages * ratios).mean(), dist_new
    
    def actor_loss_disc(self, states: Tensor, actions: Tensor, advantages: Tensor, old_log_probs: tuple):
        new_logits = self(states)
        dist_new = dist.Categorical(logits=new_logits)

        log_std_new = dist_new.log_prob(actions.squeeze().long()).view(-1, 1)
        ratios = torch.exp(log_std_new - old_log_probs)
        
        return (advantages * ratios).mean(), dist_new
    
    def kl_div(self, dist_new: dist.Distribution, dist_old: dist.Distribution):
        return torch.distributions.kl_divergence(dist_new, dist_old).mean()
    
    def train(self, total_timesteps: int, window_size: int = 100):
        act_losses = []
        cr_losses = []
        n_updates = 0
        
        num_eps = int(total_timesteps / (self.n_envs * self.n_steps))
        for ep in (pbar := tqdm(range(1, num_eps+1))):
            rollout_data = self.runner.run_rollout(self)
            act_loss, cr_loss, kl, updated = self.update_step(rollout_data)
            n_updates += int(updated)
            act_losses.append(act_loss)
            cr_losses.append(cr_loss)

            postfix = {}
            postfix["avg_rew"] = np.mean(list(self.runner.envs.return_queue)[-window_size:])
            postfix["act_loss"] = act_losses[-1]
            postfix["critic_loss"] = cr_losses[-1]
            postfix["kl_div"] = kl
            postfix["num_updates"] = n_updates
        
            pbar.set_description(f"Episode: {ep}")
            pbar.set_postfix(postfix)
        
        return act_losses, cr_losses
    
    
    def get_distribution(self, states: Tensor):
        if self.categorical:
            logits = self(states)
            dist_new = dist.Categorical(logits=logits)
        else:
            means, stds = self(states)
            dist_new = dist.Normal(loc=means, scale=stds)

        return dist_new


    def update_step(self, rollout_data: RolloutBufferBatch):
        for batch in rollout_data.get(batch_size=None):
            with torch.no_grad():
                if self.categorical:
                    old_logits = self(batch.states)
                    dist_old = dist.Categorical(logits=old_logits.detach())
                else:
                    old_means, old_stds = self(batch.states)
                    dist_old = dist.Normal(loc=old_means.detach(), scale=old_stds.detach())

            # compute advs
            advantages = batch.advantages
            if self.norm_adv:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = advantages.detach()
            actor_loss, _ = self.actor_loss(batch.states, batch.actions, advantages, batch.log_probs)            

            # grads point in dir to maximize (ie lower reward)
            grads_pg = torch.autograd.grad(actor_loss, self.actor.parameters(), retain_graph=True) 
            grads_pg = torch.cat([grad.view(-1) for grad in grads_pg])
            
            # fisher-vector product
            def FVP(y: Tensor):
                # kl grads
                dist_new = self.get_distribution(batch.states)
                kl = self.kl_div(dist_new, dist_old)
                grads_kl = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True, retain_graph=True, 
                                            allow_unused=True)
                flat_grads_kl = self.flatten_grads(grads_kl)

                prod = torch.dot(flat_grads_kl, y)
                # second derivative
                grads = torch.autograd.grad(prod, self.actor.parameters())
                flat_grads = torch.cat([grad.contiguous().view(-1) for grad in grads]).data
                
                return flat_grads + y * self.damping
            
            x_k = conjugate_gradient(FVP, grads_pg, device=self.device) # negate pg so that we continue to point
            hessian_search = FVP(x_k)
            beta = torch.sqrt(2 * self.trust_region / (torch.dot(x_k, hessian_search)))
            step_dir = beta * x_k
            
            updated, kl_div, actor_loss = backtracking_linesearch_with_kl(self, batch, advantages, dist_old, step_dir, 1.0, actor_loss)
        
        # update critic
        for _ in range(self.critic_steps):
            for batch in rollout_data.get(batch_size=self.batch_size):
                values = self.critic(batch.states)
                critic_loss = self.critic_loss(batch.returns, values)
                
                self.critic_opt.zero_grad()
                critic_loss.backward()
                self.critic_opt.step()
    
        return actor_loss.item(), critic_loss.item(), kl_div, updated
    
    def actor_loss(self, states: Tensor, actions: Tensor, advantages: Tensor, old_log_probs: Tensor):
        if self.categorical:
            return self.actor_loss_disc(states, actions, advantages, old_log_probs)
        else:
            return self.actor_loss_cont(states, actions, advantages, old_log_probs)
    
    def flatten_grads(self, grads: Tensor):
        return torch.cat([grad.view(-1) for grad in grads])
    
    def save(self, file_path: str):
        torch.save({
            'actor_dict': self.actor.state_dict(),
            'critic_dict': self.critic.state_dict()
        }, f'models/TRPO_{file_path}.pt')
        
    def load(self, file_path: str): 
        saved_model = torch.load(file_path, weights_only=True)
        self.actor.load_state_dict(saved_model['actor_dict'])
        self.critic.load_state_dict(saved_model['critic_dict'])
    
    @property
    def actor_params(self): 
        # returns a copy 
        return torch.cat([grad.flatten() for grad in self.actor.parameters()])
    
    
    
if __name__ == "__main__":
    # test run
    agent = TRPOAgent(
        'CartPole-v1', 
        device='cpu',
        n_envs=1, 
        n_steps=2048, 
        n_update_steps=2000,
    )
    agent.train(total_timesteps=700000, window_size=100)
