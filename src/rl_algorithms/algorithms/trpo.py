import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy
from gymnasium.spaces.discrete import Discrete
from torch import Tensor
from torch import distributions as dist
from tqdm import tqdm

from rl_algorithms.common.optimization_methods import conjugate_gradient, backtracking_linesearch_with_kl
from rl_algorithms.common.buffers import EpisodeBatch, Runner
from rl_algorithms.common.utils import RollingAverage

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
        lr_critic: float = 1e-4,
        discount: float = 0.99, 
        trust_region: float = 0.01,
        act_hidden: list = list([400, 300]),
        critic_hidden: list = list([400, 300]),
        act: nn.Module = nn.ReLU,
        damping: float = 0.1, 
        device: str = 'cuda', 
        num_critic_updates: int = 10, 
        *args, 
        **kwargs
    ):
        self.runner = Runner(env_id, device=device)
        self.categorical = True if isinstance(self.runner.env.action_space, Discrete) else False
        self.action_shape = self.runner.env.action_space.n if self.categorical else np.prod(self.runner.env.action_space.shape)
        self.obs_shape = np.prod(self.runner.env.observation_space.shape)
        self.discount = discount
        self.trust_region = trust_region
        self.damping = damping
        self.device = device
        self.critic_steps = num_critic_updates

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
    def sample_action(self, states: Tensor):
        if self.categorical:
            logits = self(states)
            return dist.Categorical(logits=logits).sample()
        else:
            means, stds = self(states)
            return dist.Normal(loc=means, scale=stds).sample()
        
    def update_actor_weights(self, new_params: Tensor):
        offset = 0
        for param in list(self.actor.parameters()):
            n = param.numel()
            new_param = new_params[offset:offset+n]
            param.data.copy_(new_param.reshape(param.shape))
            offset += n 
            
    
    def actor_loss_cont(self, states: Tensor, actions: Tensor, advantages: Tensor, old_prob: tuple):
        old_means, old_std = old_prob
        new_means, new_std = self(states)
        dist_new = dist.Normal(loc=new_means, scale=new_std)
        dist_old = dist.Normal(loc=old_means, scale=old_std)
        log_std_new = dist_new.log_prob(actions).sum(1, keepdim=True)
        log_std_old = dist_old.log_prob(actions).sum(1, keepdim=True)
        ratios = torch.exp(log_std_new - log_std_old)
    
        return (advantages * ratios).mean(), (new_means, new_std)
    
    def actor_loss_disc(self, states: Tensor, actions: Tensor, advantages: Tensor, old_prob: tuple):
        old_logits = old_prob
        new_logits = self(states)
        dist_new = dist.Categorical(logits=new_logits)
        dist_old = dist.Categorical(logits=old_logits)

        log_std_new = dist_new.log_prob(actions.squeeze()).view(-1, 1)
        log_std_old = dist_old.log_prob(actions.squeeze()).view(-1, 1)
        ratios = torch.exp(log_std_new - log_std_old)
        
        return (advantages * ratios).mean(), (new_logits)
    
    def kl_gaussian(self, states: Tensor, old_dist: tuple):
        a_mean, a_std = old_dist
        b_mean, b_std = self(states)
        kl = 0.5 * (torch.log(b_std.pow(2)/(a_std.pow(2) + 1e-6)) - 1 + (b_std/(a_std + 1e-6)) 
                      + (b_mean - a_mean).pow(2)/(b_std.pow(2) + 1e-6))
        return kl.sum(1, keepdim=True)
    
    def kl_categorical(self, states, old_dist: tuple):
        old_logits = old_dist
        new_logits = self(states)
        dist_new = dist.Categorical(logits=new_logits)
        dist_old = dist.Categorical(logits=old_logits)
        return torch.distributions.kl_divergence(dist_old, dist_new)
    
    def train(self, num_episodes: int = 500, window_size: int = 10):
        act_losses = []
        cr_losses = []
        avg_ep_rewards = RollingAverage(window_size)
        n_updates = 0
        
        for ep in (pbar := tqdm(range(1, num_episodes+1))):
            batch = self.runner.run_trajectory(self, self.discount)
            act_loss, cr_loss, kl, updated = self.update_step(batch)
            n_updates += 1 if updated else 0
            avg_ep_rewards.update(batch.rewards.sum().item())
            act_losses.append(act_loss)
            cr_losses.append(cr_loss)

            postfix = {}
            postfix["avg_rew"] = avg_ep_rewards.get_average
            postfix["act_loss"] = act_losses[-1]
            postfix["critic_loss"] = cr_losses[-1]
            postfix["kl_div"] = kl
            postfix["num_updates"] = n_updates
        
            pbar.set_description(f"Episode: {ep}")
            pbar.set_postfix(postfix)
        
        return act_losses, cr_losses
    
    def update_step(self, batch: EpisodeBatch):
        with torch.no_grad():
            if self.categorical:
                old_probs = self(batch.states)
            else:
                old_means, old_stds = self(batch.states)
        # compute advs
        with torch.no_grad():
            values = self.critic(batch.states)
            next_values = self.critic(batch.next_states)
        advantages = batch.rewards + self.discount * next_values * (1 - batch.dones) - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        if self.categorical:
            actor_loss, _ = self.actor_loss_disc(batch.states, batch.actions, advantages, (old_probs))
        else:
            actor_loss, _ = self.actor_loss_cont(batch.states, batch.actions, advantages, (old_means, old_stds))
        
        # grads point in dir to maximize (ie lower reward)
        grads_pg = torch.autograd.grad(actor_loss, self.actor.parameters()) 
        grads_pg = torch.cat([grad.view(-1) for grad in grads_pg])
        
        # fisher-vector product
        def FVP(y: Tensor):
            if self.categorical:
                kl = self.kl_categorical(batch.states, (old_probs.detach())).mean()
            else:
                kl = self.kl_gaussian(batch.states, (old_means.detach(), old_stds.detach())).mean()
            grads = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True, retain_graph=True, 
                                        allow_unused=True)
            flat_grads = self.flatten_grads(grads)
            
            prod = torch.dot(flat_grads, y)
            # second derivative
            grads = torch.autograd.grad(prod, self.actor.parameters())
            flat_grads = torch.cat([grad.contiguous().view(-1) for grad in grads]).data
            
            return flat_grads + y * self.damping
        
        x_k = conjugate_gradient(FVP, grads_pg, device=self.device) # negate pg so that we continue to point
        hessian_search = FVP(x_k)
        beta = torch.sqrt(2 * self.trust_region / (torch.dot(x_k, hessian_search)))
        step_dir = beta * x_k
        
        if self.categorical: 
            updated, kl_div, actor_loss = backtracking_linesearch_with_kl(self, batch, advantages, (old_probs), step_dir, 1, actor_loss)
        else:
            updated, kl_div, actor_loss = backtracking_linesearch_with_kl(self, batch, advantages, (old_means, old_stds), step_dir, 1, actor_loss)
        
        # update critic
        for _ in range(self.critic_steps):
            values = self.critic(batch.states)
            critic_loss = self.critic_loss(values, batch.returns)
            
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()
        
        
        return actor_loss.item(), critic_loss.item(), kl_div, updated
    
    def actor_loss(self, states: Tensor, actions: Tensor, advantages: Tensor, old_prob: tuple):
        if self.categorical:
            return self.actor_loss_disc(states, actions, advantages, old_prob)
        else:
            return self.actor_loss_cont(states, actions, advantages, old_prob)
    
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
    agent = TRPOAgent('CartPole-v1')
    agent.train(1000, window_size=35)
    