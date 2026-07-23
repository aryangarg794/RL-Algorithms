import gymnasium as gym
import dill
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import torch
import torch.nn as nn
import typer

from copy import deepcopy
from collections import defaultdict
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces.multi_discrete import MultiDiscrete
from torch import Tensor
from torch import distributions as dist
from tqdm import tqdm
from typing import List, Annotated

from rl_algorithms.common.optimization_methods import conjugate_gradient, backtracking_linesearch_with_kl
from rl_algorithms.common.buffers import RolloutBufferBatch, Runner
from rl_algorithms.common.utils import explained_variance, convert_results
from rl_algorithms.common.plotting import plot_results

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

class TRPO:
    
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
        results = defaultdict(list)
        n_updates = 0
        
        num_eps = int(total_timesteps / (self.n_envs * self.n_steps))
        for ep in (pbar := tqdm(range(1, num_eps+1))):
            rollout_data = self.runner.run_rollout(self)
            act_loss, cr_loss, kl, updated = self.update_step(rollout_data)
            n_updates += int(updated)
            results['act_losses'].append(act_loss)
            results['cr_losses'].append(cr_loss)
            
            postfix = {}
            rew_mean = np.mean(list(self.runner.envs.return_queue)[-window_size:])
            postfix["avg_rew"] = rew_mean
            postfix["act_loss"] = act_loss
            postfix["critic_loss"] = cr_loss
            postfix["kl_div"] = kl
            postfix["num_updates"] = n_updates

            results['ep_rew_mean'].append(rew_mean)
            results['kls'].append(kl)
        
            pbar.set_description(f"Num Update Steps: {ep}")
            pbar.set_postfix(postfix)
        
        return results
    
    
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
        }, f'{file_path}.pt')
        
    def load(self, file_path: str): 
        saved_model = torch.load(file_path, weights_only=True)
        self.actor.load_state_dict(saved_model['actor_dict'])
        self.critic.load_state_dict(saved_model['critic_dict'])
    
    @property
    def actor_params(self): 
        # returns a copy 
        return torch.cat([grad.flatten() for grad in self.actor.parameters()])
    

def train_trpo(
    env_id: Annotated[str, typer.Option(help='environment to test on (has to be gym env)')], 
    lr_critic: Annotated[float, typer.Option(help='lr for critic')] = 1e-3, 
    discount: Annotated[float, typer.Option(help='gamma')] = 0.99,
    trust_region: Annotated[float, typer.Option(help='trust region coeff')] = 0.01, 
    act_hidden: Annotated[List[int], typer.Option(help='actor hidden dims')] = list([400, 300]),
    critic_hidden: Annotated[List[int], typer.Option(help='critic hidden dims')] = list([400, 300]),
    damping: Annotated[float, typer.Option(help='damping coeff')] = 0.0, 
    device: Annotated[str, typer.Option(help='device')] = 'cuda', 
    num_critic_updates: Annotated[int, typer.Option(help='num critic updates per actor updates')] = 10, 
    norm_adv: Annotated[bool, typer.Option(help='normalize adv')] = True, 
    n_envs: Annotated[int, typer.Option(help='num of envs')] = 5, 
    batch_size: Annotated[int, typer.Option(help='batch size')] = 128, 
    n_steps: Annotated[int, typer.Option(help='num of steps per rollout')] = 512,
    total_timesteps: Annotated[int, typer.Option(help='num total training steps')] = 100_000,
    window_size: Annotated[int, typer.Option(help='window size to average the ep rews')] = 100,
    seeds: Annotated[List[int], typer.Option(help='seeds to run')] = list([0, 1, 2, 3]),
    save: Annotated[bool, typer.Option(help='save model/results')] = False,
    run_name: Annotated[str, typer.Option(help='name of the run')] = "trpo"
):
    results = {}
    run_name = run_name + f"_{env_id}"
    os.makedirs('./results/trpo/media', exist_ok=True)
    os.makedirs('./results/trpo/objs', exist_ok=True)
    os.makedirs('./results/trpo/models', exist_ok=True)
    plt.style.use('ggplot')

    for seed in seeds:

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        model = TRPO(
            env_id=env_id,
            lr_critic=lr_critic,
            discount=discount,
            trust_region=trust_region,
            act_hidden=act_hidden,
            critic_hidden=critic_hidden,
            damping=damping, 
            device=device, 
            num_critic_updates=num_critic_updates, 
            norm_adv=norm_adv, 
            n_envs=n_envs, 
            batch_size=batch_size, 
            n_steps=n_steps, 
        )

        results_seed = model.train(total_timesteps=total_timesteps, window_size=window_size)
        results[seed] = results_seed

        if save:
            model.save(f'results/trpo/models/{run_name}_seed_{seed}')

    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    results = convert_results(results)
    plot_results(results['ep_rew_mean'], axes[0, 0], total_timesteps, 'TRPO', 'Timesteps', 'Average Ep Reward', 
                 'Average Episodic Reward over Time', 'red')
    plot_results(results['act_losses'], axes[0, 1], total_timesteps, 'TRPO', 'Timesteps', 'Actor Loss', 
                 'Actor Loss over Time', 'blue')
    plot_results(results['cr_losses'], axes[1, 0], total_timesteps, 'TRPO', 'Timesteps', 'Critic Loss', 
                 'Critic Loss over Time', 'green')
    plot_results(results['kls'], axes[1, 1], total_timesteps, 'TRPO', 'Timesteps', 'KL Div (new/old)', 
                 'KL Div between new/old Policy over Time', 'purple')
    
    line_style = {
        "linestyle": "--",
        "color": "purple",
        "linewidth": 1.5,
        "alpha": 0.8,
        "zorder": 2 
    }
    axes[1, 1].axhline(y=trust_region, label='Trust Region', **line_style)
    axes[1, 1].legend()

    plt.tight_layout()
    fig.savefig(f'results/trpo/media/{run_name}.png')


    if save:
        with open(f'/results/trpo/objs/{run_name}.pl', 'wb') as file:
            dill.dump(results, file)
            file.close()
        
        

    return 
