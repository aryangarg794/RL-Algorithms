import gymnasium as gym
import numpy as np
import torch 

from gymnasium.spaces.discrete import Discrete
from torch import Tensor

from rl_algorithms.common.utils import compute_returns

class EpisodeBatch:
    
    def __init__(
        self,
        obs_shape, 
        action_shape, 
        states: list, 
        actions: list,
        rewards: list, 
        next_states: list,
        dones: list,  
        returns: list, 
        device: str = 'cuda'
    ):
        self.states = torch.from_numpy(np.array(states)).float().to(device=device).view(-1, obs_shape) \
            if not isinstance(states, Tensor) else states
        self.actions = torch.from_numpy(np.array(actions)).float().to(device=device).view(-1, action_shape) \
            if not isinstance(actions, Tensor) else actions
        self.rewards = torch.from_numpy(np.array(rewards)).float().to(device=device).view(-1, 1) \
            if not isinstance(rewards, Tensor) else rewards
        self.next_states = torch.from_numpy(np.array(next_states)).float().to(device=device).view(-1, obs_shape) \
            if not isinstance(next_states, Tensor) else next_states
        self.dones = torch.from_numpy(np.array(dones)).float().to(device=device).view(-1, 1) \
            if not isinstance(dones, Tensor) else dones
        self.returns = torch.from_numpy(np.array(returns)).float().to(device=device).view(-1, 1)
        
        self.action_shape = action_shape
    
    
class Runner:
    
    def __init__(
        self,
        env_id: str, 
        device: str = 'cuda',
        capacity: int = int(5e4)
    ):
        self.device = device
        self.env = gym.make(env_id)
        self.capacity = capacity
        self.pointer = 0
        self.size = 0
        
        self.states = torch.zeros((self.capacity, *self.env.observation_space.shape), dtype=torch.float, device=self.device)
        self.rewards = torch.zeros((self.capacity, 1) ,dtype=torch.float,device=self.device)
        self.next_states = torch.zeros((self.capacity, *self.env.observation_space.shape), dtype=torch.float, device=self.device)
        self.dones = torch.zeros((self.capacity, 1), dtype=torch.int64, device=self.device)
        
        if isinstance(self.env.action_space, Discrete):
            self.action_shape = 1
            self.actions = torch.zeros((self.capacity, 1), dtype=torch.int64, device=self.device)
        else:
            self.action_shape = np.prod(self.env.action_space.shape)
            self.actions = torch.zeros((self.capacity, self.action_shape), dtype=torch.float, device=self.device)
            
        self.done = False
        self.last_ep_rew = 0
        self.global_step = 0
        self.start_idx = 0
        
    def run_trajectory(self, policy, discount: float):
        states = []
        actions = []
        rewards = []
        next_states = []
        
        done = False
        obs, _ = self.env.reset()
        while not done:
            action = policy.sample_action(self.sanitize(obs)).cpu().numpy().reshape(-1)
            action = action[0] if policy.categorical else action
            next_obs, reward, trunc, term, _ = self.env.step(action)
            
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_obs)
            
            obs = next_obs
            done = trunc or term

        returns = compute_returns(rewards, discount)
        dones = [False for _ in range(len(states))]
        dones[-1] = True
        
        return EpisodeBatch(
            np.prod(self.env.observation_space.shape), 
            self.action_shape,
            states=states, 
            actions=actions, 
            rewards=rewards,
            next_states=next_states, 
            dones=dones, 
            returns=returns, 
            device=self.device
        )
    
    
    def run_step(self, policy):
        
        if self.done or self.global_step == 0: 
            obs, _ = self.env.reset()
            self.done = False
            self.last_ep_rew = 0
        else:
            obs = self.next_states[-1]
            
        action = policy.sample_action(self.sanitize(obs)).cpu().numpy().reshape(-1)
        action = action[0] if policy.categorical else action
        next_obs, reward, trunc, term, _ = self.env.step(action)
        self.done = trunc or term
        self.last_ep_rew += reward
        
        self.states[self.pointer] = torch.as_tensor(obs).to(self.device)
        self.actions[self.pointer] = torch.as_tensor(action).to(self.device) \
            if not policy.categorical else action
        self.rewards[self.pointer] = reward
        self.next_states[self.pointer] = torch.as_tensor(next_obs).to(self.device)
        self.dones[self.pointer] = int(self.done)
        
        self.pointer = (self.pointer + 1) % self.capacity 
        self.size = min(self.size + 1, self.capacity)
        
        self.global_step += 1
    
    def return_rollout(self, discount: float):
        returns = compute_returns(self.rewards, discount)
        
        batch = EpisodeBatch(
            np.prod(self.env.observation_space.shape), 
            self.action_shape,
            states=self.states, 
            actions=self.actions, 
            rewards=self.rewards,
            next_states=self.next_states,
            dones=self.dones, 
            returns=returns, 
            device=self.device
        )
        
        return batch
    
    
    def sample(self, batch_size):
        ind = torch.randint(0, self.size, device=self.device, size=(batch_size,))
        return (
            self.states[ind], 
            self.actions[ind], 
            self.rewards[ind], 
            self.next_states[ind], 
            self.dones[ind]
        )
    
    def sanitize(self, state: np.ndarray):
        return torch.from_numpy(state).float().to(device=self.device).view(1, -1)