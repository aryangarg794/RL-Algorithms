import gymnasium as gym
import numpy as np
import torch 

from functools import partial
from gymnasium.spaces.discrete import Discrete
from torch import Tensor

from rl_algorithms.common.utils import compute_returns

class RolloutBufferBatch:

    def __init__(
        self,
        states: Tensor, 
        actions: Tensor,
        rewards: Tensor, 
        next_states: Tensor,
        dones: Tensor,  
        log_probs: Tensor, 
        returns: Tensor, 
        advantages: Tensor, 
    ):
        
        self.states = states
        self.actions = actions 
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones
        self.returns = returns
        self.log_probs = log_probs
        self.advantages = advantages

class RolloutBuffer:
    
    def __init__(
        self,
        obs_shape, 
        action_shape, 
        states: list, 
        actions: list,
        rewards: list, 
        next_states: list,
        dones: list,  
        log_probs: list, 
        returns: list, 
        advantages: list, 
        device: str = 'cuda'
    ):
        self.states = torch.from_numpy(np.array(states)).float().to(device=device).view(-1, obs_shape) \
            if not isinstance(states, Tensor) else states.view(-1, obs_shape)
        self.actions = torch.from_numpy(np.array(actions)).float().to(device=device).view(-1, action_shape) \
            if not isinstance(actions, Tensor) else actions.view(-1, action_shape)
        self.rewards = torch.from_numpy(np.array(rewards)).float().to(device=device).view(-1, 1) \
            if not isinstance(rewards, Tensor) else rewards.view(-1, 1)
        self.next_states = torch.from_numpy(np.array(next_states)).float().to(device=device).view(-1, obs_shape) \
            if not isinstance(next_states, Tensor) else next_states.view(-1, obs_shape)
        self.dones = torch.from_numpy(np.array(dones)).float().to(device=device).view(-1, 1) \
            if not isinstance(dones, Tensor) else dones.view(-1, 1)
        self.log_probs = torch.from_numpy(np.array(log_probs)).float().to(device=device).view(-1, 1) \
            if not isinstance(log_probs, Tensor) else log_probs.view(-1, 1)
        self.returns = torch.from_numpy(np.array(returns)).float().to(device=device).view(-1, 1) \
            if not isinstance(returns, Tensor) else returns.view(-1, 1)
        self.advantages = torch.from_numpy(np.array(advantages)).float().to(device=device).view(-1, 1) \
            if not isinstance(advantages, Tensor) else advantages.view(-1, 1)
        
        self.action_shape = action_shape

    def get(self, batch_size: int = None):
        if batch_size is None:
            batch_size = self.states.size(0)

        indices = torch.randperm(self.states.size(0))
        state_batches = list(torch.split(self.states[indices], batch_size, dim=0))
        actions_batches = list(torch.split(self.actions[indices], batch_size, dim=0))
        rewards_batches = list(torch.split(self.rewards[indices], batch_size, dim=0))
        next_states_batches = list(torch.split(self.next_states[indices], batch_size, dim=0))
        dones_batches = list(torch.split(self.dones[indices], batch_size, dim=0))
        log_probs_batches = list(torch.split(self.log_probs[indices], batch_size, dim=0))
        returns_batches = list(torch.split(self.returns[indices], batch_size, dim=0))
        advs_batches = list(torch.split(self.advantages[indices], batch_size, dim=0))

        rollout_data = [RolloutBufferBatch(
            states=state_batches[i],
            actions=actions_batches[i],
            rewards=rewards_batches[i],
            next_states=next_states_batches[i],
            dones=dones_batches[i],
            log_probs=log_probs_batches[i],
            advantages=advs_batches[i],
            returns=returns_batches[i]
        ) for i in range(len(state_batches))]

        return rollout_data

class BasicBuffer:
    
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
        
        batch = RolloutBuffer(
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
class Runner:
    
    def __init__(
        self,
        env_id: str, 
        device: str = 'cuda',
        n_steps: int = 512, 
        n_envs: int = 8, 
        discount: float = 0.99
    ):
        self.device = device
        self.num_envs = n_envs
        self.n_steps = n_steps
        self.discount = discount
        envs = gym.make_vec(env_id, num_envs=n_envs)
            
        self.envs = gym.wrappers.vector.RecordEpisodeStatistics(
            envs, buffer_length=1000
        )

        categorical = isinstance(self.envs.single_action_space, Discrete)
        self.action_shape = 1 if categorical else np.prod(self.envs.single_action_space.shape)
        self.obs_shape = np.prod(self.envs.single_observation_space.shape)
        
        self.batch_class = partial(RolloutBuffer, self.obs_shape, self.action_shape)
        self.global_step = 0
        self._cur_states = None
        
    def run_rollout(self, policy):
        states_buffer = torch.zeros((self.n_steps, self.num_envs, self.obs_shape), device=self.device)
        actions_buffer = torch.zeros((self.n_steps, self.num_envs, self.action_shape), device=self.device)
        rewards_buffer = torch.zeros((self.n_steps, self.num_envs, 1), device=self.device)
        next_states_buffer = torch.zeros((self.n_steps, self.num_envs, self.obs_shape), device=self.device)
        dones_buffer = torch.zeros((self.n_steps, self.num_envs, 1), device=self.device)
        log_probs_buffer = torch.zeros((self.n_steps, self.num_envs, 1), device=self.device)

        for step in range(self.n_steps):

            if self.global_step == 0:
                self._cur_states, _ = self.envs.reset()

            states_buffer[step] = torch.from_numpy(self._cur_states).to(self.device)
            actions, log_probs = policy.sample_action(self._cur_states, return_np=True)
            actions_buffer[step] = torch.from_numpy(actions).to(self.device).view(-1, self.action_shape)
            log_probs_buffer[step] = log_probs

            self._cur_states, rewards, terminated, truncated, _ = self.envs.step(actions)
            next_states_buffer[step] = torch.from_numpy(self._cur_states).to(self.device)
            rewards_buffer[step] = torch.from_numpy(rewards).to(self.device).view(-1, 1)
            dones_buffer[step] = torch.from_numpy(terminated).to(self.device).view(-1, 1)

            self.global_step += 1

        returns_buffer, advs_buffer = self.compute_boot_returns(
            policy=policy, states=states_buffer, next_states=next_states_buffer, 
            rewards=rewards_buffer, dones=dones_buffer
        )
        
        return self.batch_class(
            states=states_buffer, 
            actions=actions_buffer, 
            rewards=rewards_buffer,
            next_states=next_states_buffer, 
            dones=dones_buffer, 
            log_probs=log_probs_buffer, 
            returns=returns_buffer,
            advantages=advs_buffer, 
            device=self.device
        )
    
    def compute_boot_returns(self, policy, states, next_states, rewards, dones):
        with torch.no_grad():
            values = policy.critic(states)
            next_values = policy.critic(next_states)

        returns = rewards + self.discount * next_values * (1 - dones.float())
        advantages = returns - values

        return returns, advantages
    
    def sanitize(self, state: np.ndarray):
        return torch.from_numpy(state).float().to(device=self.device).view(1, -1)