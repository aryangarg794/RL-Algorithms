import gymnasium as gym
import numpy as np
import torch 

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
        returns: list, 
        device: str = 'cuda'
    ):
        self.states = torch.from_numpy(np.array(states)).float().to(device=device).view(-1, obs_shape)
        self.actions = torch.from_numpy(np.array(actions)).float().to(device=device).view(-1, action_shape)
        self.rewards = torch.from_numpy(np.array(rewards)).float().to(device=device).view(-1, 1)
        self.next_states = torch.from_numpy(np.array(next_states)).float().to(device=device).view(-1, obs_shape)
        self.returns = torch.from_numpy(np.array(returns)).float().to(device=device).view(-1, 1)
    
    
class Runner:
    
    def __init__(
        self,
        env_id: str, 
        device: str = 'cuda'
    ):

        self.device = device
        self.env = gym.make(env_id)
        
    def run_trajectory(self, policy, discount: float):
        states = []
        actions = []
        rewards = []
        next_states = []
        
        done = False
        obs, _ = self.env.reset()
        while not done:
            action = policy.sample_action(self.sanitize(obs)).cpu().numpy().reshape(-1)
            next_obs, reward, trunc, term, _ = self.env.step(action)
            
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_obs)
            
            obs = next_obs
            done = trunc or term

        returns = compute_returns(rewards, discount)
        
        return EpisodeBatch(
            np.prod(self.env.observation_space.shape), 
            np.prod(self.env.action_space.shape),
            states=states, 
            actions=actions, 
            rewards=rewards,
            next_states=next_states, 
            returns=returns, 
            device=self.device
        )
    
    def sanitize(self, state: np.ndarray):
        return torch.from_numpy(state).float().to(device=self.device).view(1, -1)