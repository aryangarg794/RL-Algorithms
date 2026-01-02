import numpy as np

from collections import deque

        
def compute_returns(rewards: list, discount: float):
    G = 0 
    returns = []
    for reward in reversed(rewards):
        G = reward + discount * G
        returns.insert(0, G)
    return returns

class RollingAverage:
    def __init__(
        self, 
        window_size: int = 5, 
    ) -> None:
        
        self.window = deque(maxlen=window_size)
        self.averages = []
        self.num_eps = 0 
        
    def increment_ep(self) -> None:
        self.num_eps += 1

    def update(
        self, 
        value: float
    ) -> None:
        self.window.append(float(value))
        self.averages.append(self.get_average)
        
    @property
    def get_average(self) -> float:
        return sum(self.window) / len(self.window) if self.window else 0.0