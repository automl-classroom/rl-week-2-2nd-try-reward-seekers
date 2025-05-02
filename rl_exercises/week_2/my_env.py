from __future__ import annotations

import gymnasium as gym
import numpy as np


# ------------- TODO: Implement the following environment -------------
class MyEnv(gym.Env):
    """
    Simple 2-state, 2-action environment with deterministic transitions.

    Actions
    -------
    Discrete(2):
    - 0: move to state 0
    - 1: move to state 1

    Observations
    ------------
    Discrete(2): the current state (0 or 1)

    Reward
    ------
    Equal to the action taken.

    Start/Reset State
    -----------------
    Always starts in state 0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        """Initializes the observation and action space for the environment."""
        # super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(2)
        self.current_state = 0
        self.steps = 0
        self.horizon = 10

    def get_info(self) -> dict[str, int]:
        return {"state": self.current_state}

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[int, dict]:
        self.current_state = 0
        self.steps = 0
        return int(self.current_state), {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise RuntimeError(f"Invalid action: {action}")

        reward = float(action)
        self.current_state = int(action)
        self.steps += 1
        terminated = truncated = False
        if self.steps >= self.horizon:
            truncated = True

        return (self.current_state, reward, terminated, truncated, {})

    def get_reward_per_action(self) -> np.ndarray:
        num_action, num_state = self.action_space.n, self.observation_space.n

        reward_matrix = np.zeros((num_state, num_action), dtype=float)

        for state in range(num_state):
            for action in range(num_action):
                reward_matrix[state][action] = float(action)
        return reward_matrix

    def get_transition_matrix(self) -> np.ndarray:
        num_action, num_state = self.action_space.n, self.observation_space.n

        transition_matrix = np.zeros((num_state, num_action, num_state))

        for state in range(num_state):
            for action in range(num_action):
                transition_matrix[state][action][action] = 1.0
        return transition_matrix


class PartialObsWrapper(gym.Wrapper):
    """Wrapper that makes the underlying env partially observable by injecting
    observation noise: with probability `noise`, the true state is replaced by
    a random (incorrect) observation.

    Parameters
    ----------
    env : gym.Env
        The fully observable base environment.
    noise : float, default=0.1
        Probability in [0,1] of seeing a random wrong observation instead
        of the true one.
    seed : int | None, default=None
        Optional RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: gym.Env, noise: float = 0.1, seed: int | None = None):
        pass
