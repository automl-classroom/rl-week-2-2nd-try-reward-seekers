from __future__ import annotations

from typing import Any

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
        self.action_space = gym.spaces.Discrete(2)  # 2 actions: 0 and 1
        self.observation_space = gym.spaces.Discrete(2)  # 2 states: 0 and 1
        self.current_state = 0
        self.steps = 0  # count the number of steps
        self.horizon = 10  # max number of steps in an episode

    def get_info(self) -> dict[str, int]:  # get the current state of the environment
        return {"state": self.current_state}

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[int, dict]:  # reset the environment
        self.current_state = 0
        self.steps = 0
        return int(self.current_state), {}

    def step(
        self, action: int
    ) -> tuple[int, float, bool, bool, dict]:  # take a step in the environment
        if not self.action_space.contains(action):  # check if the action is valid
            raise RuntimeError(f"Invalid action: {action}")

        reward = float(action)  # reward is equal to the action taken
        self.current_state = int(
            action
        )  # move to the next state (equal to the action taken)
        self.steps += 1
        terminated = truncated = False
        if self.steps >= self.horizon:  # check if the maximum number of steps
            truncated = True

        return (self.current_state, reward, terminated, truncated, {})

    def get_reward_per_action(self) -> np.ndarray:  # get the reward for each action
        num_action, num_state = self.action_space.n, self.observation_space.n

        reward_matrix = np.zeros(
            (num_state, num_action), dtype=float
        )  # creat a reward matrix

        # fill the matrix with the action taken
        for state in range(num_state):
            for action in range(num_action):
                reward_matrix[state][action] = float(action)
        return reward_matrix

    def get_transition_matrix(self) -> np.ndarray:  # get the transition matrix
        num_action, num_state = self.action_space.n, self.observation_space.n

        transition_matrix = np.zeros(
            (num_state, num_action, num_state)
        )  # creat a transition matrix

        # fill the matrix with the action taken
        for state in range(num_state):
            for action in range(num_action):
                transition_matrix[state][action][action] = (
                    1.0  # determenistic transition
                )
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
        super().__init__(env)  # Initialize the environment
        assert 0.0 <= noise <= 1.0, "Noise must be in [0,1]"
        self.noise = noise  # Noise level
        self.rng = np.random.default_rng(seed)  # generate a random number

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)  # reset the environment
        # Get the (noisy) observation and info from the environment
        return self._noisy_obs(obs), info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(
            action
        )  # Take a step in the environment
        return self._noisy_obs(obs), reward, terminated, truncated, info

    def _noisy_obs(self, obs: int) -> int:
        if (
            self.rng.random() < self.noise
        ):  # Check if the random number is less than the noise level
            # return a random observation from the possible observations
            possible_obs = list(range(self.observation_space.n))
            possible_obs.remove(obs)  # Remove the not noisy observation
            return self.rng.choice(possible_obs)
        return obs
