"""Agent definitions for Gymnasium environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym


class Agent(ABC):
    """Abstract base class for all agents.

    Subclass this and implement :meth:`select_action` to create a custom agent.
    """

    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    @abstractmethod
    def select_action(self, observation: Any) -> Any:
        """Choose an action given the current observation.

        Args:
            observation: The current environment observation.

        Returns:
            An action that is valid in ``self.action_space``.
        """

    def reset(self) -> None:
        """Called at the start of every episode. Override if needed."""


class RandomAgent(Agent):
    """An agent that samples uniformly at random from the action space."""

    def __init__(self, env: gym.Env, seed: int | None = None) -> None:
        super().__init__(env)
        if seed is not None:
            self.action_space.seed(seed)

    def select_action(self, observation: Any) -> Any:  # noqa: ARG002
        return self.action_space.sample()
