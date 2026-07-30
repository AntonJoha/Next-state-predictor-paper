"""Agent definitions for Gymnasium environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from next_state_predictor.replay_buffer import ReplayBuffer


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
        return None


class RandomAgent(Agent):
    """An agent that samples uniformly at random from the action space."""

    def __init__(self, env: gym.Env, seed: int | None = None) -> None:
        super().__init__(env)
        if seed is not None:
            self.action_space.seed(seed)

    def select_action(self, observation: Any) -> Any:  # noqa: ARG002
        return self.action_space.sample()


class _QNetwork(nn.Module):
    """Two-layer MLP for Q-value estimation.

    Args:
        input_dim: Number of input features (flattened observation size).
        hidden_dim: Number of units in the hidden layer.
        output_dim: Number of discrete actions.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class DQNAgent(Agent):
    """Deep Q-Network agent for discrete-action Gymnasium environments.

    Uses a two-layer MLP built with PyTorch, an experience replay buffer,
    and a target network updated periodically.

    Args:
        env: A Gymnasium environment with a discrete action space.
        hidden_dim: Hidden layer width of the Q-network.
        lr: Learning rate for the Adam optimiser.
        gamma: Discount factor.
        epsilon_start: Initial exploration probability.
        epsilon_end: Minimum exploration probability.
        epsilon_decay: Decay constant (steps; higher → slower decay).
        batch_size: Mini-batch size for each optimisation step.
        replay_capacity: Maximum capacity of the replay buffer.
        target_update_freq: Steps between target-network weight copies.
        device: PyTorch device string (e.g. ``"cpu"``, ``"cuda"``).
            Defaults to ``"cuda"`` if available, else ``"cpu"``.
        seed: Optional integer seed for reproducibility.
    """

    def __init__(
        self,
        env: gym.Env,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 500,
        batch_size: int = 64,
        replay_capacity: int = 10_000,
        target_update_freq: int = 100,
        device: str | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(env)
        if not hasattr(env.action_space, "n"):
            msg = "DQNAgent requires a discrete (Discrete) action space."
            raise ValueError(msg)

        obs_dim = int(np.prod(env.observation_space.shape))
        n_actions = int(env.action_space.n)

        self.gamma = gamma
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = int(epsilon_decay)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if seed is not None:
            torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed)
        self._step = 0

        self.q_network = _QNetwork(obs_dim, hidden_dim, n_actions).to(self.device)
        self.target_network = _QNetwork(obs_dim, hidden_dim, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer: ReplayBuffer = ReplayBuffer(capacity=replay_capacity)

    def select_action(self, observation: Any) -> Any:
        """Epsilon-greedy action selection."""
        if self._rng.random() < self._epsilon():
            return self.action_space.sample()
        obs = torch.tensor(
            np.array(observation, dtype=np.float32).flatten(),
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(obs)
        return int(q_values.argmax(dim=1).item())

    def reset(self) -> None:
        """Reset per-episode state (no-op for DQN)."""

    def observe(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> float | None:
        """Store a transition and perform one optimisation step.

        Args:
            state: Observation before the action.
            action: Action that was taken.
            reward: Reward received.
            next_state: Observation after the action.
            done: Whether the episode ended.

        Returns:
            The Bellman loss for this step, or ``None`` if the replay buffer
            does not yet contain enough transitions to form a batch.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
        self._step += 1

        if len(self.replay_buffer) < self.batch_size:
            return None

        transitions = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(
            np.array([t.state for t in transitions]), device=self.device
        )
        actions = torch.tensor(
            np.array([t.action for t in transitions], dtype=np.int64),
            device=self.device,
        )
        rewards = torch.tensor(
            np.array([t.reward for t in transitions], dtype=np.float32),
            device=self.device,
        )
        next_states = torch.tensor(
            np.array([t.next_state for t in transitions]), device=self.device
        )
        dones = torch.tensor(
            np.array([t.done for t in transitions], dtype=np.float32),
            device=self.device,
        )

        q_pred = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.target_network(next_states).max(dim=1).values
            targets = rewards + self.gamma * q_next * (1.0 - dones)

        loss = self.loss_fn(q_pred, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self._step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return float(loss.item())

    def save_transitions_to_sqlite(
        self,
        db_path: str,
        table: str = "transitions",
        trajectory_length: int = 1,
    ) -> None:
        """Persist all replay-buffer transitions to a SQLite database.

        Args:
            db_path: Path to the SQLite database file (created if absent).
            table: Table name to write to (created if absent).
            trajectory_length: Number of recent states to store per transition.
        """
        self.replay_buffer.save_to_sqlite(db_path, table, trajectory_length)

    def _epsilon(self) -> float:
        """Current exploration probability (exponential decay)."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -self._step / self.epsilon_decay
        )
