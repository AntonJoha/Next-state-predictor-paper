"""Agent definitions for Gymnasium environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np

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


class RandomAgent(Agent):
    """An agent that samples uniformly at random from the action space."""

    def __init__(self, env: gym.Env, seed: int | None = None) -> None:
        super().__init__(env)
        if seed is not None:
            self.action_space.seed(seed)

    def select_action(self, observation: Any) -> Any:  # noqa: ARG002
        return self.action_space.sample()


# ---------------------------------------------------------------------------
# DQN (Deep Q-Network) — numpy-only implementation
# ---------------------------------------------------------------------------


class _NumpyQNetwork:
    """Two-layer MLP for Q-value estimation implemented with plain NumPy.

    Args:
        input_dim: Number of input features (flattened observation size).
        hidden_dim: Number of units in the hidden layer.
        output_dim: Number of discrete actions.
        lr: Learning rate for gradient descent.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        lr: float = 1e-3,
    ) -> None:
        self.lr = lr
        # He / Xavier initialisation
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward pass; *x* may be a single row or a batch (shape ``[B, D]``)."""
        h = np.maximum(0.0, x @ self.W1 + self.b1)  # ReLU
        return h @ self.W2 + self.b2

    def update(
        self, x: np.ndarray, targets: np.ndarray, actions: np.ndarray
    ) -> float:
        """One gradient-descent step on the TD-error for the chosen actions.

        Args:
            x: Batch of states, shape ``[B, D]``.
            targets: 1-D array of target Q-values, shape ``[B]``.
            actions: 1-D integer array of chosen actions, shape ``[B]``.

        Returns:
            Mean-squared Bellman error (scalar).
        """
        batch_size = x.shape[0]

        # Forward
        z1 = x @ self.W1 + self.b1
        h1 = np.maximum(0.0, z1)
        q = h1 @ self.W2 + self.b2

        q_pred = q[np.arange(batch_size), actions]
        loss = float(np.mean((q_pred - targets) ** 2))

        # Backward
        dq = np.zeros_like(q)
        dq[np.arange(batch_size), actions] = 2.0 * (q_pred - targets) / batch_size

        dW2 = h1.T @ dq
        db2 = dq.sum(axis=0)

        dh1 = dq @ self.W2.T
        dz1 = dh1 * (z1 > 0.0)  # ReLU gradient

        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        return loss

    def copy_weights_from(self, other: _NumpyQNetwork) -> None:
        """Copy weights from *other* into this network (target-network sync)."""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()


class DQNAgent(Agent):
    """Deep Q-Network agent for discrete-action Gymnasium environments.

    Uses a two-layer MLP implemented in pure NumPy, an experience replay
    buffer, and a target network updated periodically.

    Args:
        env: A Gymnasium environment with a discrete action space.
        hidden_dim: Hidden layer width of the Q-network.
        lr: Learning rate.
        gamma: Discount factor.
        epsilon_start: Initial exploration probability.
        epsilon_end: Minimum exploration probability.
        epsilon_decay: Decay constant (steps; higher → slower decay).
        batch_size: Mini-batch size for each optimisation step.
        replay_capacity: Maximum capacity of the replay buffer.
        target_update_freq: Steps between target-network weight copies.
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

        self._rng = np.random.default_rng(seed)
        self._step = 0

        self.q_network = _NumpyQNetwork(obs_dim, hidden_dim, n_actions, lr)
        self.target_network = _NumpyQNetwork(obs_dim, hidden_dim, n_actions, lr)
        self.target_network.copy_weights_from(self.q_network)

        self.replay_buffer: ReplayBuffer = ReplayBuffer(capacity=replay_capacity)

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def select_action(self, observation: Any) -> Any:
        """Epsilon-greedy action selection."""
        if self._rng.random() < self._epsilon():
            return self.action_space.sample()
        obs = np.array(observation, dtype=np.float32).flatten()[np.newaxis, :]
        q_values = self.q_network.predict(obs)[0]
        return int(np.argmax(q_values))

    def reset(self) -> None:
        """No per-episode state to reset for DQN."""

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

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
        states = np.array([t.state for t in transitions], dtype=np.float32)
        actions = np.array([t.action for t in transitions], dtype=np.int32)
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        next_states = np.array([t.next_state for t in transitions], dtype=np.float32)
        dones = np.array([t.done for t in transitions], dtype=np.float32)

        q_next = self.target_network.predict(next_states)
        targets = rewards + self.gamma * np.max(q_next, axis=1) * (1.0 - dones)

        loss = self.q_network.update(states, targets, actions)

        if self._step % self.target_update_freq == 0:
            self.target_network.copy_weights_from(self.q_network)

        return loss

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_transitions_to_sqlite(
        self, db_path: str, table: str = "transitions"
    ) -> None:
        """Persist all replay-buffer transitions to a SQLite database.

        Args:
            db_path: Path to the SQLite database file (created if absent).
            table: Table name to write to (created if absent).
        """
        self.replay_buffer.save_to_sqlite(db_path, table)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _epsilon(self) -> float:
        """Current exploration probability (exponential decay)."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -self._step / self.epsilon_decay
        )

