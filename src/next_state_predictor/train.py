"""Training and evaluation loops for Gymnasium environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from next_state_predictor.agent import Agent, DQNAgent


def train(
    agent: Agent,
    n_episodes: int = 100,
    max_steps: int | None = None,
    render: bool = False,
) -> list[float]:
    """Run *n_episodes* of training and return the per-episode total rewards.

    Args:
        agent: The agent to train.
        n_episodes: Number of episodes to run.
        max_steps: Optional cap on steps per episode (``None`` uses the
            environment's own limit).
        render: Whether to call ``env.render()`` each step.

    Returns:
        A list of total rewards, one entry per episode.
    """
    rewards: list[float] = []

    for _ in range(n_episodes):
        observation, _info = agent.env.reset()
        agent.reset()

        episode_reward = 0.0
        step = 0

        while True:
            if render:
                agent.env.render()

            action = agent.select_action(observation)
            observation, reward, terminated, truncated, _info = agent.env.step(action)
            episode_reward += float(reward)
            step += 1

            if terminated or truncated:
                break
            if max_steps is not None and step >= max_steps:
                break

        rewards.append(episode_reward)

    return rewards


def train_dqn(
    agent: DQNAgent,
    n_episodes: int = 500,
    max_steps: int | None = None,
    render: bool = False,
    db_path: str | None = None,
    db_table: str = "transitions",
    trajectory_length: int = 1,
) -> list[float]:
    """Train a :class:`~next_state_predictor.agent.DQNAgent` and optionally
    persist all collected transitions to SQLite.

    Each step calls :meth:`~next_state_predictor.agent.DQNAgent.observe` so the
    agent can store the transition in its replay buffer and perform a
    gradient-descent update.

    Args:
        agent: The DQN agent to train.
        n_episodes: Number of training episodes.
        max_steps: Optional cap on steps per episode.
        render: Whether to call ``env.render()`` each step.
        db_path: If given, all replay-buffer transitions are saved to this
            SQLite database file after training completes.
        db_table: Table name used when writing to SQLite.
        trajectory_length: Number of recent states to store per transition.

    Returns:
        A list of total rewards, one entry per episode.
    """
    rewards: list[float] = []

    for _ in range(n_episodes):
        observation, _info = agent.env.reset()
        agent.reset()

        episode_reward = 0.0
        step = 0

        while True:
            if render:
                agent.env.render()

            action = agent.select_action(observation)
            next_observation, reward, terminated, truncated, _info = agent.env.step(
                action
            )
            done = terminated or truncated
            agent.observe(observation, action, reward, next_observation, done)

            episode_reward += float(reward)
            observation = next_observation
            step += 1

            if done:
                break
            if max_steps is not None and step >= max_steps:
                break

        rewards.append(episode_reward)

    if db_path is not None:
        agent.save_transitions_to_sqlite(db_path, db_table, trajectory_length)

    return rewards


def evaluate(
    agent: Agent,
    n_episodes: int = 10,
    render: bool = False,
) -> dict[str, float]:
    """Evaluate an agent over *n_episodes* and return summary statistics.

    Args:
        agent: The agent to evaluate.
        n_episodes: Number of episodes to evaluate over.
        render: Whether to call ``env.render()`` each step.

    Returns:
        A dictionary with keys ``mean``, ``std``, ``min``, and ``max``.
    """
    rewards = train(agent, n_episodes=n_episodes, render=render)
    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
    }
