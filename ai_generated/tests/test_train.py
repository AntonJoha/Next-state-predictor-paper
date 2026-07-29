"""Tests for train module."""

import gymnasium as gym
import pytest
from next_state_predictor.agent import RandomAgent
from next_state_predictor.train import evaluate, train


@pytest.fixture
def env_and_agent():
    env = gym.make("CartPole-v1")
    agent = RandomAgent(env, seed=0)
    yield env, agent
    env.close()


def test_train_returns_correct_number_of_rewards(env_and_agent):
    _, agent = env_and_agent
    rewards = train(agent, n_episodes=5)
    assert len(rewards) == 5


def test_train_rewards_are_positive(env_and_agent):
    _, agent = env_and_agent
    rewards = train(agent, n_episodes=5)
    assert all(r > 0 for r in rewards)


def test_train_respects_max_steps(env_and_agent):
    _, agent = env_and_agent
    rewards = train(agent, n_episodes=5, max_steps=3)
    # With max_steps=3 each episode reward should be at most 3
    assert all(r <= 3 for r in rewards)


def test_evaluate_returns_stats_keys(env_and_agent):
    _, agent = env_and_agent
    stats = evaluate(agent, n_episodes=5)
    assert set(stats.keys()) == {"mean", "std", "min", "max"}


def test_evaluate_stats_consistent(env_and_agent):
    _, agent = env_and_agent
    stats = evaluate(agent, n_episodes=5)
    assert stats["min"] <= stats["mean"] <= stats["max"]
