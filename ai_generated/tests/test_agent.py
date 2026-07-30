"""Tests for agent module."""

import gymnasium as gym
import pytest
from next_state_predictor.agent import RandomAgent


@pytest.fixture
def cartpole_env():
    env = gym.make("CartPole-v1")
    yield env
    env.close()


def test_random_agent_action_in_space(cartpole_env):
    agent = RandomAgent(cartpole_env, seed=0)
    obs, _ = cartpole_env.reset(seed=0)
    action = agent.select_action(obs)
    assert cartpole_env.action_space.contains(action)


def test_random_agent_reset_does_not_raise(cartpole_env):
    agent = RandomAgent(cartpole_env, seed=0)
    agent.reset()  # should not raise


def test_random_agent_consistent_with_seed():
    env_a = gym.make("CartPole-v1")
    env_b = gym.make("CartPole-v1")
    agent_a = RandomAgent(env_a, seed=42)
    agent_b = RandomAgent(env_b, seed=42)
    obs_a, _ = env_a.reset(seed=0)
    obs_b, _ = env_b.reset(seed=0)
    actions_a = [agent_a.select_action(obs_a) for _ in range(20)]
    actions_b = [agent_b.select_action(obs_b) for _ in range(20)]
    env_a.close()
    env_b.close()
    assert actions_a == actions_b
