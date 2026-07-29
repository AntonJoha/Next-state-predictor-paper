

import argparse
import os
import time
from collections.abc import Callable, Iterable
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from next_state_predictor.dqn_agent import DQNAgent, ReplayBuffer


def _get_optimizer_lambda(args: argparse.Namespace) -> Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer]:
    if hasattr(args, "optimizer") and args.optimizer == "adam":
        return lambda parameters: torch.optim.Adam(parameters, lr=args.lr if hasattr(args, "lr") else 0.001)
    elif hasattr(args, "optimizer") and args.optimizer == "sgd":
        return lambda parameters: torch.optim.SGD(parameters, lr=args.lr if hasattr(args, "lr") else 0.001)
    else:
        raise ValueError("Unsupported optimizer type. Please choose 'adam' or 'sgd'.")


def _get_dqn_model(args: argparse.Namespace, env: gym.Env) -> DQNAgent:

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    conf: dict[str, Any] = {
        "input": input_dim,
        "output": output_dim,
        "layers": [256, 256],
        "target_network": True,
        "lr": args.lr if hasattr(args, "lr") else 0.001,
        "discount": args.discount if hasattr(args, "discount") else 0.99,
        "optimizer": _get_optimizer_lambda(args)
    }

    return DQNAgent(conf)


def _make_model(args: argparse.Namespace, env: gym.Env) -> torch.nn.Module:
    if args.rl == "DQN":
        return _get_dqn_model(args, env)


    raise ValueError(f"Unsupported RL algorithm: {args.rl}. Please choose 'DQN'.")


def _evaluate_model(model: torch.nn.Module, env: gym.Env, num_episodes: int = 10) -> float:
    total_reward = 0.0

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action: int = model.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state

            if terminated or truncated:
                done = True
                break

    average_reward = total_reward / num_episodes
    return average_reward


def train_rl(args) -> None:

    os.makedirs(args.rl_output_dir, exist_ok=True)

    output_file = os.path.join(args.rl_output_dir,  f"length_{args.num_episodes}_id_{str(time.time())}.npz")
    env: gym.Env = gym.make(args.env)

    model: torch.nn.Module = _make_model(args, env)
    buffer: ReplayBuffer = ReplayBuffer(10000)


    states = []
    actions = []
    rewards = []
    next_states = []
    state_trajectories = []
    epsilon = 1  # Initial exploration rate

    for episode in range(args.num_episodes):
        state, _ = env.reset()
        done = False
        step = 0

        state_trajectory = np.zeros((args.max_steps, env.observation_space.shape[0] +env.action_space.n))  # +1 for reward
        state_trajectory.fill(np.nan)  # Fill with NaN for unused steps
        while not done and step < args.max_steps:
            action: int = model.select_action(state, epsilon=epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            buffer.add([state,action,reward,next_state, done])


            model.replay(buffer, batch_size=64, target_network=True)

            state_trajectory[step, :env.observation_space.shape[0]] = state
            state_trajectory[step, env.observation_space.shape[0]:env.observation_space.shape[0]+env.action_space.n] = action
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            state_trajectories.append(state_trajectory.copy())  


            state = next_state
            step += 1

            if done:    
                break


        score = _evaluate_model(model, env, num_episodes=20)
        epsilon = max(0.01, epsilon * 0.995)  # Decay epsilon
        if (episode) % 10 == 0:
            model.update_target_q_network()  # Update target network periodically
        print(f"Episode {episode + 1}/{args.num_episodes}, Score: {score}, Epsilon: {epsilon:.4f}")


    # Save the state, action, reward tuple
    np.savez(
        output_file,
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        state_trajectories=state_trajectories
    )

    env.close()
    return output_file


if __name__ == "__main__":
    import argparse
    import os

    import numpy as np

    parser = argparse.ArgumentParser(description="Train RL agent and save data")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--num_episodes", type=int, default=500, help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum number of steps per episode")
    parser.add_argument("--rl_output_dir", type=str, default="rl_data", help="Directory to save RL data")
    parser.add_argument("--rl", type=str, default="DQN", help="RL algorithm to use (e.g., DQN)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for the optimizer")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer type (adam or sgd)")

    args = parser.parse_args()
    train_rl(args)
