"""Entry-point script demonstrating the project template."""

from __future__ import annotations

import argparse

import gymnasium as gym

from next_state_predictor.agent import RandomAgent
from next_state_predictor.train import evaluate, train
from next_state_predictor.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Next-state predictor template")
    parser.add_argument(
        "--env",
        default="CartPole-v1",
        help="Gymnasium environment ID (default: CartPole-v1)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment (requires a display)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    render_mode = "human" if args.render else None
    env = gym.make(args.env, render_mode=render_mode)

    agent = RandomAgent(env, seed=args.seed)

    print(f"Environment : {args.env}")
    print(f"Observation : {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    print(f"Running {args.episodes} training episodes …")
    rewards = train(agent, n_episodes=args.episodes)
    print(f"Total rewards per episode: {[round(r, 2) for r in rewards]}")
    print()

    print("Evaluating agent …")
    stats = evaluate(agent, n_episodes=args.episodes)
    for key, value in stats.items():
        print(f"  {key:>4}: {value:.2f}")

    env.close()


if __name__ == "__main__":
    main()
