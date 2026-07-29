import argparse

from next_state_predictor.next_state import (
    evaluate_next_state_predictor,
    train_next_state_predictor,
)
from next_state_predictor.rl import train_rl


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Next state predictor experiments")

    ## RL CONFIG
    ## RL CONFIG
    ## RL CONFIG

    parser.add_argument(
        "--env", type=str, default="CartPole-v1", help="Environment name"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=10, help="Number of episodes to run"
    )
    parser.add_argument(
        "--max_steps", type=int, default=50, help="Maximum number of steps per episode"
    )
    parser.add_argument(
        "--rl_output_dir",
        type=str,
        default="results_dev/rl",
        help="Directory to save RL data",
    )
    parser.add_argument(
        "--rl", type=str, default="DQN", help="RL algorithm to use (e.g., DQN)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=0.99,
        help="Discount factor for future rewards",
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="Optimizer type (adam or sgd)"
    )

    ## NEXT STATE PREDICTOR CONFIG
    ## NEXT STATE PREDICTOR CONFIG
    ## NEXT STATE PREDICTOR CONFIG
    parser.add_argument(
        "--next_state_predictor",
        type=str,
        default="tdlgm",
        help="Should you train a next state predictor?",
    )
    parser.add_argument(
        "--next_state_predictor_train_data",
        type=str,
        default=None,
        help="File to save the next state predictor train and val data.",
    )

    parser.add_argument(
        "--next_state_predictor_test_data",
        type=str,
        default=None,
        help="File to save the next state predictor test data set.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=10,
        help="Number of previous states to consider for next state prediction.",
    )
    parser.add_argument(
        "--next_state_batch_size",
        type=int,
        default=32,
        help="Batch size for training the next state predictor.",
    )
    parser.add_argument(
        "--next_state_num_epochs",
        type=int,
        default=10,
        help="Number of epochs for training the next state predictor.",
    )
    parser.add_argument(
        "--next_state_lr",
        type=float,
        default=0.001,
        help="Learning rate for the next state predictor.",
    )
    parser.add_argument(
        "--next_state_output_dir",
        type=str,
        default="results_dev/predictor",
        help="Directory to save next state predictor data",
    )

    return parser.parse_args()


def _save_results(results, args):
    import json
    import os
    import time

    import torch

    unique_id = time.strftime("%Y%m%d-%H%M%S")

    output_dir = args.next_state_output_dir
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(
        output_dir, f"model_{args.next_state_predictor}_{unique_id}.pt"
    )
    json_path = os.path.join(
        output_dir, f"json_{args.next_state_predictor}_{unique_id}.json"
    )

    to_save = {"results": results["results"], "model": model_path, "config": vars(args)}

    with open(json_path, "w") as f:
        json.dump(to_save, f, indent=4)

    torch.save(results["model"], model_path)


def main():
    print("Hello, World!")

    args = parse_arguments()

    training_data = args.next_state_predictor_train_data
    testing_data = args.next_state_predictor_test_data

    if args.rl is not None:
        training_data = train_rl(args)
        testing_data = train_rl(args)

    if args.next_state_predictor is not None:
        results = {}
        results["results"] = {}
        output = train_next_state_predictor(args, training_data)
        results["results"]["train"] = output["results"]
        results["model"] = output["model"]
        results["results"]["eval"] = evaluate_next_state_predictor(
            args, output["model"], testing_data
        )

        _save_results(results, args)


if __name__ == "__main__":
    main()
