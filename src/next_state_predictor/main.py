
import argparse

from next_state_predictor.rl import train_rl
from next_state_predictor.next_state import train_next_state_predictor


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Next state predictor experiments")

    ## RL CONFIG
    ## RL CONFIG
    ## RL CONFIG

    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--num_episodes", type=int, default=500, help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum number of steps per episode")
    parser.add_argument("--rl_output_dir", type=str, default="rl_data", help="Directory to save RL data")
    parser.add_argument("--rl", type=str, default="DQN", help="RL algorithm to use (e.g., DQN)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for the optimizer")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer type (adam or sgd)")


    ## NEXT STATE PREDICTOR CONFIG
    ## NEXT STATE PREDICTOR CONFIG
    ## NEXT STATE PREDICTOR CONFIG
    parser.add_argument("--next_state_predictor", type=str, default=None, help="Should you train a next state predictor?")
    parser.add_argument("--next_state_predictor_input", type=str, default=None, help="File to save the next state predictor input data.")
    parser.add_argument("--lookback", type=int, default=10, help="Number of previous states to consider for next state prediction.")
    parser.add_argument("--next_state_batch_size", type=int, default=32, help="Batch size for training the next state predictor.")
    parser.add_argument("--next_state_num_epochs", type=int, default=10, help="Number of epochs for training the next state predictor.")
    parser.add_argument("--next_state_lr", type=float, default=0.001, help="Learning rate for the next state predictor.")

    return parser.parse_args()






def main():
    print("Hello, World!")
    
    args = parse_arguments()

    predictor_input = args.next_state_predictor_input

    if args.rl is not None:
        predictor_input = train_rl(args)

    if args.next_state_predictor is not None:
        train_next_state_predictor(args, predictor_input)


if __name__ == "__main__":
    main()
