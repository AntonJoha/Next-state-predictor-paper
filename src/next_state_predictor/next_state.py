import argparse
import os
import torch
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader, random_split


def _prepare_trajectories(lookback: int, trajectories) -> np.ndarray:
    
    shape = trajectories.shape
    to_return = np.zeros((shape[0], lookback, shape[2]))

    for i, trajectory in enumerate(trajectories):

        temp= [state for state in trajectory if not np.isnan(state).any()]
        if len(temp) >= lookback:
            temp = temp[-lookback:]  # Keep only the last lookback + 1 states
        else:
            # If there are not enough states, pad with zeros
            temp = [np.zeros(shape[2])]*(lookback-len(temp)) + temp
        to_return[i]= np.array(temp)
    return to_return



class NextStateDataset(Dataset):
    def __init__(self, states, actions, rewards, trajectories, next_states, lookback):
        self.lookback = lookback
        self.trajectories = _prepare_trajectories(lookback, trajectories)
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.states = states


    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]
        reward = self.rewards[idx]
        state = self.states[idx]
        next_state = self.next_states[idx]
        action = self.actions[idx]

        return torch.tensor(trajectory, dtype=torch.float32), torch.tensor(action, dtype=torch.long), torch.tensor(reward, dtype=torch.float32), torch.tensor(state, dtype=torch.float32), torch.tensor(next_state, dtype=torch.float32)



def _get_tdlgm_model(args: argparse.Namespace, dataset: NextStateDataset):
    from next_state_predictor.tdlgm import tDLGM

    input_dim = dataset["state_trajectories"].shape[2]   # Flatten the trajectory
    output_dim = dataset["next_states"].shape[1]  # Assuming next_states is of shape (N, state_dim)

    layers = args.tdlg_layers if hasattr(args, 'tdlg_layers') else 2
    hidden_size = args.tdlg_hidden_size if hasattr(args, 'tdlg_hidden_size') else 64
    latent_dim = args.tdlg_latent_dim if hasattr(args, 'tdlg_latent_dim') else 32

    return tDLGM(input_dim=input_dim, output_dim=output_dim, layers=layers, hidden_size=hidden_size, latent_dim=latent_dim)




def _get_basic_mlp_model(args: argparse.Namespace, dataset: NextStateDataset):



    input_dim = dataset["state_trajectories"].shape[2] * args.lookback  # Flatten the trajectory
    output_dim = dataset["next_states"].shape[1]  # Assuming next_states is of shape (N, state_dim)


    class BasicMLP(torch.nn.Module):
        def __init__(self, input_dim, output_dim):
            super(BasicMLP, self).__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, output_dim)
            )
            self.mean = torch.nn.Sequential(
                torch.nn.Linear(output_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, output_dim)
            )

            self.var = torch.nn.Sequential(
                torch.nn.Linear(output_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, output_dim)
            )
            self.loss = torch.nn.GaussianNLLLoss()

        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten the trajectory for MLP input
            return self.model(x)

        def train_step(self, trajectory, y, optimizer):
            optimizer.zero_grad()
            output = self.forward(trajectory)
            mean = self.mean(output)
            var = torch.exp(self.var(output))  # Ensure variance is positive
            loss = self.loss(mean, y, var)
            loss.backward()
            optimizer.step()
            return loss.item()
    
    model = BasicMLP(input_dim, output_dim)
    return model


def _get_model(args: argparse.Namespace, predictor_input: NextStateDataset):

    # It is easier to initialize the model with the dataset to get the input and output dimensions correct.
    dataset = np.load(predictor_input)

    if args.next_state_predictor == "mlp":
        return _get_basic_mlp_model(args, dataset)
    if args.next_state_predictor == "tdlgm":
        return _get_tdlgm_model(args, dataset)


def _get_dataset(args: argparse.Namespace, predictor_input: str, percent_split: float) -> NextStateDataset:


    dataset = np.load(predictor_input)


    dataset = NextStateDataset(
        states=dataset['states'],
        actions=dataset['actions'],
        rewards=dataset['rewards'],
        trajectories=dataset['state_trajectories'],
        next_states=dataset['next_states'],
        lookback=args.lookback
    )

    train_dataset, val_dataset = random_split(dataset, [int((1-percent_split) * len(dataset)), len(dataset) - int((1-percent_split) * len(dataset))])

    return DataLoader(train_dataset, shuffle=True,
                      batch_size=args.batch_size if hasattr(args, 'batch_size') else 32), DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size if hasattr(args, 'batch_size') else 32)




def _test_loss(model: torch.nn.Module, dataset: DataLoader) -> float:
    model.eval()
    eval_list = []
    with torch.no_grad():
        for _, (trajectory, _, _, _, next_state) in enumerate(dataset):
            output = model.get_loss(trajectory, next_state)[1]
            eval_list.append(output)

    return eval_list


def _train(args: argparse.Namespace, model: torch.nn.Module, train_dataset: DataLoader, test_dataset: DataLoader) -> torch.nn.Module:

    optimizer = torch.optim.Adam(model.parameters(), lr=args.next_state_lr if hasattr(args, 'next_state_lr') else 0.001)

    num_epochs = args.next_state_num_epochs if hasattr(args, 'next_state_num_epochs') else 10

    loss_history = []
    test_loss_history = []

    
    model.train()
    for epoch in range(num_epochs):
        loss_epoch = []
        for _, (trajectory, _, _, _, next_state) in enumerate(train_dataset):
            loss = model.train_step(trajectory, next_state, optimizer)
            loss_epoch.append(loss)

        test_loss_history.append(_test_loss(model, test_dataset))
        loss_history.append(loss_epoch)

            
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss} Eval Loss: {np.mean(test_loss_history[-1])}")


    return model,{"loss_history": loss_history}


def evaluate_next_state_predictor(args: argparse.Namespace, model: torch.nn.Module, predictor_input: str) -> dict:
    _, test_dataset = _get_dataset(args, predictor_input, percent_split=0.2)
    eval_results = _test_loss(model, test_dataset)
    return {"eval_results": eval_results}


def train_next_state_predictor(args: argparse.Namespace, predictor_input: str) -> None:
    os.makedirs(args.rl_output_dir, exist_ok=True)

    train_dataset, test_dataset = _get_dataset(args, predictor_input, percent_split=0.2)
    
    model = _get_model(args, predictor_input)

    model, results = _train(args, model, train_dataset, test_dataset)

    return {"model": model, "results": results}



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Next State Predictor")
    parser.add_argument(
        "--rl_output_dir", type=str, default="rl_data", help="Directory to load RL data"
    )
    parser.add_argument(
        "--next_state_predictor_input", type=str, default=None, help="File to save the next state predictor input data."
    )
    args = parser.parse_args()

    train_next_state_predictor(args, args.next_state_predictor_input)
