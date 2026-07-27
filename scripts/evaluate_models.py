"""Evaluate and compare next-state predictor models.

This script:

1. Collects state transitions from a Gymnasium environment (default: CartPole-v1).
2. Trains three next-state predictor models — tDLGM, VRNN, and DiffusionPredictor.
3. Evaluates each model independently and saves individual diagnostic plots.
4. Produces cross-model comparison plots.
5. Saves all figures and a JSON metrics summary to *output_dir*.

Usage
-----
    python scripts/evaluate_models.py [options]

Run ``python scripts/evaluate_models.py --help`` for a full option list.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # headless backend — must come before pyplot import
import matplotlib.pyplot as plt  # noqa: E402

from next_state_predictor.diffusion import DiffusionPredictor
from next_state_predictor.tdlgm import tDLGM
from next_state_predictor.utils import set_seed
from next_state_predictor.vrnn import VRNN

# ── colour palette ─────────────────────────────────────────────────────────────

_COLOURS = {
    "tDLGM": "#4C72B0",
    "VRNN": "#DD8452",
    "DiffusionPredictor": "#55A868",
}


# ── data collection ────────────────────────────────────────────────────────────


def collect_transitions(
    env_id: str,
    n_episodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out a random policy and collect (state, next_state) pairs.

    Args:
        env_id: Gymnasium environment identifier.
        n_episodes: Number of episodes to run.
        seed: RNG seed.

    Returns:
        states, next_states: Arrays of shape ``(N, obs_dim)``.
    """
    env = gym.make(env_id)
    env.action_space.seed(seed)

    states: list[np.ndarray] = []
    next_states: list[np.ndarray] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        while True:
            action = env.action_space.sample()
            next_obs, _, terminated, truncated, _ = env.step(action)
            states.append(np.array(obs, dtype=np.float32))
            next_states.append(np.array(next_obs, dtype=np.float32))
            obs = next_obs
            if terminated or truncated:
                break

    env.close()
    return np.stack(states), np.stack(next_states)


def build_sequences(
    states: np.ndarray,
    next_states: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide a window over the flat transition list to build sequence batches.

    Args:
        states: ``(N, obs_dim)`` array of current observations.
        next_states: ``(N, obs_dim)`` array of next observations.
        seq_len: Length of each input sequence window.

    Returns:
        x:   ``(M, seq_len, obs_dim)`` current-step sequences.
        x_1: ``(M, seq_len, obs_dim)`` next-step sequences.
        y:   ``(M, 1, obs_dim)`` target next states.
    """
    N, obs_dim = states.shape
    if N <= seq_len:
        msg = f"Need more than seq_len={seq_len} transitions; got {N}."
        raise ValueError(msg)

    xs, x1s, ys = [], [], []
    for i in range(N - seq_len):
        xs.append(states[i : i + seq_len])
        x1s.append(next_states[i : i + seq_len])
        ys.append(next_states[i + seq_len - 1 : i + seq_len])

    return np.stack(xs), np.stack(x1s), np.stack(ys)


def normalise(
    arr: np.ndarray,
    lo: np.ndarray | None = None,
    hi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-max normalise *arr* to [0, 1] along the feature axis.

    Args:
        arr: Input array ``(..., obs_dim)``.
        lo:  Per-feature minimum (computed from *arr* if ``None``).
        hi:  Per-feature maximum (computed from *arr* if ``None``).

    Returns:
        normalised, lo, hi
    """
    if lo is None:
        lo = arr.reshape(-1, arr.shape[-1]).min(0)
    if hi is None:
        hi = arr.reshape(-1, arr.shape[-1]).max(0)
    scale = np.where(hi - lo > 0, hi - lo, 1.0)
    return (arr - lo) / scale, lo, hi


def to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32, device=device)


# ── model factory ──────────────────────────────────────────────────────────────


def build_model(
    name: str,
    input_dim: int,
    hidden_size: int,
    latent_dim: int,
    output_dim: int,
    seq_len: int,
    device: torch.device,
) -> torch.nn.Module:
    """Instantiate a named next-state predictor model.

    Args:
        name: One of ``"tDLGM"``, ``"VRNN"``, or ``"DiffusionPredictor"``.
        input_dim: Feature dimension of each observation.
        hidden_size: Hidden layer / GRU / LSTM size.
        latent_dim: Latent space dimensionality.
        output_dim: Predicted next-state dimensionality.
        seq_len: Input sequence length.
        device: PyTorch device.

    Returns:
        An instantiated model moved to *device*.
    """
    if name == "tDLGM":
        return tDLGM(
            input_dim=input_dim,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            output_dim=output_dim,
            layers=2,
            seq_len=seq_len,
            device=device,
        ).to(device)
    if name == "VRNN":
        return VRNN(
            input_dim=input_dim,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            output_dim=output_dim,
            device=device,
        ).to(device)
    if name == "DiffusionPredictor":
        return DiffusionPredictor(
            input_dim=input_dim,
            hidden_size=hidden_size,
            output_dim=output_dim,
            T=200,
            t_emb_dim=16,
            device=device,
        ).to(device)
    msg = f"Unknown model: {name}"
    raise ValueError(msg)


# ── training ───────────────────────────────────────────────────────────────────


def train_model(
    model: torch.nn.Module,
    x_train: torch.Tensor,
    x1_train: torch.Tensor,
    y_train: torch.Tensor,
    n_epochs: int,
    lr: float,
    batch_size: int,
) -> list[float]:
    """Train *model* for *n_epochs* and return the per-epoch mean loss.

    Args:
        model: A next-state predictor with a ``train_step`` method.
        x_train: ``(N, seq_len, input_dim)`` input sequences.
        x1_train: ``(N, seq_len, input_dim)`` next-step sequences.
        y_train: ``(N, 1, output_dim)`` target next states.
        n_epochs: Number of complete passes over the data.
        lr: Learning rate for Adam.
        batch_size: Mini-batch size.

    Returns:
        List of mean losses, one value per epoch.
    """
    optimizer = torch.optim.Adam(model.get_parameters(), lr=lr)
    N = x_train.size(0)
    epoch_losses: list[float] = []

    for _ in range(n_epochs):
        perm = torch.randperm(N)
        batch_losses: list[float] = []
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            loss = model.train_step(x_train[idx], x1_train[idx], y_train[idx], optimizer)
            batch_losses.append(loss)
        epoch_losses.append(float(np.mean(batch_losses)))

    return epoch_losses


# ── evaluation ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    x_test: torch.Tensor,
    x1_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int,
) -> dict[str, object]:
    """Compute predictions and error metrics on a held-out test set.

    Args:
        model: A trained next-state predictor with a ``forward`` method and
            a ``get_loss`` method.
        x_test: ``(N, seq_len, input_dim)`` test sequences.
        x1_test: ``(N, seq_len, input_dim)`` next-step test sequences.
        y_test: ``(N, 1, output_dim)`` test targets.
        batch_size: Inference batch size.

    Returns:
        Dictionary with keys:
        - ``predictions``: ``(N, output_dim)`` numpy array.
        - ``targets``: ``(N, output_dim)`` numpy array.
        - ``mse_per_sample``: ``(N,)`` per-sample MSE.
        - ``mae_per_sample``: ``(N,)`` per-sample MAE.
        - ``mean_mse``: scalar mean MSE.
        - ``mean_mae``: scalar mean MAE.
        - ``test_loss``: mean ``get_loss`` over batches.
    """
    model.eval()
    N = x_test.size(0)
    preds_list: list[np.ndarray] = []
    test_losses: list[float] = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xb = x_test[start:end]
        x1b = x1_test[start:end]
        yb = y_test[start:end]

        pred = model(xb)
        preds_list.append(pred.cpu().numpy())
        test_losses.append(model.get_loss(xb, x1b, yb))

    predictions = np.concatenate(preds_list, axis=0)  # (N, output_dim)
    targets = y_test.squeeze(1).cpu().numpy()  # (N, output_dim)

    diff = predictions - targets
    mse_per_sample = (diff**2).mean(axis=1)
    mae_per_sample = np.abs(diff).mean(axis=1)

    model.train()
    return {
        "predictions": predictions,
        "targets": targets,
        "mse_per_sample": mse_per_sample,
        "mae_per_sample": mae_per_sample,
        "mean_mse": float(mse_per_sample.mean()),
        "mean_mae": float(mae_per_sample.mean()),
        "test_loss": float(np.mean(test_losses)),
    }


# ── individual plots ───────────────────────────────────────────────────────────


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_individual(
    model_name: str,
    loss_history: list[float],
    eval_result: dict[str, object],
    output_dir: Path,
) -> None:
    """Save four diagnostic plots for a single model.

    Plots saved:
    - ``<model>_loss_curve.png``: training loss per epoch.
    - ``<model>_pred_vs_actual.png``: scatter of predicted vs actual per dimension.
    - ``<model>_error_per_dim.png``: mean absolute error per output dimension.
    - ``<model>_error_histogram.png``: histogram of per-sample MSE errors.

    Args:
        model_name: Display name of the model.
        loss_history: Per-epoch mean training loss.
        eval_result: Return value of :func:`evaluate_model`.
        output_dir: Directory in which to save figures.
    """
    colour = _COLOURS.get(model_name, "steelblue")
    predictions: np.ndarray = eval_result["predictions"]
    targets: np.ndarray = eval_result["targets"]
    mse_per_sample: np.ndarray = eval_result["mse_per_sample"]
    output_dim = predictions.shape[1]

    # 1. Training loss curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_history, color=colour, linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title(f"{model_name} — training loss")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, output_dir / f"{model_name}_loss_curve.png")

    # 2. Predicted vs actual scatter (one panel per output dimension)
    ncols = min(output_dim, 4)
    nrows = (output_dim + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    for d in range(output_dim):
        row, col = divmod(d, ncols)
        ax = axes[row][col]
        ax.scatter(targets[:, d], predictions[:, d], alpha=0.3, s=10, color=colour)
        lo = min(targets[:, d].min(), predictions[:, d].min())
        hi = max(targets[:, d].max(), predictions[:, d].max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Dim {d}")
        ax.grid(True, alpha=0.3)
    # hide unused axes
    for d in range(output_dim, nrows * ncols):
        row, col = divmod(d, ncols)
        axes[row][col].set_visible(False)
    fig.suptitle(f"{model_name} — predicted vs actual", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, output_dir / f"{model_name}_pred_vs_actual.png")

    # 3. Mean absolute error per output dimension
    mae_per_dim = np.abs(predictions - targets).mean(axis=0)
    fig, ax = plt.subplots(figsize=(max(5, output_dim * 1.2), 4))
    bars = ax.bar(range(output_dim), mae_per_dim, color=colour)
    ax.bar_label(bars, fmt="%.4f", fontsize=8)
    ax.set_xlabel("Output dimension")
    ax.set_ylabel("Mean absolute error")
    ax.set_title(f"{model_name} — MAE per output dimension")
    ax.set_xticks(range(output_dim))
    ax.grid(True, axis="y", alpha=0.3)
    _save_fig(fig, output_dir / f"{model_name}_error_per_dim.png")

    # 4. Per-sample MSE histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(mse_per_sample, bins=40, color=colour, edgecolor="white", alpha=0.85)
    ax.axvline(mse_per_sample.mean(), color="black", linestyle="--", label=f"mean={mse_per_sample.mean():.4f}")
    ax.set_xlabel("Per-sample MSE")
    ax.set_ylabel("Count")
    ax.set_title(f"{model_name} — prediction error distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, output_dir / f"{model_name}_error_histogram.png")


# ── comparison plots ───────────────────────────────────────────────────────────


def plot_comparison(
    all_loss_histories: dict[str, list[float]],
    all_eval_results: dict[str, dict],
    output_dir: Path,
) -> None:
    """Save cross-model comparison figures.

    Plots saved:
    - ``comparison_loss_curves.png``: all training loss curves on one axes.
    - ``comparison_mse_bar.png``: mean test-set MSE bar chart.
    - ``comparison_mae_bar.png``: mean test-set MAE bar chart.
    - ``comparison_mse_boxplot.png``: box plot of per-sample MSE distributions.

    Args:
        all_loss_histories: ``{model_name: loss_history}`` mapping.
        all_eval_results: ``{model_name: eval_result}`` mapping.
        output_dir: Directory in which to save figures.
    """
    model_names = list(all_loss_histories.keys())
    colours = [_COLOURS.get(n, "#333333") for n in model_names]

    # 1. Loss curves
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, colour in zip(model_names, colours):
        ax.plot(all_loss_histories[name], label=name, color=colour, linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("Training loss — all models")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, output_dir / "comparison_loss_curves.png")

    # 2. MSE bar chart
    mean_mses = [all_eval_results[n]["mean_mse"] for n in model_names]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(model_names, mean_mses, color=colours)
    ax.bar_label(bars, fmt="%.4f")
    ax.set_ylabel("Mean MSE (test set)")
    ax.set_title("Mean prediction MSE — all models")
    ax.grid(True, axis="y", alpha=0.3)
    _save_fig(fig, output_dir / "comparison_mse_bar.png")

    # 3. MAE bar chart
    mean_maes = [all_eval_results[n]["mean_mae"] for n in model_names]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(model_names, mean_maes, color=colours)
    ax.bar_label(bars, fmt="%.4f")
    ax.set_ylabel("Mean MAE (test set)")
    ax.set_title("Mean prediction MAE — all models")
    ax.grid(True, axis="y", alpha=0.3)
    _save_fig(fig, output_dir / "comparison_mae_bar.png")

    # 4. Box plot of per-sample MSE
    mse_data = [all_eval_results[n]["mse_per_sample"] for n in model_names]
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(
        mse_data,
        tick_labels=model_names,
        patch_artist=True,
        notch=False,
        medianprops={"color": "black", "linewidth": 2},
    )
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)
    ax.set_ylabel("Per-sample MSE")
    ax.set_title("MSE distribution — all models")
    ax.grid(True, axis="y", alpha=0.3)
    _save_fig(fig, output_dir / "comparison_mse_boxplot.png")

    # 5. Per-dimension MAE comparison (grouped bar)
    output_dim = all_eval_results[model_names[0]]["predictions"].shape[1]
    x = np.arange(output_dim)
    width = 0.8 / len(model_names)
    fig, ax = plt.subplots(figsize=(max(6, output_dim * 2), 5))
    for i, (name, colour) in enumerate(zip(model_names, colours)):
        preds = all_eval_results[name]["predictions"]
        targets = all_eval_results[name]["targets"]
        mae_per_dim = np.abs(preds - targets).mean(axis=0)
        offset = (i - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, mae_per_dim, width, label=name, color=colour, alpha=0.85)
    ax.set_xlabel("Output dimension")
    ax.set_ylabel("Mean absolute error")
    ax.set_title("MAE per output dimension — all models")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save_fig(fig, output_dir / "comparison_mae_per_dim.png")


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate next-state predictor models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", default="CartPole-v1", help="Gymnasium environment ID")
    parser.add_argument("--episodes", type=int, default=50, help="Collection episodes")
    parser.add_argument("--seq-len", type=int, default=4, help="Input sequence length")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per model")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden layer width")
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent space dimension")
    parser.add_argument("--test-split", type=float, default=0.2, help="Fraction of data for test")
    parser.add_argument("--output-dir", default="results", help="Directory to save outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tDLGM", "VRNN", "DiffusionPredictor"],
        choices=["tDLGM", "VRNN", "DiffusionPredictor"],
        help="Models to evaluate",
    )
    return parser.parse_args()


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device      : {device}")
    print(f"Environment : {args.env}")
    print(f"Output dir  : {output_dir.resolve()}")
    print()

    # ── collect transitions ────────────────────────────────────────────────────
    print(f"Collecting transitions ({args.episodes} episodes) …")
    states, next_states = collect_transitions(args.env, args.episodes, args.seed)
    print(f"  {len(states)} transitions collected; obs_dim={states.shape[1]}")

    # ── normalise ──────────────────────────────────────────────────────────────
    all_obs = np.concatenate([states, next_states], axis=0)
    _, lo, hi = normalise(all_obs)
    states_n, _, _ = normalise(states, lo, hi)
    next_states_n, _, _ = normalise(next_states, lo, hi)

    # ── build sequences ────────────────────────────────────────────────────────
    x_all, x1_all, y_all = build_sequences(states_n, next_states_n, args.seq_len)
    input_dim = output_dim = x_all.shape[2]
    N = x_all.shape[0]
    split = int(N * (1.0 - args.test_split))

    perm = np.random.default_rng(args.seed).permutation(N)
    train_idx, test_idx = perm[:split], perm[split:]

    x_train = to_tensor(x_all[train_idx], device)
    x1_train = to_tensor(x1_all[train_idx], device)
    y_train = to_tensor(y_all[train_idx], device)

    x_test = to_tensor(x_all[test_idx], device)
    x1_test = to_tensor(x1_all[test_idx], device)
    y_test = to_tensor(y_all[test_idx], device)

    print(f"  Train samples: {x_train.size(0)}   Test samples: {x_test.size(0)}")
    print()

    # ── train and evaluate each model ─────────────────────────────────────────
    all_loss_histories: dict[str, list[float]] = {}
    all_eval_results: dict[str, dict] = {}
    all_metrics: dict[str, dict] = {}

    for model_name in args.models:
        print(f"[{model_name}] building model …")
        model = build_model(
            model_name,
            input_dim=input_dim,
            hidden_size=args.hidden_size,
            latent_dim=args.latent_dim,
            output_dim=output_dim,
            seq_len=args.seq_len,
            device=device,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  parameters : {n_params:,}")

        print(f"  training   : {args.epochs} epochs …")
        t0 = time.time()
        loss_history = train_model(
            model, x_train, x1_train, y_train,
            n_epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )
        train_time = time.time() - t0

        print(f"  evaluating …")
        eval_result = evaluate_model(model, x_test, x1_test, y_test, args.batch_size)

        all_loss_histories[model_name] = loss_history
        all_eval_results[model_name] = eval_result

        mean_mse = eval_result["mean_mse"]
        mean_mae = eval_result["mean_mae"]
        final_loss = loss_history[-1]
        print(f"  final loss : {final_loss:.6f}")
        print(f"  test MSE   : {mean_mse:.6f}")
        print(f"  test MAE   : {mean_mae:.6f}")
        print(f"  time       : {train_time:.1f}s")
        print()

        all_metrics[model_name] = {
            "n_parameters": n_params,
            "final_train_loss": float(final_loss),
            "mean_test_mse": mean_mse,
            "mean_test_mae": mean_mae,
            "test_loss": eval_result["test_loss"],
            "train_time_seconds": round(train_time, 2),
        }

        print(f"  saving individual plots …")
        plot_individual(model_name, loss_history, eval_result, output_dir)

    # ── comparison plots ───────────────────────────────────────────────────────
    if len(args.models) > 1:
        print("Saving comparison plots …")
        plot_comparison(all_loss_histories, all_eval_results, output_dir)

    # ── save metrics JSON ──────────────────────────────────────────────────────
    metrics_path = output_dir / "metrics.json"
    summary = {
        "config": {
            "env": args.env,
            "episodes": args.episodes,
            "seq_len": args.seq_len,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_size": args.hidden_size,
            "latent_dim": args.latent_dim,
            "test_split": args.test_split,
            "seed": args.seed,
            "input_dim": int(input_dim),
            "output_dim": int(output_dim),
            "n_train": int(x_train.size(0)),
            "n_test": int(x_test.size(0)),
        },
        "models": all_metrics,
    }
    metrics_path.write_text(json.dumps(summary, indent=2))

    print(f"\nAll outputs saved to: {output_dir.resolve()}")
    print(f"Metrics summary    : {metrics_path.resolve()}")
    print("\nFiles written:")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
