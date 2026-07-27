"""Hyperparameter tuning utilities for DQN agents."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Callable

import gymnasium as gym

from next_state_predictor.agent import DQNAgent
from next_state_predictor.train import evaluate, train_dqn


@dataclass
class TrialResult:
    """Result of a single hyperparameter trial.

    Attributes:
        params: The hyperparameter values used in this trial.
        score: Mean evaluation reward achieved with these hyperparameters.
    """

    params: dict[str, Any]
    score: float


@dataclass
class TuningResult:
    """Aggregated result of a hyperparameter tuning run.

    Attributes:
        best_params: Hyperparameter values that achieved the highest score.
        best_score: The mean evaluation reward of the best trial.
        trials: All individual trial results, ordered by the sequence they ran.
    """

    best_params: dict[str, Any]
    best_score: float
    trials: list[TrialResult] = field(default_factory=list)


def tune_dqn(
    make_env: Callable[[], gym.Env],
    param_grid: dict[str, list[Any]],
    n_train_episodes: int = 200,
    n_eval_episodes: int = 20,
    n_trials: int | None = None,
    seed: int | None = None,
) -> TuningResult:
    """Search for the best DQN hyperparameters over a given parameter grid.

    Two search strategies are available:

    * **Grid search** (``n_trials=None``): every combination of values in
      *param_grid* is evaluated exhaustively.
    * **Random search** (``n_trials=<int>``): *n_trials* combinations are
      sampled uniformly at random from *param_grid*.

    Args:
        make_env: A callable that returns a fresh Gymnasium environment.
            It is called once per trial so that each trial starts with an
            independent environment.
        param_grid: Mapping from :class:`~next_state_predictor.agent.DQNAgent`
            constructor keyword names to a list of candidate values.
            Example::

                {
                    "lr": [1e-3, 5e-4],
                    "hidden_dim": [64, 128],
                    "gamma": [0.99],
                }

        n_train_episodes: Number of training episodes per trial.
        n_eval_episodes: Number of evaluation episodes used to score each
            trial.  The mean reward across these episodes is the score.
        n_trials: Number of random trials.  ``None`` triggers an exhaustive
            grid search over all combinations.
        seed: Optional integer seed for reproducibility of random search.

    Returns:
        A :class:`TuningResult` containing the best hyperparameters and their
        score, together with every individual :class:`TrialResult`.

    Raises:
        ValueError: If *param_grid* is empty or *n_trials* is not positive.
    """
    if not param_grid:
        msg = "param_grid must not be empty."
        raise ValueError(msg)
    if n_trials is not None and n_trials < 1:
        msg = "n_trials must be a positive integer."
        raise ValueError(msg)

    keys = list(param_grid.keys())
    all_combinations: list[dict[str, Any]] = [
        dict(zip(keys, combo)) for combo in itertools.product(*param_grid.values())
    ]

    if n_trials is None:
        combinations_to_try = all_combinations
    else:
        rng = random.Random(seed)
        combinations_to_try = [
            rng.choice(all_combinations) for _ in range(n_trials)
        ]

    trials: list[TrialResult] = []
    best_score = float("-inf")
    best_params: dict[str, Any] = {}

    for params in combinations_to_try:
        env = make_env()
        try:
            agent = DQNAgent(env, seed=seed, **params)
            train_dqn(agent, n_episodes=n_train_episodes)
            stats = evaluate(agent, n_episodes=n_eval_episodes)
        finally:
            env.close()

        score = stats["mean"]
        trial = TrialResult(params=dict(params), score=score)
        trials.append(trial)

        if score > best_score:
            best_score = score
            best_params = dict(params)

    return TuningResult(best_params=best_params, best_score=best_score, trials=trials)
