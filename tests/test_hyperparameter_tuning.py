"""Tests for the hyperparameter_tuning module."""

from __future__ import annotations

import gymnasium as gym
import pytest

from next_state_predictor.hyperparameter_tuning import (
    TrialResult,
    TuningResult,
    tune_dqn,
)


def make_cartpole() -> gym.Env:
    return gym.make("CartPole-v1")


# ---------------------------------------------------------------------------
# Smoke / integration tests (small grids, few episodes)
# ---------------------------------------------------------------------------


def test_grid_search_returns_tuning_result():
    result = tune_dqn(
        make_cartpole,
        param_grid={"lr": [1e-3], "hidden_dim": [32]},
        n_train_episodes=2,
        n_eval_episodes=2,
    )
    assert isinstance(result, TuningResult)


def test_grid_search_exhausts_all_combinations():
    result = tune_dqn(
        make_cartpole,
        param_grid={"lr": [1e-3, 5e-4], "hidden_dim": [32, 64]},
        n_train_episodes=2,
        n_eval_episodes=2,
    )
    # 2 lr values × 2 hidden_dim values = 4 combinations
    assert len(result.trials) == 4


def test_random_search_respects_n_trials():
    result = tune_dqn(
        make_cartpole,
        param_grid={"lr": [1e-3, 5e-4, 1e-4], "hidden_dim": [32, 64, 128]},
        n_train_episodes=2,
        n_eval_episodes=2,
        n_trials=3,
        seed=0,
    )
    assert len(result.trials) == 3


def test_best_params_contained_in_trials():
    result = tune_dqn(
        make_cartpole,
        param_grid={"lr": [1e-3, 5e-4], "hidden_dim": [32]},
        n_train_episodes=2,
        n_eval_episodes=2,
    )
    trial_params = [t.params for t in result.trials]
    assert result.best_params in trial_params


def test_best_score_matches_best_trial():
    result = tune_dqn(
        make_cartpole,
        param_grid={"lr": [1e-3, 5e-4], "hidden_dim": [32]},
        n_train_episodes=2,
        n_eval_episodes=2,
    )
    max_trial_score = max(t.score for t in result.trials)
    assert result.best_score == max_trial_score


def test_trial_result_has_correct_types():
    result = tune_dqn(
        make_cartpole,
        param_grid={"lr": [1e-3], "hidden_dim": [32]},
        n_train_episodes=2,
        n_eval_episodes=2,
    )
    for trial in result.trials:
        assert isinstance(trial, TrialResult)
        assert isinstance(trial.params, dict)
        assert isinstance(trial.score, float)


def test_random_search_is_reproducible():
    kwargs = dict(
        make_env=make_cartpole,
        param_grid={"lr": [1e-3, 5e-4, 1e-4], "hidden_dim": [32, 64]},
        n_train_episodes=2,
        n_eval_episodes=2,
        n_trials=3,
        seed=42,
    )
    result1 = tune_dqn(**kwargs)
    result2 = tune_dqn(**kwargs)
    # Same seed → same sequence of param combinations sampled
    assert [t.params for t in result1.trials] == [t.params for t in result2.trials]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_empty_param_grid_raises():
    with pytest.raises(ValueError, match="param_grid must not be empty"):
        tune_dqn(make_cartpole, param_grid={})


def test_non_positive_n_trials_raises():
    with pytest.raises(ValueError, match="n_trials must be a positive integer"):
        tune_dqn(
            make_cartpole,
            param_grid={"lr": [1e-3]},
            n_trials=0,
        )
