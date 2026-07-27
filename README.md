# Next-state-predictor-paper

A template for reinforcement learning projects built on top of
[Gymnasium](https://gymnasium.farama.org/).

---

## Project layout

```
.
├── shell.nix                        # Nix development shell
├── pyproject.toml                   # Project metadata & dependencies
├── requirements.txt                 # Pip-compatible dependency list
├── src/
│   └── next_state_predictor/
│       ├── __init__.py
│       ├── agent.py                 # Agent base class + RandomAgent
│       ├── train.py                 # Training & evaluation loops
│       ├── utils.py                 # Seed helpers
│       └── main.py                  # CLI entry-point
└── tests/
    ├── test_agent.py
    └── test_train.py
```

---

## Quick start

### With Nix

```bash
nix-shell          # drops you into a shell with all dependencies available
python -m next_state_predictor.main --help
```

### Without Nix

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m next_state_predictor.main --help
```

### Run the example

```bash
# 20 episodes of CartPole with a random agent
python -m next_state_predictor.main --env CartPole-v1 --episodes 20 --seed 42
```

```bash
# DQN training with transition persistence and 4-state trajectories
python -m next_state_predictor.main --agent dqn --episodes 20 --db transitions.db --trajectory-length 4
```

---

## Running tests

```bash
pytest
```

---

## Extending the template

1. Subclass `Agent` in `src/next_state_predictor/agent.py` and implement
   `select_action`.
2. Pass your agent to `train()` / `evaluate()` in `train.py`.
3. Register any new Gymnasium environments you create in a separate
   `envs/` package and import them before calling `gym.make()`.