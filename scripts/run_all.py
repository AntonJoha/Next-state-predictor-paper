"""Run the full next-state predictor evaluation pipeline.

This script is a convenience launcher that sets sensible defaults and calls
``evaluate_models.py`` for all three models.  Override any option via
command-line arguments — they are forwarded verbatim to the evaluation script.

Usage
-----
    python scripts/run_all.py [options]

Examples
--------
    # Quick smoke test (few episodes + epochs)
    python scripts/run_all.py --episodes 10 --epochs 5

    # Full run with custom output directory
    python scripts/run_all.py --episodes 100 --epochs 100 --output-dir my_results

    # Evaluate only two models
    python scripts/run_all.py --models tDLGM VRNN
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Default hyperparameters — can be overridden on the command line.
_DEFAULTS: dict[str, str] = {
    "--env": "CartPole-v1",
    "--episodes": "50",
    "--seq-len": "4",
    "--epochs": "50",
    "--batch-size": "64",
    "--lr": "1e-3",
    "--hidden-size": "64",
    "--latent-dim": "16",
    "--test-split": "0.2",
    "--output-dir": "results",
    "--seed": "42",
}

EVALUATE_SCRIPT = Path(__file__).parent / "evaluate_models.py"


def build_command(extra_args: list[str]) -> list[str]:
    """Merge defaults with any user-supplied arguments.

    User arguments take precedence: if an argument flag already appears in
    *extra_args*, the corresponding default is skipped.

    Args:
        extra_args: Raw ``sys.argv[1:]`` arguments passed to this script.

    Returns:
        A complete argument list for ``evaluate_models.py``.
    """
    # Collect flags that the user has explicitly provided.
    user_flags: set[str] = set()
    for token in extra_args:
        if token.startswith("--"):
            user_flags.add(token.split("=")[0])

    cmd = [sys.executable, str(EVALUATE_SCRIPT)]
    for flag, value in _DEFAULTS.items():
        if flag not in user_flags:
            cmd.extend([flag, value])
    cmd.extend(extra_args)
    return cmd


def main() -> None:
    extra_args = sys.argv[1:]
    cmd = build_command(extra_args)

    print("=" * 60)
    print("Next-state predictor — evaluation pipeline")
    print("=" * 60)
    print("Command:")
    print("  " + " ".join(cmd))
    print("=" * 60)
    print()

    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
