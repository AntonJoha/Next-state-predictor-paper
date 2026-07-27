{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python312;

  pythonEnv = python.withPackages (ps: with ps; [
    # Core dependencies
    gymnasium
    numpy

    # Notebook support
    notebook
    ipykernel

    # Development tools
    pytest
    pytest-cov
  ]);
in

pkgs.mkShell {
  name = "next-state-predictor";

  packages = [
    pythonEnv
  ];

  shellHook = ''
    # Install the project in editable mode so imports resolve correctly.
    pip install --quiet --no-deps -e . 2>/dev/null || true
    echo "next-state-predictor dev shell ready."
    echo "Run:  python -m next_state_predictor.main --help"
    echo "Run:  jupyter notebook"
  '';
}
