"""
Convenience runner: executes the entire project pipeline in order.

    .venv/bin/python -m src.run_all

Equivalent to running, one after another:

    python -m src.data_preparation
    python -m src.train_models
    python -m src.evaluate_models
    python -m src.shap_explain
"""

from src import compat  # noqa: F401  (must be first)
from src.data_preparation import main as run_data
from src.evaluate_models import main as run_eval
from src.shap_explain import main as run_shap
from src.train_models import main as run_train


def main() -> None:
    run_data()
    print("\n")
    run_train()
    print("\n")
    run_eval()
    print("\n")
    run_shap()
    print("\nAll stages completed successfully.")


if __name__ == "__main__":
    main()
