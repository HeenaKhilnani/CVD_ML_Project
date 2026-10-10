"""
Compatibility helpers.

Why this file exists
--------------------
The genuine Learning Vector Quantization model in this project comes from the
``sklvq`` library (version 0.1.2). That library was written for NumPy 1.x and
scikit-learn < 1.6. Our environment uses NumPy 2.x, which removed the old
``np.int`` alias that ``sklvq`` still uses internally.

Restoring the alias is a one-line, harmless fix (``np.int`` was just an alias
for the built-in ``int``) and lets the *unmodified* sklvq implementation run.

Import this module BEFORE importing sklvq, for example::

    from src import compat  # noqa: F401  (must come first)
    from sklvq import GLVQ
"""

import numpy as np

if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
