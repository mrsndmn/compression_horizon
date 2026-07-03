"""Root conftest.

Its mere presence makes pytest add the repository root to ``sys.path`` (prepend
import mode inserts the rootdir that holds this file), so test modules can do
``from tests.trainer_helpers import ...`` under a bare ``pytest`` invocation — not
only under ``python -m pytest`` (which implicitly puts the CWD on the path).
Without it, ``PYTHONPATH=./src pytest`` fails at collection with
``ModuleNotFoundError: No module named 'tests'``.
"""
