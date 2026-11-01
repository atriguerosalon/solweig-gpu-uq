"""The unit tests run on the CPU, also on machines with a GPU.

They build small tiles as CPU tensors, while SOLWEIG-GPU puts the tensors it creates on
cuda whenever a GPU is visible (``device = cuda if available``, inside its functions), and
the two cannot be mixed. pytest loads this file before the test modules, so the GPU is
hidden before CUDA starts. The GPU path is checked on real tiles by scripts/verify_*.py.
"""

import os
import sys

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def pytest_collection_finish(session):
    # CUDA reads the variable once, when it starts; fail clearly if something started it first
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        raise pytest.UsageError("CUDA started before tests/conftest.py could hide the GPU "
                                "(a plugin importing torch?); the tests must run on the CPU")
