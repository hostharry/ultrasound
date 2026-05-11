"""Backward-compatible shim for the shared ``Utils/train_common.py`` module.

The real implementation lives in ``Utils/train_common.py``; this module
re-exports it for legacy callers that still ``from train_common import ...``
with ``Admm_net`` on ``sys.path``.  The underlying module is cached in
``sys.modules`` so identity checks succeed across both import paths.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parent.parent / "Utils"
_IMPL_PATH = _UTILS_DIR / "train_common.py"
_IMPL_KEY = "_ultrasound_utils_train_common_impl"

_utils_dir_str = str(_UTILS_DIR)
if _utils_dir_str not in sys.path:
    sys.path.insert(0, _utils_dir_str)

if _IMPL_KEY in sys.modules:
    _impl = sys.modules[_IMPL_KEY]
else:
    _spec = importlib.util.spec_from_file_location(_IMPL_KEY, _IMPL_PATH)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Cannot load shared train_common: {_IMPL_PATH}")
    _impl = importlib.util.module_from_spec(_spec)
    sys.modules[_IMPL_KEY] = _impl
    _spec.loader.exec_module(_impl)

sys.modules[__name__] = _impl
sys.modules.setdefault("train_common", _impl)
