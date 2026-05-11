"""Backward-compatible shim for the shared ``Utils/loss.py`` module.

The implementation lives in ``Utils/loss.py``; this module re-exports it so
older ``from loss import CombinedLoss`` statements (executed with
``Admm_net`` on ``sys.path``) keep working.  The shim caches the underlying
module in ``sys.modules`` so identity checks (``isinstance``) succeed across
both import paths.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parent.parent / "Utils"
_IMPL_PATH = _UTILS_DIR / "loss.py"
_IMPL_KEY = "_ultrasound_utils_loss_impl"

_utils_dir_str = str(_UTILS_DIR)
if _utils_dir_str not in sys.path:
    sys.path.insert(0, _utils_dir_str)

if _IMPL_KEY in sys.modules:
    _impl = sys.modules[_IMPL_KEY]
else:
    _spec = importlib.util.spec_from_file_location(_IMPL_KEY, _IMPL_PATH)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Cannot load shared loss: {_IMPL_PATH}")
    _impl = importlib.util.module_from_spec(_spec)
    sys.modules[_IMPL_KEY] = _impl
    _spec.loader.exec_module(_impl)

# Make `import loss` resolve to the same module object regardless of which
# directory triggered the shim.  This keeps `isinstance` checks across both
# import paths consistent.
sys.modules[__name__] = _impl
sys.modules.setdefault("loss", _impl)
