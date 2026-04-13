from importlib import import_module
import sys

_impl = import_module("src.data.adapters")
_vnstock_adapter = import_module("src.data.adapters.vnstock_adapter")
sys.modules[__name__] = _impl
sys.modules[f"{__name__}.vnstock_adapter"] = _vnstock_adapter
