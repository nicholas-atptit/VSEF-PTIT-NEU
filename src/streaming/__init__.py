from importlib import import_module
import sys

_impl = import_module("src.api.streaming")
_session_manager = import_module("src.api.streaming.session_manager")
sys.modules[__name__] = _impl
sys.modules[f"{__name__}.session_manager"] = _session_manager
