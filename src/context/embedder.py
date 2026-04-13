from importlib import import_module
import sys

_impl = import_module("src.data.context.embedder")
sys.modules[__name__] = _impl

