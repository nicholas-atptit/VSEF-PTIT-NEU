from importlib import import_module
import sys

_impl = import_module("src.ml.llm.pipeline")
sys.modules[__name__] = _impl
