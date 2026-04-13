from importlib import import_module
import sys

_impl = import_module("src.data.context.ingestion_pipeline")
sys.modules[__name__] = _impl

