# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

from importlib import import_module


__all__ = ["configs", "distributed", "modules"]


def __getattr__(module_name):
    """Lazily import wan_va submodules to avoid config side effects at import time.

    Preprocessing scripts import lightweight utilities such as
    ``wan_va.modules.utils``. Eagerly importing ``wan_va.configs`` here also
    imports training configs, some of which require user-specific fields such as
    ``cache_path``. Lazy imports keep those training configs opt-in while
    preserving ``from wan_va import configs`` style access.
    """
    if module_name in __all__:
        module = import_module(f"{__name__}.{module_name}")
        globals()[module_name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {module_name!r}")
