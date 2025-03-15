"""Utility functions module."""

from .helpers import slugify, setup_logging
from .compat import merge_dicts

if not hasattr(type, "__or__"):
    # For Python 3.9, create a compatibility layer
    from .compat import UnionOr
    
    # Export globally for use in type hints
    globals()["UnionOr"] = UnionOr

__all__ = ["slugify", "setup_logging", "merge_dicts"]
