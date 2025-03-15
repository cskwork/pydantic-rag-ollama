"""Compatibility layer for different Python versions."""
import sys
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic

# Check if we're using Python 3.10+ for union type operator
PY_310_PLUS = sys.version_info >= (3, 10)

if not PY_310_PLUS:
    # For Python 3.9 compatibility
    T = TypeVar('T')
    
    class UnionOperator:
        """Class to simulate the | operator for union types in Python 3.10+."""
        
        def __class_getitem__(cls, types):
            if isinstance(types, tuple):
                return Union[types]
            return Union[types]
    
    # Create a global instance to be used like Type | None
    UnionOr = UnionOperator()
    
    # Helper function for dict merging (| operator in Python 3.9+)
    def merge_dicts(dict1: Dict[Any, Any], dict2: Dict[Any, Any]) -> Dict[Any, Any]:
        """Merge two dictionaries (equivalent to dict1 | dict2 in Python 3.10+).
        
        Args:
            dict1 (Dict[Any, Any]): First dictionary.
            dict2 (Dict[Any, Any]): Second dictionary.
            
        Returns:
            Dict[Any, Any]: Merged dictionary.
        """
        result = dict1.copy()
        result.update(dict2)
        return result
else:
    # For Python 3.10+, use the built-in | operator
    def merge_dicts(dict1: Dict[Any, Any], dict2: Dict[Any, Any]) -> Dict[Any, Any]:
        """Merge two dictionaries using the | operator.
        
        Args:
            dict1 (Dict[Any, Any]): First dictionary.
            dict2 (Dict[Any, Any]): Second dictionary.
            
        Returns:
            Dict[Any, Any]: Merged dictionary.
        """
        return dict1 | dict2
