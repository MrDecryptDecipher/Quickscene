"""
Utility modules for Quickscene system.

Modules:
- config_loader: YAML configuration management
- file_manager: File I/O operations and path management
- validators: Data validation and format checking
"""

__all__ = [
    "ConfigLoader",
    "FileManager",
    "Validators"
]

def __getattr__(name):
    """Lazy import to handle missing dependencies gracefully."""
    if name == "ConfigLoader":
        from .config_loader import ConfigLoader
        return ConfigLoader
    elif name == "FileManager":
        from .file_manager import FileManager
        return FileManager
    elif name == "Validators":
        from .validators import Validators
        return Validators
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
