"""Job-fit assistant: grounded job-posting vs. profile matching with Claude."""
from .analyze import analyze
from .profile import load_profile, parse_profile

__all__ = ["analyze", "load_profile", "parse_profile"]
__version__ = "0.1.0"
