"""Model definitions."""
from .series_model import Net as SeriesNet
from .whole_model import Net as WholeNet
from .factory import create_model

__all__ = ["SeriesNet", "WholeNet", "create_model"]
