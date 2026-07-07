"""ramentruck"""

__version__ = "0.3.0"


from .broth import Broth
from .noodles import (
    slurp,
    DatasetMenu,
    ChefRecommendation,
)
from .results import BrothResult

__all__ = [
    "Broth",
    "BrothResult",
    "slurp",
    "DatasetMenu",
    "ChefRecommendation",
]
