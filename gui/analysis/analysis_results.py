"""Data container for analysis results."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
try:
    import matplotlib.figure
    Figure = matplotlib.figure.Figure
except ImportError:
    Figure = Any


@dataclass
class AnalysisResults:
    """Container for analysis results."""
    
    analysis_name: str
    figures: List[Figure] = field(default_factory=list)  # Plot figures
    images: Dict[str, np.ndarray] = field(default_factory=dict)  # Named image arrays
    data: Dict[str, Any] = field(default_factory=dict)  # CSV data, tables, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)  # Frame range, parameters, etc.
    
    def has_results(self) -> bool:
        """Check if results contain any data."""
        return (
            len(self.figures) > 0 or
            len(self.images) > 0 or
            len(self.data) > 0
        )
