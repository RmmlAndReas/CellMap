"""Base analysis interface for all analyses."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any, List
from qtpy.QtWidgets import QWidget
from gui.analysis.analysis_results import AnalysisResults


class BaseAnalysis(ABC):
    """Abstract base class that all analyses must implement."""
    
    # Required properties (must be set by subclasses)
    name: str = ""  # Display name
    description: str = ""  # Tooltip/help text
    selection_mode: str = "single"  # 'single', 'pair', 'group', 'custom'
    selection_count: Optional[int] = None  # Required count (2 for pairs, None for unlimited)
    
    def get_ui_widgets(self) -> Dict[str, QWidget]:
        """
        Return optional UI widgets for analysis parameters.
        
        Returns:
            Dictionary mapping parameter names to QWidget instances.
            These widgets will be added to the analysis tab UI.
            Empty dict if no custom widgets needed.
        """
        return {}
    
    def validate_selection(self, selected_cells: Any) -> Tuple[bool, str]:
        """
        Validate that selected cells meet the analysis requirements.
        
        Args:
            selected_cells: The selected cells in the format expected by this analysis
                (list of track_ids, list of pairs, etc.)
        
        Returns:
            Tuple of (is_valid, error_message)
            is_valid: True if selection is valid, False otherwise
            error_message: Empty string if valid, error description if invalid
        """
        return True, ""
    
    @abstractmethod
    def run(
        self,
        selected_cells: Any,
        ta_output_folder: str,
        frame_range: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> AnalysisResults:
        """
        Execute the analysis and return results.
        
        Args:
            selected_cells: The selected cells in the format expected by this analysis
            ta_output_folder: Path to TA output folder
            frame_range: Optional tuple of (start_frame, end_frame) for analysis
            **kwargs: Additional parameters from UI widgets
        
        Returns:
            AnalysisResults object containing figures, images, data, and metadata
        """
        pass
