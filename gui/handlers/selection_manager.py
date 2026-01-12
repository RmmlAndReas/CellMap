"""Generic cell selection manager for analyses."""

from typing import List, Tuple, Optional, Dict, Any, Callable
from utils.logger import TA_logger

logger = TA_logger()


class SelectionManager:
    """Manages cell selection state for analyses."""
    
    def __init__(self):
        self._current_analysis: Optional[str] = None
        self._selection_mode: str = "single"
        self._selection_count: Optional[int] = None
        self._selected_cells: List[int] = []  # Current selection buffer
        self._completed_selections: List[Any] = []  # Completed selections (pairs, groups, etc.)
        self._callbacks: Dict[str, List[Callable]] = {
            'selection_changed': [],
            'selection_completed': [],
        }
    
    def set_analysis_context(
        self,
        analysis_name: str,
        selection_mode: str,
        selection_count: Optional[int] = None
    ):
        """
        Set the current analysis context for selection.
        
        Args:
            analysis_name: Name of the active analysis
            selection_mode: Selection mode ('single', 'pair', 'group', 'custom')
            selection_count: Required count per selection (None for unlimited)
        """
        self._current_analysis = analysis_name
        self._selection_mode = selection_mode
        self._selection_count = selection_count
        self._selected_cells = []
        self._completed_selections = []
        logger.info(f"Selection context set: {analysis_name}, mode={selection_mode}, count={selection_count}")
    
    def clear_analysis_context(self):
        """Clear the current analysis context."""
        self._current_analysis = None
        self._selection_mode = "single"
        self._selection_count = None
        self._selected_cells = []
        self._completed_selections = []
    
    def add_cell(self, track_id: int) -> bool:
        """
        Add a cell to the current selection.
        
        Args:
            track_id: Track ID of the cell to add
        
        Returns:
            True if selection was completed (for pair/group modes), False otherwise
        """
        if track_id in self._selected_cells:
            logger.debug(f"Cell {track_id} already selected, ignoring")
            return False
        
        self._selected_cells.append(track_id)
        self._notify_callbacks('selection_changed')
        
        # Check if selection is complete
        if self._selection_count is not None and len(self._selected_cells) >= self._selection_count:
            return self._complete_selection()
        
        return False
    
    def _complete_selection(self) -> bool:
        """Complete the current selection based on mode."""
        if self._selection_mode == "pair" and len(self._selected_cells) == 2:
            # Form a pair
            pair = (self._selected_cells[0], self._selected_cells[1])
            self._completed_selections.append(pair)
            logger.info(f"Pair completed: {pair}")
            self._selected_cells = []
            self._notify_callbacks('selection_completed')
            return True
        elif self._selection_mode == "single" and len(self._selected_cells) == 1:
            # Single cell selection
            cell = self._selected_cells[0]
            self._completed_selections.append(cell)
            logger.info(f"Cell selected: {cell}")
            self._selected_cells = []
            self._notify_callbacks('selection_completed')
            return True
        elif self._selection_mode == "group":
            # Group selection - user controls when to complete
            return False
        
        return False
    
    def remove_cell(self, track_id: int):
        """Remove a cell from the current selection."""
        if track_id in self._selected_cells:
            self._selected_cells.remove(track_id)
            self._notify_callbacks('selection_changed')
    
    def clear_current_selection(self):
        """Clear the current selection buffer (but keep completed selections)."""
        self._selected_cells = []
        self._notify_callbacks('selection_changed')
    
    def clear_all(self):
        """Clear all selections (current and completed)."""
        self._selected_cells = []
        self._completed_selections = []
        self._notify_callbacks('selection_changed')
    
    def get_selection(self) -> Any:
        """
        Get the current selection in the format expected by the analysis.
        
        Returns:
            Selection in appropriate format (list, pairs, etc.)
        """
        if self._selection_mode == "pair":
            return self._completed_selections.copy()  # List of pairs
        elif self._selection_mode == "single":
            return self._completed_selections.copy()  # List of single cells
        elif self._selection_mode == "group":
            # For group mode, return completed selections if any, otherwise return buffer as a group
            if self._completed_selections:
                return self._completed_selections.copy()
            elif self._selected_cells:
                # Return buffer as a single group (list containing list of track_ids)
                return [self._selected_cells.copy()]
            else:
                return []
        else:
            # Custom mode - return as-is
            return self._completed_selections.copy()
    
    def get_current_buffer(self) -> List[int]:
        """Get the current selection buffer (incomplete selection)."""
        return self._selected_cells.copy()
    
    def get_status_message(self) -> str:
        """Get a status message describing the current selection state."""
        if not self._current_analysis:
            return "No analysis active"
        
        completed_count = len(self._completed_selections)
        buffer_count = len(self._selected_cells)
        
        if self._selection_mode == "pair":
            if buffer_count == 0:
                return f"Select pair {completed_count + 1}: Click first cell"
            elif buffer_count == 1:
                return f"Select pair {completed_count + 1}: Click second cell ({completed_count} pairs completed)"
            else:
                return f"Pair {completed_count + 1} completed ({completed_count + 1} pairs total)"
        elif self._selection_mode == "single":
            return f"Selected {completed_count} cell(s). Click to select more."
        else:
            return f"Selected {completed_count} item(s), {buffer_count} in buffer"
    
    def register_callback(self, event: str, callback: Callable):
        """
        Register a callback for selection events.
        
        Args:
            event: Event name ('selection_changed', 'selection_completed')
            callback: Callback function
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _notify_callbacks(self, event: str):
        """Notify all callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in selection callback: {e}")
