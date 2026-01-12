"""Concrete analysis implementations."""

from utils.logger import TA_logger

logger = TA_logger()

# Import all analyses here for auto-discovery
__all__ = []

try:
    from gui.analysis.analyses.cell_pair_distance import CellPairDistanceAnalysis
    __all__.append('CellPairDistanceAnalysis')
except ImportError as e:
    # Analysis not yet implemented
    logger.error(f"Failed to import CellPairDistanceAnalysis: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    # Catch any other errors during import
    logger.error(f"Error importing CellPairDistanceAnalysis: {e}")
    import traceback
    traceback.print_exc()

try:
    from gui.analysis.analyses.tissue_trajectory import TissueTrajectoryAnalysis
    __all__.append('TissueTrajectoryAnalysis')
except ImportError as e:
    # Analysis not yet implemented
    logger.error(f"Failed to import TissueTrajectoryAnalysis: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    # Catch any other errors during import
    logger.error(f"Error importing TissueTrajectoryAnalysis: {e}")
    import traceback
    traceback.print_exc()
