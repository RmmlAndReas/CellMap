"""Tracking method implementations.

This package contains different tracking algorithms for matching cells between frames.
"""

from tracking.methods.pyramidal import match_cells_by_pyramidal_registration

__all__ = [
    'match_cells_by_pyramidal_registration',
]
