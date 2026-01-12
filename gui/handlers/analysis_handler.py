"""Shared utilities for analysis operations."""

import os
import sys
import sqlite3
from typing import Dict, List, Optional, Tuple
from utils.logger import TA_logger

logger = TA_logger()


def detect_master_db(ta_output_folder: str) -> Optional[str]:
    """
    Auto-detect the master TA database path.
    
    Args:
        ta_output_folder: Path to TA output folder
    
    Returns:
        Path to master DB or None if not found
    """
    possible_db_paths = [
        os.path.join(ta_output_folder, "TA.db"),
        os.path.join(ta_output_folder, "TA", "TA.db"),
        os.path.join(ta_output_folder, "Master.db"),
        os.path.join(ta_output_folder, "TA", "Master.db"),
    ]

    for path in possible_db_paths:
        if os.path.exists(path):
            return path

    logger.warning(f"TA.db/Master.db not found in {ta_output_folder}")
    return None


def detect_frame_db_base(ta_output_folder: str, master_db_path: str) -> str:
    """
    Determine base path containing per-frame TA.db and images.
    
    Args:
        ta_output_folder: Path to TA output folder
        master_db_path: Path to master database
    
    Returns:
        Base path for frame databases
    """
    master_dir = os.path.dirname(master_db_path)
    if master_dir.endswith("/TA") or master_dir.endswith("\\TA"):
        frame_db_base_path = master_dir
    else:
        ta_subfolder = os.path.join(ta_output_folder, "TA")
        if os.path.exists(ta_subfolder):
            frame_db_base_path = ta_subfolder
        else:
            frame_db_base_path = ta_output_folder
    return frame_db_base_path


def list_frames_from_db(master_db_path: str, frame_db_base: Optional[str] = None) -> List[int]:
    """
    Return sorted list of frame numbers present in the cells table, or discover from directory structure.
    
    Args:
        master_db_path: Path to master database
        frame_db_base: Optional base path for frame directories (for fallback discovery)
    
    Returns:
        List of frame numbers
    """
    if not os.path.exists(master_db_path):
        # Try to discover frames from directory structure
        if frame_db_base and os.path.exists(frame_db_base):
            return _discover_frames_from_directories(frame_db_base)
        return []
    
    conn = sqlite3.connect(master_db_path)
    cursor = conn.cursor()
    try:
        # #region agent log
        try:
            import json, time
            # List all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"analysis_handler.py:list_frames_from_db:tables","message":"Master DB tables","data":{"master_db_path":master_db_path,"tables":tables},"timestamp":int(time.time()*1000)}) + '\n')
        except: pass
        # #endregion
        
        # Try cells table first
        try:
            cursor.execute("SELECT DISTINCT frame_nb FROM cells ORDER BY frame_nb")
            frames = [row[0] for row in cursor.fetchall()]
            if frames:
                # #region agent log
                try:
                    import json, time
                    with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"analysis_handler.py:list_frames_from_db:success","message":"Frames query successful","data":{"frames_count":len(frames),"frames":frames[:10]},"timestamp":int(time.time()*1000)}) + '\n')
                except: pass
                # #endregion
                return frames
        except sqlite3.Error:
            pass
        
        # Try cells_2D table as fallback
        try:
            cursor.execute("SELECT DISTINCT frame_nb FROM cells_2D ORDER BY frame_nb")
            frames = [row[0] for row in cursor.fetchall()]
            if frames:
                return frames
        except sqlite3.Error:
            pass
        
        frames = []
    except sqlite3.Error as e:
        # #region agent log
        try:
            import json, time
            with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"analysis_handler.py:list_frames_from_db:error","message":"Error querying frames","data":{"error":str(e)},"timestamp":int(time.time()*1000)}) + '\n')
        except: pass
        # #endregion
        logger.warning(f"Error reading frames from database: {e}")
        frames = []
    finally:
        conn.close()
    
    # If no frames found in database, try to discover from directory structure
    if not frames and frame_db_base and os.path.exists(frame_db_base):
        frames = _discover_frames_from_directories(frame_db_base)
    
    return frames


def _discover_frames_from_directories(frame_db_base: str) -> List[int]:
    """
    Discover frame numbers from Image#### directory names.
    
    Args:
        frame_db_base: Base path containing Image#### directories
    
    Returns:
        List of frame numbers
    """
    frames = []
    try:
        for item in os.listdir(frame_db_base):
            if item.startswith("Image") and os.path.isdir(os.path.join(frame_db_base, item)):
                try:
                    # Extract frame number from "Image0001" -> 1
                    frame_nb = int(item[5:])
                    frames.append(frame_nb)
                except ValueError:
                    continue
        frames.sort()
    except Exception as e:
        logger.warning(f"Error discovering frames from directories: {e}")
    return frames


def get_centroids_over_time(
    master_db_path: str,
    frame_db_base: str,
    track_id: int,
    frame_range: Optional[Tuple[int, int]] = None
) -> Dict[int, Tuple[float, float]]:
    """
    For a given track_id_cells, return a mapping frame_nb -> (center_x_cells, center_y_cells).
    
    Args:
        master_db_path: Path to master database
        frame_db_base: Base path for frame databases
        track_id: Track ID to get centroids for
        frame_range: Optional tuple of (start_frame, end_frame) to filter frames
    
    Returns:
        Dictionary mapping frame numbers to (x, y) centroid coordinates
    """
    # #region agent log
    try:
        import json, time
        with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"analysis_handler.py:get_centroids_over_time:entry","message":"get_centroids_over_time called","data":{"master_db_path":master_db_path,"frame_db_base":frame_db_base,"track_id":track_id,"frame_range":frame_range,"master_db_exists":os.path.exists(master_db_path) if master_db_path else False},"timestamp":int(time.time()*1000)}) + '\n')
    except: pass
    # #endregion
    if not os.path.exists(master_db_path):
        # #region agent log
        try:
            import json, time
            with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"analysis_handler.py:get_centroids_over_time:no_db","message":"Master DB does not exist","data":{"master_db_path":master_db_path},"timestamp":int(time.time()*1000)}) + '\n')
        except: pass
        # #endregion
        return {}
    
    conn = sqlite3.connect(master_db_path)
    cursor = conn.cursor()

    # Ensure tracked_cells table exists; if not, try frame-specific DBs (fallback)
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='tracked_cells'"
    )
    has_tracked = cursor.fetchone() is not None
    # #region agent log
    try:
        import json, time
        with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"analysis_handler.py:get_centroids_over_time:has_tracked","message":"Checked for tracked_cells table","data":{"has_tracked":has_tracked,"track_id":track_id},"timestamp":int(time.time()*1000)}) + '\n')
    except: pass
    # #endregion

    centroids: Dict[int, Tuple[float, float]] = {}

    if has_tracked:
        # Join tracked_cells with cells on local_id_cells and frame_nb
        query = """
            SELECT c.frame_nb, c.center_x_cells, c.center_y_cells
            FROM tracked_cells tc
            JOIN cells c
              ON tc.local_id_cells = c.local_id_cells
             AND tc.frame_nb = c.frame_nb
            WHERE tc.track_id_cells = ?
            ORDER BY c.frame_nb
        """
        try:
            cursor.execute(query, (int(track_id),))
            rows = cursor.fetchall()
            # #region agent log
            try:
                import json, time
                with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"analysis_handler.py:get_centroids_over_time:query_result","message":"Query executed","data":{"track_id":track_id,"rows_count":len(rows),"frame_range":frame_range},"timestamp":int(time.time()*1000)}) + '\n')
            except: pass
            # #endregion
            for frame_nb, cx, cy in rows:
                if cx is None or cy is None:
                    continue
                frame_nb_int = int(frame_nb)
                # Filter by frame range if specified
                if frame_range is not None:
                    start_frame, end_frame = frame_range
                    if frame_nb_int < start_frame or frame_nb_int > end_frame:
                        # #region agent log
                        try:
                            import json, time
                            with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"analysis_handler.py:get_centroids_over_time:frame_filtered","message":"Frame filtered out by range","data":{"frame_nb":frame_nb_int,"start_frame":start_frame,"end_frame":end_frame},"timestamp":int(time.time()*1000)}) + '\n')
                        except: pass
                        # #endregion
                        continue
                centroids[frame_nb_int] = (float(cx), float(cy))
            # #region agent log
            try:
                import json, time
                with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"analysis_handler.py:get_centroids_over_time:centroids_built","message":"Centroids dictionary built","data":{"track_id":track_id,"centroids_count":len(centroids),"centroids_frames":list(centroids.keys())[:10]},"timestamp":int(time.time()*1000)}) + '\n')
            except: pass
            # #endregion
        except sqlite3.Error as e:
            # #region agent log
            try:
                import json, time
                with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"analysis_handler.py:get_centroids_over_time:sql_error","message":"SQL error querying centroids","data":{"track_id":track_id,"error":str(e)},"timestamp":int(time.time()*1000)}) + '\n')
            except: pass
            # #endregion
            logger.warning(f"Error querying centroids: {e}")
    else:
        # Fallback: per-frame TA.db (slower, but robust)
        logger.warning("No tracked_cells table in master DB, falling back to frame TA.db.")
        # #region agent log
        try:
            import json, time
            with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"analysis_handler.py:get_centroids_over_time:fallback","message":"Using fallback method","data":{"track_id":track_id},"timestamp":int(time.time()*1000)}) + '\n')
        except: pass
        # #endregion
        frames = list_frames_from_db(master_db_path, frame_db_base)
        # #region agent log
        try:
            import json, time
            with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"analysis_handler.py:get_centroids_over_time:fallback_frames","message":"Got frames list","data":{"track_id":track_id,"frames_count":len(frames),"frames":frames[:10]},"timestamp":int(time.time()*1000)}) + '\n')
        except: pass
        # #endregion
        for frame_nb in frames:
            # Filter by frame range if specified
            if frame_range is not None:
                start_frame, end_frame = frame_range
                if frame_nb < start_frame or frame_nb > end_frame:
                    continue
            frame_name = f"Image{frame_nb:04d}"
            # Try both TA.db and pyTA.db
            frame_db_path = os.path.join(frame_db_base, frame_name, "TA.db")
            if not os.path.exists(frame_db_path):
                frame_db_path = os.path.join(frame_db_base, frame_name, "pyTA.db")
            # #region agent log
            try:
                import json, time
                with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"analysis_handler.py:get_centroids_over_time:fallback_frame_check","message":"Checking frame DB","data":{"track_id":track_id,"frame_nb":frame_nb,"frame_db_path":frame_db_path,"exists":os.path.exists(frame_db_path)},"timestamp":int(time.time()*1000)}) + '\n')
            except: pass
            # #endregion
            if not os.path.exists(frame_db_path):
                continue
            try:
                fconn = sqlite3.connect(frame_db_path)
                fcur = fconn.cursor()
                # #region agent log
                try:
                    import json, time
                    # Check what tables exist
                    fcur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in fcur.fetchall()]
                    with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"analysis_handler.py:get_centroids_over_time:fallback_tables","message":"Frame DB tables","data":{"track_id":track_id,"frame_nb":frame_nb,"tables":tables},"timestamp":int(time.time()*1000)}) + '\n')
                except: pass
                # #endregion
                
                # First try: query cell_tracks table directly for centroids (new approach)
                row = None
                try:
                    # Check if cell_tracks has center_x_cells and center_y_cells columns
                    fcur.execute("PRAGMA table_info(cell_tracks)")
                    columns = [col[1] for col in fcur.fetchall()]
                    has_centroids = 'center_x_cells' in columns and 'center_y_cells' in columns
                    
                    if has_centroids:
                        fcur.execute(
                            """
                            SELECT center_x_cells, center_y_cells
                            FROM cell_tracks
                            WHERE track_id = ?
                            """,
                            (int(track_id),),
                        )
                        row = fcur.fetchone()
                except sqlite3.Error:
                    pass
                
                # Fallback: try old method with tracked_cells and Cells tables
                if row is None:
                    try:
                        fcur.execute(
                            """
                            SELECT c.center_x_cells, c.center_y_cells
                            FROM tracked_cells tc
                            JOIN Cells c
                              ON tc.local_id_cells = c.local_id_cells
                            WHERE tc.track_id_cells = ?
                            """,
                            (int(track_id),),
                        )
                        row = fcur.fetchone()
                    except sqlite3.Error:
                        pass
                
                # #region agent log
                try:
                    import json, time
                    with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"analysis_handler.py:get_centroids_over_time:fallback_query_result","message":"Fallback query result","data":{"track_id":track_id,"frame_nb":frame_nb,"row":row if row else None},"timestamp":int(time.time()*1000)}) + '\n')
                except: pass
                # #endregion
                fconn.close()
                if row and row[0] is not None and row[1] is not None:
                    centroids[int(frame_nb)] = (float(row[0]), float(row[1]))
            except sqlite3.Error as e:
                # #region agent log
                try:
                    import json, time
                    with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"analysis_handler.py:get_centroids_over_time:fallback_error","message":"Fallback query error","data":{"track_id":track_id,"frame_nb":frame_nb,"error":str(e)},"timestamp":int(time.time()*1000)}) + '\n')
                except: pass
                # #endregion
                continue

    conn.close()
    # #region agent log
    try:
        import json, time
        with open(r'c:\Users\andre\OneDrive\Documents\Lemkes\006_Side\pyTissueAnalyzer\pyTissueAnalyzer\.cursor\debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"analysis_handler.py:get_centroids_over_time:return","message":"Returning centroids","data":{"track_id":track_id,"centroids_count":len(centroids)},"timestamp":int(time.time()*1000)}) + '\n')
    except: pass
    # #endregion
    return centroids


def validate_frame_range(frame_range: Optional[Tuple[int, int]], available_frames: List[int]) -> Tuple[bool, str]:
    """
    Validate a frame range against available frames.
    
    Args:
        frame_range: Tuple of (start_frame, end_frame) or None
        available_frames: List of available frame numbers
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if frame_range is None:
        return True, ""
    
    start_frame, end_frame = frame_range
    
    if start_frame < 0:
        return False, "Start frame must be >= 0"
    
    if end_frame < start_frame:
        return False, "End frame must be >= start frame"
    
    if available_frames:
        min_frame = min(available_frames)
        max_frame = max(available_frames)
        if start_frame < min_frame or end_frame > max_frame:
            return False, f"Frame range must be within [{min_frame}, {max_frame}]"
    
    return True, ""
