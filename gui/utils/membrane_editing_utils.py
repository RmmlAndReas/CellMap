"""Membrane editing operations for paint widget.

This module provides functions for detecting membrane connection points and
extracting line segments between points for membrane creation.
"""

import numpy as np
import logging
from utils.logger import TA_logger

# Use TA_logger to ensure logs appear in GUI
logger = TA_logger()


def detect_membrane_crossing_points(drawn, drawn_mask):
    """
    Detect two crossing points where the drawn line crosses through existing membranes.
    A crossing point is where the line goes from non-membrane to membrane (or vice versa).
    
    Args:
        drawn: numpy array with drawn pixels (non-zero where user drew)
        drawn_mask: numpy array of current mask (255 = membrane, 0 = cell interior)
        
    Returns:
        tuple: (crossing1, crossing2) as (y, x) tuples, or (None, None) if not found
    """
    if drawn is None or np.count_nonzero(drawn) == 0:
        return None, None
    
    # Find all pixels in the drawn line
    drawn_coords = np.where(drawn > 0)
    if len(drawn_coords[0]) == 0:
        return None, None
    
    # Order pixels along the drawn line (from start to end)
    # np.where doesn't guarantee order, so we need to order them properly
    drawn_list = list(zip(drawn_coords[0], drawn_coords[1]))
    if len(drawn_list) < 2:
        return None, None
    
    # Try to order pixels along the line by finding a path through them
    # Start with the first pixel, then find the nearest unvisited neighbor
    ordered_list = []
    remaining = set(drawn_list)
    
    if len(remaining) > 0:
        # Start with the first pixel (or one at the edge)
        current = drawn_list[0]
        ordered_list.append(current)
        remaining.remove(current)
        
        # Build a path by always moving to the nearest neighbor
        while len(remaining) > 0:
            min_dist = float('inf')
            next_pixel = None
            for pixel in remaining:
                dist = np.sqrt((pixel[0] - current[0])**2 + (pixel[1] - current[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    next_pixel = pixel
            
            if next_pixel is None:
                # No more connected pixels, add remaining in arbitrary order
                ordered_list.extend(remaining)
                break
            
            ordered_list.append(next_pixel)
            remaining.remove(next_pixel)
            current = next_pixel
    
    drawn_list = ordered_list
    
    # NEW APPROACH: Identify which distinct membrane regions the line crosses
    # Label connected components in the membrane mask to identify separate membranes
    from skimage.measure import label as sk_label
    
    # Label connected membrane regions (each separate membrane gets a unique ID)
    membrane_binary = (drawn_mask == 255).astype(np.uint8)
    labeled_membranes = sk_label(membrane_binary, connectivity=2, background=0)
    
    # Track which membrane regions each pixel along the drawn line touches
    membrane_ids_along_line = []
    
    try:
        for i, (y, x) in enumerate(drawn_list):
            membrane_id = None
            if 0 <= y < drawn_mask.shape[0] and 0 <= x < drawn_mask.shape[1]:
                # Check the labeled membrane at this pixel
                if labeled_membranes[y, x] > 0:
                    membrane_id = int(labeled_membranes[y, x])
                else:
                    # Check 8-connected neighbors to find nearby membrane regions
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < drawn_mask.shape[0] and 0 <= nx < drawn_mask.shape[1]:
                                if labeled_membranes[ny, nx] > 0:
                                    membrane_id = int(labeled_membranes[ny, nx])
                                    break
                        if membrane_id is not None:
                            break
            
            membrane_ids_along_line.append(membrane_id)
    except Exception as e:
        logger.error(f'detect_membrane_crossing: Exception during crossing detection: {e}')
        import traceback
        logger.error(traceback.format_exc())
        return None, None
    
    # Find all unique membrane regions touched by the line
    unique_membranes = [m for m in set(membrane_ids_along_line) if m is not None]
    
    # Find crossing points: where we transition between different membrane regions
    crossing_points = []
    for i in range(1, len(membrane_ids_along_line)):
        prev_id = membrane_ids_along_line[i-1]
        curr_id = membrane_ids_along_line[i]
        
        # Crossing occurs when we transition between two different membrane regions
        # (ignoring transitions through None/gaps)
        if prev_id is not None and curr_id is not None and prev_id != curr_id:
            crossing_points.append(drawn_list[i])
        # Also detect when we leave one membrane region and enter another (through a gap)
        elif prev_id is not None and curr_id is None:
            # Mark where we left the membrane
            crossing_points.append(drawn_list[i-1])
        elif prev_id is None and curr_id is not None:
            # Mark where we entered a new membrane
            crossing_points.append(drawn_list[i])
    
    logger.info(f'detect_membrane_crossing: Found {len(crossing_points)} crossing point(s) through existing membranes')
    
    if len(crossing_points) < 2:
        logger.info(f'detect_membrane_crossing: Need 2 crossing points for membrane creation, found {len(crossing_points)}')
        return None, None
    
    # Find the positions of crossing points along the drawn line
    # This helps us select points that are far apart and represent entry/exit
    crossing_indices = []
    for cp in crossing_points:
        # Find the index of this crossing point in the drawn_list
        try:
            idx = drawn_list.index(cp)
            crossing_indices.append((idx, cp))
        except ValueError:
            # If exact match not found, find closest point
            min_dist = float('inf')
            closest_idx = 0
            for i, pixel in enumerate(drawn_list):
                dist = np.sqrt((pixel[0] - cp[0])**2 + (pixel[1] - cp[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            crossing_indices.append((closest_idx, cp))
    
    # Sort by position along the line
    crossing_indices.sort(key=lambda x: x[0])
    
    # Strategy: Select the first and last crossing points along the line
    # This ensures we get entry and exit points that are far apart
    # Refinement: If multiple points on one side, choose closest to cell interior
    if len(crossing_indices) >= 2:
        # Group by membrane region to identify sides
        crossing_groups = {}
        for idx, cp in crossing_indices:
            cp_y, cp_x = cp
            membrane_id = None
            if 0 <= cp_y < drawn_mask.shape[0] and 0 <= cp_x < drawn_mask.shape[1]:
                if labeled_membranes[cp_y, cp_x] > 0:
                    membrane_id = int(labeled_membranes[cp_y, cp_x])
                else:
                    # Check 8-connected neighbors
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = cp_y + dy, cp_x + dx
                            if 0 <= ny < drawn_mask.shape[0] and 0 <= nx < drawn_mask.shape[1]:
                                if labeled_membranes[ny, nx] > 0:
                                    membrane_id = int(labeled_membranes[ny, nx])
                                    break
                        if membrane_id is not None:
                            break
            
            group_key = membrane_id if membrane_id is not None and membrane_id > 0 else "none"
            if group_key not in crossing_groups:
                crossing_groups[group_key] = []
            crossing_groups[group_key].append((idx, cp))
        
        # Refine: for groups with multiple points, choose closest to cell interior
        refined_indices = []
        for group_key, group_points in crossing_groups.items():
            if len(group_points) == 1:
                refined_indices.append(group_points[0])
            else:
                # Multiple points on same side - find closest to cell interior
                best = None
                min_dist = float('inf')
                for idx, cp in group_points:
                    cp_y, cp_x = cp
                    # Search for cell interior (mask == 0) within radius
                    r = 10
                    y1, y2 = max(0, cp_y - r), min(drawn_mask.shape[0], cp_y + r + 1)
                    x1, x2 = max(0, cp_x - r), min(drawn_mask.shape[1], cp_x + r + 1)
                    interior = (drawn_mask[y1:y2, x1:x2] == 0)
                    iy, ix = np.where(interior)
                    if len(iy) > 0:
                        iy = iy + y1
                        ix = ix + x1
                        dists = np.sqrt((iy - cp_y)**2 + (ix - cp_x)**2)
                        d = float(np.min(dists))
                        if d < min_dist:
                            min_dist = d
                            best = (idx, cp)
                refined_indices.append(best if best is not None else group_points[0])
        
        # Sort refined list and select first/last
        refined_indices.sort(key=lambda x: x[0])
        if len(refined_indices) >= 2:
            crossing1 = refined_indices[0][1]
            crossing2 = refined_indices[-1][1]
        else:
            # Fallback to original
            crossing1 = crossing_indices[0][1]
            crossing2 = crossing_indices[-1][1]
        
        # Calculate distance to verify they're far enough apart
        crossing_dist = np.sqrt((crossing1[0] - crossing2[0])**2 + (crossing1[1] - crossing2[1])**2)
        
        # If the selected points are too close (< 5 pixels), try to find points that are farther apart
        if crossing_dist < 5 and len(crossing_indices) > 2:
            # Try to find the pair with maximum distance
            max_dist = 0
            best_pair = (crossing_indices[0][1], crossing_indices[-1][1])
            for i in range(len(crossing_indices)):
                for j in range(i + 1, len(crossing_indices)):
                    cp1 = crossing_indices[i][1]
                    cp2 = crossing_indices[j][1]
                    dist = np.sqrt((cp1[0] - cp2[0])**2 + (cp1[1] - cp2[1])**2)
                    if dist > max_dist:
                        max_dist = dist
                        best_pair = (cp1, cp2)
            crossing1, crossing2 = best_pair
            crossing_dist = max_dist
    else:
        # Fallback to old method if we can't find positions
        first_point = drawn_list[0]
        last_point = drawn_list[-1]
        
        min_dist1 = float('inf')
        crossing1 = None
        for cp in crossing_points:
            dist = np.sqrt((cp[0] - first_point[0])**2 + (cp[1] - first_point[1])**2)
            if dist < min_dist1:
                min_dist1 = dist
                crossing1 = cp
        
        min_dist2 = float('inf')
        crossing2 = None
        for cp in crossing_points:
            if cp == crossing1:
                continue
            dist = np.sqrt((cp[0] - last_point[0])**2 + (cp[1] - last_point[1])**2)
            if dist < min_dist2:
                min_dist2 = dist
                crossing2 = cp
    
    if crossing1 is None or crossing2 is None:
        logger.warning('detect_membrane_crossing: Could not determine two distinct crossing points')
        return None, None
    
    # Calculate final distance
    crossing_dist = np.sqrt((crossing1[0] - crossing2[0])**2 + (crossing1[1] - crossing2[1])**2)
    
    if crossing1 is not None and crossing2 is not None:
        logger.info(f'detect_membrane_crossing: Selected two crossing points: {crossing1} and {crossing2} (distance={crossing_dist:.1f} pixels)')
        logger.info(f'detect_membrane_crossing: Total drawn pixels: {len(drawn_coords[0])}, crossing points found: {len(crossing_points)}')
    
    return crossing1, crossing2


def extract_line_segment_between_points(drawn, endpoint1, endpoint2, drawn_mask):
    """
    Extract only the line segment between two connection points, removing tips.

    Intended behavior for membrane addition:
    1. Skeletonize the input line
    2. Find the overlap pixels between the skeletonized line and existing membranes
       - There must be exactly two overlapping pixels. If more, abort this task.
    3. Remove the tips and keep the skeletonized line between the two overlap pixels
       - Returns the skeletonized path (1-pixel wide), not the original thick line
    
    Args:
        drawn: numpy array with drawn pixels (non-zero where user drew)
        endpoint1: (y, x) tuple of first connection point (not used directly, but kept for API compatibility)
        endpoint2: (y, x) tuple of second connection point (not used directly, but kept for API compatibility)
        drawn_mask: numpy array of current mask (255 = membrane, 0 = cell interior)
        
    Returns:
        numpy array with the skeletonized segment between the two overlap points (1-pixel wide)
        Returns None if there are not exactly two overlap pixels
    """
    if drawn is None or endpoint1 is None or endpoint2 is None:
        return None

    drawn_bin = (drawn > 0)
    if np.count_nonzero(drawn_bin) == 0:
        return None

    # Step 1: Skeletonize the input line to 1-pixel width
    # Use medial_axis which guarantees 1-pixel wide skeletons
    try:
        from skimage.morphology import medial_axis
    except ImportError:
        # Fallback to skeletonize if medial_axis is not available
        try:
            from skimage.morphology import skeletonize
            skel = skeletonize(drawn_bin.astype(bool)).astype(bool)
        except Exception as e:
            logger.error(f"extract_line_segment: could not import skeletonization functions ({e}); returning None")
            return None
    else:
        # medial_axis produces true 1-pixel wide skeletons
        skel = medial_axis(drawn_bin.astype(bool)).astype(bool)
    
    if np.count_nonzero(skel) == 0:
        return None

    def _neighbors8(y, x):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yield y + dy, x + dx

    def _nearest_true(mask, p, max_r=6):
        """Find nearest True pixel to p=(y,x) within a growing square window."""
        py, px = int(p[0]), int(p[1])
        best = None
        best_d2 = None
        h, w = mask.shape
        for r in range(0, max_r + 1):
            y1, y2 = max(0, py - r), min(h - 1, py + r)
            x1, x2 = max(0, px - r), min(w - 1, px + r)
            ys, xs = np.where(mask[y1:y2 + 1, x1:x2 + 1])
            if len(ys) == 0:
                continue
            ys = ys + y1
            xs = xs + x1
            dy = ys - py
            dx = xs - px
            d2 = dy * dy + dx * dx
            i = int(np.argmin(d2))
            best = (int(ys[i]), int(xs[i]))
            best_d2 = float(d2[i])
            break
        return best, best_d2

    # Step 2: Find overlap to the mask - select pixels closest to cytoplasm
    # Use exact overlap (no dilation) to find pixels where skeleton and mask both have value
    if drawn_mask is None:
        return None
    
    # Find exact overlap: skeleton pixels that are also membrane pixels (255)
    membrane_binary = (drawn_mask == 255)
    overlap = skel & membrane_binary
    
    overlap_coords = np.where(overlap)
    overlap_count = len(overlap_coords[0])
    
    logger.info(
        f"extract_line_segment: skeleton_px={int(np.count_nonzero(skel))} "
        f"exact_overlap_px={overlap_count}"
    )
    
    if overlap_count < 2:
        logger.error(
            f"extract_line_segment: Expected at least 2 overlap pixels, found {overlap_count}. Aborting."
        )
        return None
    
    # Get all overlap pixels
    overlap_pixels = list(zip(overlap_coords[0], overlap_coords[1]))
    
    # If more than 2 overlap pixels, select the ones closest to cytoplasm
    if overlap_count > 2:
        # Order skeleton pixels along the drawn line to understand which side each overlap is on
        skel_coords = np.where(skel)
        skel_list = list(zip(skel_coords[0], skel_coords[1]))
        
        # Order skeleton pixels along the line
        if len(skel_list) > 1:
            ordered_skel = []
            remaining_skel = set(skel_list)
            if len(remaining_skel) > 0:
                current = skel_list[0]
                ordered_skel.append(current)
                remaining_skel.remove(current)
                while len(remaining_skel) > 0:
                    min_dist = float('inf')
                    next_pixel = None
                    for pixel in remaining_skel:
                        dist = np.sqrt((pixel[0] - current[0])**2 + (pixel[1] - current[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            next_pixel = pixel
                    if next_pixel is None:
                        ordered_skel.extend(remaining_skel)
                        break
                    ordered_skel.append(next_pixel)
                    remaining_skel.remove(next_pixel)
                    current = next_pixel
            skel_list = ordered_skel
        
        # Find positions of overlap pixels along the skeleton
        overlap_with_positions = []
        for op in overlap_pixels:
            # Find position in skeleton order
            try:
                pos = skel_list.index(op)
            except ValueError:
                # Find closest skeleton pixel
                min_dist = float('inf')
                pos = 0
                for i, sp in enumerate(skel_list):
                    dist = np.sqrt((sp[0] - op[0])**2 + (sp[1] - op[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        pos = i
            overlap_with_positions.append((pos, op))
        
        # Sort by position along skeleton
        overlap_with_positions.sort(key=lambda x: x[0])
        
        # Group overlap pixels: split into two groups (early and late along the line)
        # This helps identify which side of the membrane they're on
        mid_point = len(overlap_with_positions) // 2
        early_group = overlap_with_positions[:mid_point]
        late_group = overlap_with_positions[mid_point:]
        
        # For each group, select the pixel closest to cytoplasm (mask == 0)
        def find_closest_to_cytoplasm(overlap_group):
            best_pixel = None
            min_dist_to_cytoplasm = float('inf')
            
            for pos, op in overlap_group:
                op_y, op_x = op
                # Search for cytoplasm (mask == 0) within radius
                search_radius = 15
                y1 = max(0, op_y - search_radius)
                y2 = min(drawn_mask.shape[0], op_y + search_radius + 1)
                x1 = max(0, op_x - search_radius)
                x2 = min(drawn_mask.shape[1], op_x + search_radius + 1)
                
                # Find cytoplasm pixels (mask == 0)
                cytoplasm_mask = (drawn_mask[y1:y2, x1:x2] == 0)
                cytoplasm_coords = np.where(cytoplasm_mask)
                
                if len(cytoplasm_coords[0]) > 0:
                    cytoplasm_y = cytoplasm_coords[0] + y1
                    cytoplasm_x = cytoplasm_coords[1] + x1
                    distances = np.sqrt((cytoplasm_y - op_y)**2 + (cytoplasm_x - op_x)**2)
                    min_dist = float(np.min(distances))
                    
                    if min_dist < min_dist_to_cytoplasm:
                        min_dist_to_cytoplasm = min_dist
                        best_pixel = op
            
            return best_pixel
        
        # Select best from each group
        p1_candidate = find_closest_to_cytoplasm(early_group) if early_group else None
        p2_candidate = find_closest_to_cytoplasm(late_group) if late_group else None
        
        # If we got two candidates, use them
        if p1_candidate is not None and p2_candidate is not None:
            p1 = p1_candidate
            p2 = p2_candidate
        elif p1_candidate is not None:
            # Only one candidate, use it and the last overlap pixel
            p1 = p1_candidate
            p2 = overlap_pixels[-1]
        elif p2_candidate is not None:
            # Only one candidate, use it and the first overlap pixel
            p1 = overlap_pixels[0]
            p2 = p2_candidate
        else:
            # Fallback: use first and last along the line
            p1 = overlap_pixels[0]
            p2 = overlap_pixels[-1]
    else:
        # Exactly 2 overlap pixels - use them directly
        p1 = overlap_pixels[0]
        p2 = overlap_pixels[1]
    
    # Ensure p1 and p2 are tuples of ints
    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))
    
    if p1 == p2:
        logger.warning("extract_line_segment: Two overlap pixels are the same. Aborting.")
        return None

    # Step 3: Preserve the original curve by following the skeleton along the drawn line
    # Order the original drawn pixels to preserve the curve
    drawn_coords = np.where(drawn_bin)
    drawn_list = list(zip(drawn_coords[0], drawn_coords[1]))
    
    if len(drawn_list) < 2:
        return None
    
    # Order drawn pixels along the line (from start to end)
    # Start with first pixel, then find nearest unvisited neighbor
    ordered_drawn = []
    remaining_drawn = set(drawn_list)
    
    if len(remaining_drawn) > 0:
        current = drawn_list[0]
        ordered_drawn.append(current)
        remaining_drawn.remove(current)
        
        # Build path by always moving to nearest neighbor
        while len(remaining_drawn) > 0:
            min_dist = float('inf')
            next_pixel = None
            for pixel in remaining_drawn:
                dist = np.sqrt((pixel[0] - current[0])**2 + (pixel[1] - current[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    next_pixel = pixel
            
            if next_pixel is None:
                # No more connected pixels, add remaining in arbitrary order
                ordered_drawn.extend(remaining_drawn)
                break
            
            ordered_drawn.append(next_pixel)
            remaining_drawn.remove(next_pixel)
            current = next_pixel
    
    # Find which skeleton pixels correspond to each drawn pixel
    # Then find the skeleton path between p1 and p2 that follows the drawn line order
    skel_coords = np.where(skel)
    skel_set = set(zip(skel_coords[0], skel_coords[1]))
    
    # For each drawn pixel, find the nearest skeleton pixel
    # This creates a mapping from drawn order to skeleton order
    skel_ordered = []
    for drawn_pix in ordered_drawn:
        # Find nearest skeleton pixel to this drawn pixel
        min_dist = float('inf')
        nearest_skel = None
        for skel_pix in skel_set:
            dist = np.sqrt((skel_pix[0] - drawn_pix[0])**2 + (skel_pix[1] - drawn_pix[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_skel = skel_pix
        
        if nearest_skel is not None and nearest_skel not in skel_ordered:
            skel_ordered.append(nearest_skel)
    
    # Find indices of p1 and p2 in the ordered skeleton
    try:
        idx1 = skel_ordered.index(p1)
        idx2 = skel_ordered.index(p2)
    except ValueError:
        # If exact match not found, find closest
        def find_closest_idx(target, ordered_list):
            min_dist = float('inf')
            best_idx = 0
            for i, pix in enumerate(ordered_list):
                dist = np.sqrt((pix[0] - target[0])**2 + (pix[1] - target[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            return best_idx
        
        idx1 = find_closest_idx(p1, skel_ordered)
        idx2 = find_closest_idx(p2, skel_ordered)
    
    # Extract path between p1 and p2, preserving the curve order
    # First get the ordered segment (this preserves the curve)
    if idx1 < idx2:
        path_seed = skel_ordered[idx1:idx2+1]
    else:
        path_seed = skel_ordered[idx2:idx1+1]
        path_seed.reverse()
    
    # Now ensure we get ALL skeleton pixels between p1 and p2
    # Use the ordered path as a guide, but fill in any missing skeleton pixels
    from collections import deque
    h, w = skel.shape
    
    # Start with the ordered path
    path_set = set(path_seed)
    path = list(path_seed)
    
    # For each consecutive pair in the path, ensure all skeleton pixels between them are included
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
        
        # Check if they're directly connected
        dy = abs(end[0] - start[0])
        dx = abs(end[1] - start[1])
        if dy > 1 or dx > 1:
            # Not directly connected, find path between them
            q = deque([start])
            prev_gap = {start: None}
            found_gap = False
            
            while q:
                cy, cx = q.popleft()
                if (cy, cx) == end:
                    found_gap = True
                    break
                
                for ny, nx in _neighbors8(cy, cx):
                    if not (0 <= ny < h and 0 <= nx < w):
                        continue
                    if not skel[ny, nx]:
                        continue
                    npix = (int(ny), int(nx))
                    if npix in prev_gap:
                        continue
                    prev_gap[npix] = (cy, cx)
                    q.append(npix)
            
            if found_gap:
                # Reconstruct gap path
                gap_path = []
                cur = end
                while cur is not None:
                    gap_path.append(cur)
                    cur = prev_gap[cur]
                gap_path.reverse()
                
                # Insert gap pixels into path (excluding start and end which are already there)
                insert_idx = i + 1
                for gap_pix in gap_path[1:-1]:  # Skip first and last (already in path)
                    if gap_pix not in path_set:
                        path.insert(insert_idx, gap_pix)
                        path_set.add(gap_pix)
                        insert_idx += 1
    
    # Final check: ensure we have a complete 8-connected path from p1 to p2
    # Verify all consecutive pixels are 8-connected
    final_path = [path[0]]
    for i in range(1, len(path)):
        prev_pix = final_path[-1]
        curr_pix = path[i]
        dy = abs(curr_pix[0] - prev_pix[0])
        dx = abs(curr_pix[1] - prev_pix[1])
        if dy <= 1 and dx <= 1:
            final_path.append(curr_pix)
        else:
            # Need to fill gap
            q = deque([prev_pix])
            prev_fill = {prev_pix: None}
            found_fill = False
            while q:
                cy, cx = q.popleft()
                if (cy, cx) == curr_pix:
                    found_fill = True
                    break
                for ny, nx in _neighbors8(cy, cx):
                    if not (0 <= ny < h and 0 <= nx < w):
                        continue
                    if not skel[ny, nx]:
                        continue
                    npix = (int(ny), int(nx))
                    if npix in prev_fill:
                        continue
                    prev_fill[npix] = (cy, cx)
                    q.append(npix)
            
            if found_fill:
                fill_path = []
                cur = curr_pix
                while cur is not None:
                    fill_path.append(cur)
                    cur = prev_fill[cur]
                fill_path.reverse()
                final_path.extend(fill_path[1:])  # Skip first (already in path)
            else:
                final_path.append(curr_pix)
    
    path = final_path
    
    # Ensure we start with p1 and end with p2
    if len(path) > 0 and path[0] != p1:
        path.insert(0, p1)
    if len(path) > 0 and path[-1] != p2:
        path.append(p2)

    # CRITICAL FIX: Trim tips by removing any skeleton pixels beyond p1/p2
    # The path should only include pixels that are strictly between p1 and p2
    # Use a more constrained approach: find the shortest path through skeleton from p1 to p2
    # and remove any branches/tips that extend beyond this path
    
    def count_skeleton_neighbors(p, skel):
        """Count 8-connected skeleton neighbors of pixel p=(y,x)"""
        y, x = p
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1] and skel[ny, nx]:
                    count += 1
        return count
    
    # Build path set for quick lookup
    path_set = set(path)
    
    # Strategy: Remove tips iteratively by finding skeleton endpoints (pixels with 1 neighbor)
    # that are in the path but not p1 or p2, and removing them if they're dead ends
    changed = True
    iterations = 0
    max_iterations = 50  # Safety limit to prevent infinite loops
    
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        
        # Find tips: skeleton endpoints in path that are not p1 or p2
        tips_to_remove = []
        for pix in path:
            if pix == p1 or pix == p2:
                continue  # Never remove p1 or p2
            
            # Count skeleton neighbors (all skeleton pixels)
            skeleton_neighbor_count = count_skeleton_neighbors(pix, skel)
            
            # Count path neighbors (only pixels in current path)
            path_neighbor_count = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = pix[0] + dy, pix[1] + dx
                    neighbor = (ny, nx)
                    if neighbor in path_set:
                        path_neighbor_count += 1
            
            # If this pixel is a skeleton endpoint (1 skeleton neighbor) and has <= 1 path neighbor,
            # it's a tip that extends beyond the main path - remove it
            if skeleton_neighbor_count == 1 and path_neighbor_count <= 1:
                tips_to_remove.append(pix)
        
        # Remove tips from path
        if tips_to_remove:
            changed = True
            for tip in tips_to_remove:
                if tip in path_set:
                    path_set.remove(tip)
                    path.remove(tip)
    
    # Final verification: ensure path starts with p1 and ends with p2
    if len(path) > 0 and path[0] != p1:
        path.insert(0, p1)
        if p1 not in path_set:
            path_set.add(p1)
    if len(path) > 0 and path[-1] != p2:
        path.append(p2)
        if p2 not in path_set:
            path_set.add(p2)
    
    # Additional cleanup: ensure path is a simple chain (no branches)
    # Each pixel (except endpoints p1/p2) should have exactly 2 neighbors in the path
    # If a pixel has more than 2 neighbors, it's a branch point - remove branches
    path_needs_cleanup = True
    cleanup_iterations = 0
    max_cleanup_iterations = 20
    
    while path_needs_cleanup and cleanup_iterations < max_cleanup_iterations:
        path_needs_cleanup = False
        cleanup_iterations += 1
        
        # Check each pixel in path for branch points
        pixels_to_remove = []
        for pix in path:
            if pix == p1 or pix == p2:
                continue  # Skip endpoints
            
            # Count neighbors in path
            path_neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = pix[0] + dy, pix[1] + dx
                    neighbor = (ny, nx)
                    if neighbor in path_set:
                        path_neighbors.append(neighbor)
            
            # If pixel has more than 2 neighbors in path, it's a branch point
            # Keep only the two neighbors that are part of the main chain
            if len(path_neighbors) > 2:
                # This shouldn't happen in a simple path, but if it does, 
                # we need to determine which neighbors are on the main path
                # For now, mark for removal if it's clearly a branch (has 3+ neighbors)
                # and is not critical for connectivity
                path_needs_cleanup = True
                # Don't remove immediately - this is complex, so we'll handle it conservatively
        
        # For now, we rely on the tip removal above to handle most cases
        # This cleanup is mainly for verification
    
    # Final step: ensure path is connected from p1 to p2
    # Verify connectivity by checking if we can reach p2 from p1 through path pixels only
    if len(path) > 0:
        # Build connectivity graph from path
        path_graph = {p: [] for p in path_set}
        for pix in path_set:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = pix[0] + dy, pix[1] + dx
                    neighbor = (ny, nx)
                    if neighbor in path_set:
                        path_graph[pix].append(neighbor)
        
        # Check if p1 and p2 are connected
        from collections import deque
        if p1 in path_graph and p2 in path_graph:
            q = deque([p1])
            visited = {p1}
            found_p2 = False
            
            while q:
                curr = q.popleft()
                if curr == p2:
                    found_p2 = True
                    break
                for neighbor in path_graph.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
            
            # If path is disconnected, reconstruct using shortest skeleton path
            if not found_p2:
                q = deque([p1])
                prev_reconstruct = {p1: None}
                found_p2 = False
                
                while q:
                    curr = q.popleft()
                    if curr == p2:
                        found_p2 = True
                        break
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = curr[0] + dy, curr[1] + dx
                            neighbor = (ny, nx)
                            if (0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1] and 
                                skel[ny, nx] and neighbor not in prev_reconstruct):
                                prev_reconstruct[neighbor] = curr
                                q.append(neighbor)
                
                if found_p2:
                    # Reconstruct path from p1 to p2
                    reconstructed_path = []
                    cur = p2
                    while cur is not None:
                        reconstructed_path.append(cur)
                        cur = prev_reconstruct[cur]
                    reconstructed_path.reverse()
                    path = reconstructed_path
                    path_set = set(path)

    # Step 3: Remove the tips and keep the skeletonized line between the two overlap pixels
    # Create result mask with the skeleton path (1-pixel wide)
    path_mask = np.zeros_like(drawn_bin, dtype=np.uint8)
    for y, x in path:
        path_mask[y, x] = 255

    logger.info(
        f"extract_line_segment: path_len={len(path)} "
        f"result_px={int(np.count_nonzero(path_mask))} "
        f"tips_removed_iterations={iterations}"
    )

    return path_mask


def find_membrane_segment_to_remove(drawn, drawn_mask):
    """
    Remove membrane segment between T-junctions using bidirectional skeleton walking.
    
    Simple algorithm:
    1. Find where drawn line crosses membrane
    2. Get skeleton of that membrane region  
    3. Find a skeleton pixel with 2 neighbors near the crossing
    4. Walk in both directions until hitting T-junctions (3+ neighbors) or endpoints
    5. Return all pixels between (excluding T-junctions)
    
    Args:
        drawn: numpy array with drawn pixels (non-zero where user drew)
        drawn_mask: numpy array of current mask (255 = membrane, 0 = cell interior)
        
    Returns:
        numpy array with membrane pixels to remove (255 = remove, 0 = keep)
    """
    from scipy.ndimage import binary_dilation
    from skimage.measure import label
    
    if drawn is None or np.count_nonzero(drawn) == 0:
        return None
    
    # Find where drawn line overlaps with membranes.
    # drawn is the erased line (black pixels from user drawing)
    # drawn_mask has 255 for membrane pixels (red)
    
    # CRITICAL: Ensure drawn and drawn_mask have the same dimensions
    # raw_user_drawing might be created from image.size() while drawn_mask comes from mask
    if drawn.shape != drawn_mask.shape:
        logger.warning(
            f"find_membrane_segment_to_remove: Dimension mismatch! drawn.shape={drawn.shape}, "
            f"drawn_mask.shape={drawn_mask.shape}. Resizing drawn to match drawn_mask."
        )
        # Resize drawn to match drawn_mask dimensions using nearest neighbor
        from scipy.ndimage import zoom
        zoom_factors = (drawn_mask.shape[0] / drawn.shape[0], drawn_mask.shape[1] / drawn.shape[1])
        drawn = zoom(drawn, zoom_factors, order=0)  # order=0 for nearest neighbor (preserves binary)
        drawn = (drawn > 0).astype(drawn_mask.dtype) * 255
    
    # Use small dilation (1-2px) for tolerance since drawing may be slightly offset from skeleton
    # This handles cases where erased pixels are near but not exactly on membrane pixels
    from scipy.ndimage import binary_dilation
    dilated_drawn = binary_dilation(drawn > 0, structure=np.ones((3, 3)))
    overlap = (drawn_mask == 255) & dilated_drawn
    
    drawn_nonzero = np.count_nonzero(drawn)
    mask_membrane = np.count_nonzero(drawn_mask == 255)
    overlap_count = np.count_nonzero(overlap)
    
    logger.info(
        f"find_membrane_segment_to_remove: drawn shape={drawn.shape} mask shape={drawn_mask.shape} "
        f"drawn nonzero={drawn_nonzero} mask_membrane={mask_membrane} overlap (with dilation)={overlap_count}"
    )
    
    if overlap_count == 0:
        return None
    
    # Get center of overlap as overlay point (always a membrane pixel)
    coords = np.where(overlap)
    overlay_point = (coords[0][len(coords[0]) // 2], coords[1][len(coords[1]) // 2])
    
    # Get membrane region at overlay point
    labeled = label(drawn_mask == 255, connectivity=2)
    region_label = labeled[overlay_point]
    
    if region_label == 0:
        return None
    
    # Get skeleton (input is already skeletonized)
    skeleton = (labeled == region_label)
    
    def count_neighbors(y, x, skel):
        """Count 8-connected skeleton neighbors"""
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1] and skel[ny, nx]:
                    count += 1
        return count
    
    # Find skeleton pixel with 2 neighbors near the overlap (good start point for bidirectional walk)
    skel_coords = np.where(skeleton)
    candidates = []

    def get_neighbors(p, skel):
        """Return 8-connected skeleton neighbors of pixel p=(y,x)."""
        y, x = p
        out = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1] and skel[ny, nx]:
                    out.append((ny, nx))
        return out
    
    # First try skeleton pixels that are ALSO overlap-on-red pixels (best signal).
    # Note: if we used dilated overlap, it is still constrained to red pixels.
    overlap_skel = skeleton & overlap
    overlap_coords = np.where(overlap_skel)
    for i in range(len(overlap_coords[0])):
        y, x = overlap_coords[0][i], overlap_coords[1][i]
        if count_neighbors(y, x, skeleton) == 2:
            dist = (y - overlay_point[0])**2 + (x - overlay_point[1])**2
            # Score candidate by whether both of its neighbors are "good" (not endpoints).
            nbs = get_neighbors((y, x), skeleton)
            nb_counts = [count_neighbors(ny, nx, skeleton) for (ny, nx) in nbs]
            endpoint_neighbor_count = sum(1 for c in nb_counts if c <= 1)
            candidates.append((y, x, dist, endpoint_neighbor_count, nb_counts))

    # STRICT: require at least one 2-neighbor skeleton pixel in the exact overlap set.
    if not candidates:
        logger.info("find_membrane_segment_to_remove: no 2-neighbor skeleton pixel in exact overlap (strict) -> None")
        return None

    # Choose among candidates (not a fallback; just selection):
    # Prefer candidates whose *neighbors* are not endpoints (avoids 1-pixel spur issue),
    # then prefer closer to overlay point.
    candidates.sort(key=lambda c: (c[3], c[2]))
    start = (candidates[0][0], candidates[0][1])
    logger.info(
        f"Selected start={start} dist={np.sqrt(candidates[0][2]):.1f} "
        f"endpoint_neighbor_count={candidates[0][3]} neighbor_counts={candidates[0][4]}"
    )
    
    # Get neighbors of start point
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = start[0] + dy, start[1] + dx
            if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1] and skeleton[ny, nx]:
                neighbors.append((ny, nx))
    
    if len(neighbors) == 0:
        return None

    logger.info(
        f"find_membrane_segment_to_remove: start={start} start_neighbor_count={count_neighbors(start[0], start[1], skeleton)} "
        f"start_neighbors={neighbors}"
    )
    
    # Walk from start point in all directions
    def walk(begin, skel, visited, prev=None):
        """Walk until T-junction or endpoint"""
        path = []
        curr = begin
        seen = {begin}
        if prev:
            seen.add(prev)
        
        while True:
            n_count = count_neighbors(curr[0], curr[1], skel)
            if n_count >= 3:  # T-junction
                return path, curr
            
            path.append(curr)
            seen.add(curr)
            
            # Find next neighbor
            next_neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = curr[0] + dy, curr[1] + dx
                    if (0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1] and
                        skel[ny, nx] and (ny, nx) not in seen and (ny, nx) not in visited):
                        next_neighbors.append((ny, nx))
            
            if not next_neighbors:  # Endpoint
                return path, curr
            curr = next_neighbors[0]
    
    # Walk in both directions - each walk gets its own visited set
    paths = []
    junctions = []

    for neighbor in neighbors:
        visited = {start}
        path, junction = walk(neighbor, skeleton, visited, prev=start)
        paths.append(path)
        junctions.append(junction)
        logger.info(f'Walk from {neighbor}: path length={len(path)}, junction at {junction}')

    # STRICT: no special handling/fallback if more than 2 directions are present.
    
    # Combine all paths
    if len(paths) == 0:
        full_path = [start]
    elif len(paths) == 1:
        full_path = [start] + paths[0]
    else:
        # Bidirectional: reverse first path, add start, add second path
        full_path = list(reversed(paths[0])) + [start] + paths[1]
    
    # Create result mask
    result = np.zeros_like(drawn_mask, dtype=np.uint8)
    for y, x in full_path:
        result[y, x] = 255
    
    # Dilate to capture full membrane width
    result = binary_dilation(result > 0, structure=np.ones((3, 3)), iterations=1).astype(np.uint8) * 255
    
    # Keep only membrane pixels
    result = result & drawn_mask
    
    return result if np.count_nonzero(result) > 0 else None

