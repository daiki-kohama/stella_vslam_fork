#!/usr/bin/env python3

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from match_viz_common import (
    load_matches_from_csv,
    load_local_map_points_from_csv,
    load_mapping_points_from_csv,
)


def load_sources_from_config(config_path, default_cols, default_rows, default_width, default_height):
    """Load input sources from JSON config.

    Config example:
    {
      "grid": {"cols": 36, "rows": 18},
      "sources": [
        {"matches_dir": "matches_A", "width": 3840, "height": 1920},
        {"matches_dir": "matches_B", "width": 1920, "height": 960}
      ]
    }

    Returns:
        (sources, grid_cols, grid_rows)
        sources: list of dicts {"matches_dir": Path, "width": int, "height": int}
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    grid_cfg = cfg.get("grid", {})
    grid_cols = int(grid_cfg.get("cols", default_cols))
    grid_rows = int(grid_cfg.get("rows", default_rows))

    sources = []
    for src in cfg.get("sources", []):
        matches_dir = Path(src["matches_dir"])
        width = int(src.get("width", default_width))
        height = int(src.get("height", default_height))
        sources.append(
            {
                "matches_dir": matches_dir,
                "width": width,
                "height": height,
            }
        )

    return sources, grid_cols, grid_rows


def merge_count_dict(dst, src):
    """Merge count dictionaries by summation."""
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + v


def get_grid_cell(u, v, img_width, img_height, grid_cols, grid_rows):
    """Get grid cell index from normalized coordinates.

    Args:
        u, v: Image coordinates (0-based)
        img_width, img_height: Image dimensions
        grid_cols, grid_rows: Number of grid columns and rows

    Returns:
        (col_idx, row_idx) grid cell index, or None if out of bounds
    """
    if u < 0 or u >= img_width or v < 0 or v >= img_height:
        return None

    col_idx = int((u / img_width) * grid_cols)
    row_idx = int((v / img_height) * grid_rows)

    # Clamp to valid range
    col_idx = min(col_idx, grid_cols - 1)
    row_idx = min(row_idx, grid_rows - 1)

    return (col_idx, row_idx)


def process_matches_csv(csv_path, img_width, img_height, grid_cols, grid_rows):
    """Process matches CSV and return grid statistics.

    Args:
        csv_path: Path to CSV file
        img_width, img_height: Image dimensions
        grid_cols, grid_rows: Number of grid columns and rows

    Returns:
        (total_points, inlier_points, used_frames) where used_frames is a set of frame IDs
    """
    total_points = {}
    inlier_points = {}
    used_frames = set()

    if not os.path.exists(csv_path):
        return total_points, inlier_points, used_frames

    try:
        matches = load_matches_from_csv(csv_path)
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return total_points, inlier_points, used_frames

    # Process all matches
    for (frm1_id, frm2_id), match_list in matches.items():
        used_frames.add(frm1_id)
        used_frames.add(frm2_id)
        for match in match_list:
            u1 = match["frm1_pt"][0]
            v1 = match["frm1_pt"][1]
            u2 = match["frm2_pt"][0]
            v2 = match["frm2_pt"][1]
            is_outlier = match.get("is_outlier", False)

            # Process first frame point
            cell = get_grid_cell(u1, v1, img_width, img_height, grid_cols, grid_rows)
            if cell:
                total_points[cell] = total_points.get(cell, 0) + 1
                if not is_outlier:
                    inlier_points[cell] = inlier_points.get(cell, 0) + 1

            # # Process second frame point
            # cell = get_grid_cell(u2, v2, img_width, img_height, grid_cols, grid_rows)
            # if cell:
            #     total_points[cell] = total_points.get(cell, 0) + 1
            #     if not is_outlier:
            #         inlier_points[cell] = inlier_points.get(cell, 0) + 1

    return total_points, inlier_points, used_frames


def process_local_map_csv(csv_path, img_width, img_height, grid_cols, grid_rows):
    """Process local map tracking CSV and return grid statistics.

    Args:
        csv_path: Path to CSV file
        img_width, img_height: Image dimensions
        grid_cols, grid_rows: Number of grid columns and rows

    Returns:
        (total_points, inlier_points, used_frames) where used_frames is a set of frame IDs
    """
    total_points = {}
    inlier_points = {}
    used_frames = set()

    if not os.path.exists(csv_path):
        return total_points, inlier_points, used_frames

    try:
        points_by_frame = load_local_map_points_from_csv(csv_path)
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return total_points, inlier_points, used_frames

    # Process all points
    for frm_id, points in points_by_frame.items():
        used_frames.add(frm_id)
        for point_data in points:
            u = point_data[0]
            v = point_data[1]
            is_outlier = point_data[2] if len(point_data) > 2 else False

            cell = get_grid_cell(u, v, img_width, img_height, grid_cols, grid_rows)
            if cell:
                total_points[cell] = total_points.get(cell, 0) + 1
                if not is_outlier:
                    inlier_points[cell] = inlier_points.get(cell, 0) + 1

    return total_points, inlier_points, used_frames


def process_mapping_csv(csv_path, img_width, img_height, grid_cols, grid_rows):
    """Process mapping keyframe CSV and return grid statistics.

    Args:
        csv_path: Path to CSV file
        img_width, img_height: Image dimensions
        grid_cols, grid_rows: Number of grid columns and rows

    Returns:
        (total_points, inlier_points, used_frames) where used_frames is a set of frame IDs
    """
    total_points = {}
    inlier_points = {}
    used_frames = set()

    if not os.path.exists(csv_path):
        return total_points, inlier_points, used_frames

    try:
        points_by_frame = load_mapping_points_from_csv(csv_path)
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return total_points, inlier_points, used_frames

    # Process all points
    for frm_id, points in points_by_frame.items():
        used_frames.add(frm_id)
        for point_data in points:
            u = point_data[0]
            v = point_data[1]
            is_outlier = point_data[2] if len(point_data) > 2 else False

            cell = get_grid_cell(u, v, img_width, img_height, grid_cols, grid_rows)
            if cell:
                total_points[cell] = total_points.get(cell, 0) + 1
                if not is_outlier:
                    inlier_points[cell] = inlier_points.get(cell, 0) + 1

    return total_points, inlier_points, used_frames


def compute_inlier_ratio_heatmap(total_points, inlier_points, grid_cols, grid_rows):
    """Compute inlier ratio heatmap.

    Args:
        total_points, inlier_points: Grid statistics from processing
        grid_cols, grid_rows: Number of grid columns and rows

    Returns:
        2D numpy array of inlier ratios (0.0 to 1.0)
    """
    heatmap = np.zeros((grid_rows, grid_cols), dtype=np.float32)

    for row in range(grid_rows):
        for col in range(grid_cols):
            cell = (col, row)
            total = total_points.get(cell, 0)
            if total > 0:
                inlier = inlier_points.get(cell, 0)
                heatmap[row, col] = inlier / total
            else:
                heatmap[row, col] = np.nan

    return heatmap


def compute_point_count_heatmap(total_points, grid_cols, grid_rows):
    """Compute point count heatmap.

    Args:
        total_points: Grid statistics mapping (col_idx, row_idx) -> count
        grid_cols, grid_rows: Number of grid columns and rows

    Returns:
        2D numpy array of point counts
    """
    heatmap = np.zeros((grid_rows, grid_cols), dtype=np.float32)

    for row in range(grid_rows):
        for col in range(grid_cols):
            cell = (col, row)
            heatmap[row, col] = total_points.get(cell, 0)

    return heatmap


def visualize_heatmap(heatmap_ratio, heatmap_count, title, filename=None):
    """Visualize two heatmaps vertically: inlier ratio and point count.

    Args:
        heatmap_ratio: 2D numpy array of inlier ratios (0.0 to 1.0)
        heatmap_count: 2D numpy array of point counts
        title: Title for the figure
        filename: Optional filename to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

    # Top subplot: Inlier Ratio
    masked_ratio = np.ma.masked_invalid(heatmap_ratio)
    im1 = ax1.imshow(masked_ratio, cmap="RdYlGn", vmin=0, vmax=1, origin="upper")
    cbar1 = plt.colorbar(im1, ax=ax1, label="Inlier Ratio")
    ax1.set_title(f"{title} - Inlier Ratio")
    ax1.set_xlabel("Grid Column")
    ax1.set_ylabel("Grid Row")

    # Add grid lines to first subplot
    rows, cols = heatmap_ratio.shape
    for i in range(rows + 1):
        ax1.axhline(i - 0.5, color="gray", linewidth=0.5, alpha=0.3)
    for j in range(cols + 1):
        ax1.axvline(j - 0.5, color="gray", linewidth=0.5, alpha=0.3)

    # Bottom subplot: Point Count
    im2 = ax2.imshow(heatmap_count, cmap="viridis", origin="upper")
    cbar2 = plt.colorbar(im2, ax=ax2, label="Point Count")
    ax2.set_title(f"{title} - Point Count")
    ax2.set_xlabel("Grid Column")
    ax2.set_ylabel("Grid Row")

    # Add grid lines to second subplot
    rows, cols = heatmap_count.shape
    for i in range(rows + 1):
        ax2.axhline(i - 0.5, color="gray", linewidth=0.5, alpha=0.3)
    for j in range(cols + 1):
        ax2.axvline(j - 0.5, color="gray", linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=100, bbox_inches="tight")
        print(f"Saved: {filename}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize feature point inlier ratio as heatmap")
    parser.add_argument(
        "matches_dirs",
        nargs="*",
        type=str,
        help="One or more directories containing CSV files with match information",
    )
    parser.add_argument(
        "--width", type=int, default=3840, help="Image width for grid calculation (default: 3840)"
    )
    parser.add_argument(
        "--height", type=int, default=1920, help="Image height for grid calculation (default: 1920)"
    )
    parser.add_argument("--cols", type=int, default=36, help="Number of grid columns (default: 36)")
    parser.add_argument("--rows", type=int, default=18, help="Number of grid rows (default: 18)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for heatmap images (default: display only)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON config path for multiple directories and per-directory width/height",
    )

    args = parser.parse_args()

    grid_cols = args.cols
    grid_rows = args.rows

    sources = []
    if args.config:
        try:
            sources, grid_cols, grid_rows = load_sources_from_config(
                args.config, args.cols, args.rows, args.width, args.height
            )
        except Exception as e:
            print(f"Error loading config {args.config}: {e}")
            return
    else:
        if not args.matches_dirs:
            print("Error: specify matches_dirs or --config")
            return
        for d in args.matches_dirs:
            sources.append(
                {
                    "matches_dir": Path(d),
                    "width": args.width,
                    "height": args.height,
                }
            )

    # Validate source directories
    valid_sources = []
    for src in sources:
        if src["matches_dir"].exists():
            valid_sources.append(src)
        else:
            print(f"Skipping directory (not found): {src['matches_dir']}")

    if not valid_sources:
        print("Error: no valid matches directories")
        return

    output_dir = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

    # CSV file processors: (pattern, processor_function)
    csv_processors = [
        ("motion_based_tracking.csv", process_matches_csv),
        ("bow_match_based_tracking.csv", process_matches_csv),
        ("robust_match_based_tracking.csv", process_matches_csv),
        ("optimize_current_frame_with_local_map.csv", process_local_map_csv),
        ("mapping_with_new_keyframe.csv", process_mapping_csv),
    ]

    for csv_pattern, processor in csv_processors:
        print(f"Processing {csv_pattern}...")

        aggregated_total = {}
        aggregated_inlier = {}
        aggregated_frames = set()
        used_source_count = 0

        for src in valid_sources:
            csv_path = src["matches_dir"] / csv_pattern
            if not csv_path.exists():
                continue

            total_points, inlier_points, used_frames = processor(
                str(csv_path), src["width"], src["height"], grid_cols, grid_rows
            )
            merge_count_dict(aggregated_total, total_points)
            merge_count_dict(aggregated_inlier, inlier_points)
            aggregated_frames.update(used_frames)
            used_source_count += 1

        if used_source_count == 0:
            print(f"  Skipping {csv_pattern} (not found in all input directories)")
            continue

        if not aggregated_total:
            print(f"  No points found in {csv_pattern}")
            continue

        heatmap_ratio = compute_inlier_ratio_heatmap(
            aggregated_total, aggregated_inlier, grid_cols, grid_rows
        )
        heatmap_count = compute_point_count_heatmap(aggregated_total, grid_cols, grid_rows)

        # Print statistics
        total_count = sum(aggregated_total.values())
        inlier_count = sum(aggregated_inlier.values())
        overall_ratio = inlier_count / total_count if total_count > 0 else 0.0
        frame_count = len(aggregated_frames)
        print(
            f"  Sources used: {used_source_count}, Frames: {frame_count}, "
            f"Total points: {total_count}, Inliers: {inlier_count}, Ratio: {overall_ratio:.3f}"
        )

        # Visualize
        output_file = None
        if output_dir:
            output_file = str(output_dir / f"{csv_pattern.replace('.csv', '')}_heatmap.png")

        visualize_heatmap(
            heatmap_ratio, heatmap_count, f'Heatmap - {csv_pattern.replace(".csv", "")}', output_file
        )


if __name__ == "__main__":
    main()
