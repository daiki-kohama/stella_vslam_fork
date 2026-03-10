#!/usr/bin/env python3

import cv2
import argparse
import os
from pathlib import Path
from match_viz_common import (
    load_matches_from_csv,
    get_frame_from_video,
    draw_matches_on_frames,
    load_local_map_points_from_csv,
    load_mapping_points_from_csv,
    draw_local_map_points,
    draw_mapping_points,
)

def main():
    parser = argparse.ArgumentParser(description="Visualize matching points from CSV and video")
    parser.add_argument("csv_path", type=str, help="Path to CSV file containing matching points")
    parser.add_argument("video_path", type=str, help="Path to video file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory to save images. If not specified, images will be displayed with imshow",
    )
    parser.add_argument(
        "--local-map-csv",
        type=str,
        default=None,
        help="Optional CSV (frm_id,u,v,...) generated from optimize_current_frame_with_local_map. If provided, points are overlaid as green radius-3 circles.",
    )
    parser.add_argument(
        "--mapping-csv",
        type=str,
        default=None,
        help="Optional CSV (keyfrm_id,frm_id,lm_id,u,v,num_observations) generated from mapping_with_new_keyframe. If provided, points are overlaid as red radius-4 circles.",
    )

    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        return

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return

    local_map_points = None
    if args.local_map_csv:
        if not os.path.exists(args.local_map_csv):
            print(f"Error: local-map CSV file not found: {args.local_map_csv}")
            return
        try:
            local_map_points = load_local_map_points_from_csv(args.local_map_csv)
            print(f"Loaded local-map points for {len(local_map_points)} frame(s) from {args.local_map_csv}")
            print("注釈: 緑丸は、トラッキングモジュールでローカルマップ上のランドマークとマッチさせて最適化した後のランドマークと結びつきのある点です。")
        except Exception as e:
            print(f"Error: Failed to load local-map CSV: {e}")
            return

    mapping_points = None
    if args.mapping_csv:
        if not os.path.exists(args.mapping_csv):
            print(f"Error: mapping CSV file not found: {args.mapping_csv}")
            return
        try:
            mapping_points = load_mapping_points_from_csv(args.mapping_csv)
            print(f"Loaded mapping points for {len(mapping_points)} frame(s) from {args.mapping_csv}")
            print("注釈: 赤丸は、マッピングモジュールで新規キーフレームに登録されたランドマークと結びつきのある特徴点です。")
        except Exception as e:
            print(f"Error: Failed to load mapping CSV: {e}")
            return

    # Load matches from CSV
    print(f"Loading matches from {args.csv_path}...")
    matches = load_matches_from_csv(args.csv_path)
    print(f"Loaded {len(matches)} match pairs")

    # Create output directory if specified
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {args.output_dir}")

    # Process each match pair
    for (frm1_id, frm2_id), matches_list in sorted(matches.items()):
        print(f"Processing frames {frm1_id} and {frm2_id}...")

        # Load frames from video
        frm1 = get_frame_from_video(args.video_path, frm1_id)
        frm2 = get_frame_from_video(args.video_path, frm2_id)

        if frm1 is None or frm2 is None:
            print(f"  Warning: Could not load frames {frm1_id} or {frm2_id}")
            continue

        # Draw matches
        img_with_matches = draw_matches_on_frames(frm1, frm2, matches_list)

        # Overlay additional points if CSVs are provided
        h1 = frm1.shape[0]
        draw_local_map_points(img_with_matches, local_map_points, frm1_id, frm2_id, h1)
        draw_mapping_points(img_with_matches, mapping_points, frm1_id, frm2_id, h1)

        if args.output_dir:
            # Save to file
            output_path = os.path.join(args.output_dir, f"{frm1_id}_{frm2_id}.png")
            cv2.imwrite(output_path, img_with_matches)
            print(f"  Saved to {output_path}")
        else:
            # Display with imshow
            print(f"  Displaying matches ({len(matches_list)} matches)")
            # Create resizable window
            cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
            cv2.imshow("Matches", img_with_matches)
            key = cv2.waitKey(0)
            if key == ord("q"):
                break

    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()
