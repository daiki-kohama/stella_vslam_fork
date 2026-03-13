#!/usr/bin/env python3

import cv2
import argparse
import os
import math
import csv
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
    parser = argparse.ArgumentParser(
        description="Visualize matching points in batches from CSV and video"
    )
    parser.add_argument("csv_path", type=str, help="Path to CSV file containing matching points")
    parser.add_argument("video_path", type=str, help="Path to video file")
    parser.add_argument(
        "-n",
        type=int,
        default=10,
        help="Number of matches to display per batch (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory to save images. If not specified, images will be displayed with imshow",
    )
    parser.add_argument(
        "--scale-by-response",
        action="store_true",
        help="Scale circle radius by response value of keypoints",
    )
    parser.add_argument(
        "--response-scale",
        type=float,
        default=0.5,
        help="Scaling factor for response-based radius (default: 0.5)",
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

    # Create resizable window once for display mode
    if not args.output_dir:
        cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)

    # Convert to list for indexing
    matches_list = sorted(matches.items())
    pair_idx = 0

    # Process each match pair with keyboard navigation
    while pair_idx < len(matches_list):
        frm1_id, frm2_id = matches_list[pair_idx][0]
        matches_data = matches_list[pair_idx][1]

        # Load frames from video
        frm1 = get_frame_from_video(args.video_path, frm1_id)
        frm2 = get_frame_from_video(args.video_path, frm2_id)

        if frm1 is None or frm2 is None:
            print(f"Warning: Could not load frames {frm1_id} or {frm2_id}")
            pair_idx += 1
            continue

        # Calculate number of batches needed
        num_matches = len(matches_data)
        num_batches = math.ceil(num_matches / args.n)

        print(f"\n[{pair_idx + 1}/{len(matches_list)}] Processing frames {frm1_id} and {frm2_id}: {num_matches} matches in {num_batches} batch(es)")
        print("Controls: SPACE/n=next pair, b/p=prev pair, ]/>=+5 pairs, [/<=-5 pairs, 1=+10 pairs, 2=-10 pairs, 3=+100 pairs, 4=-100 pairs")
        print("         .=next batch, ,=prev batch, q=quit")

        # Process each batch with navigation
        batch_idx = 0
        while batch_idx < num_batches:
            start_idx = batch_idx * args.n
            end_idx = min(start_idx + args.n, num_matches)
            batch_matches = matches_data[start_idx:end_idx]

            # Draw matches for this batch
            img_with_matches = draw_matches_on_frames(
                frm1, frm2, batch_matches,
                scale_by_response=args.scale_by_response,
                response_scale=args.response_scale
            )

            # Overlay additional points if CSVs are provided
            h1 = frm1.shape[0]
            draw_local_map_points(img_with_matches, local_map_points, frm1_id, frm2_id, h1)
            draw_mapping_points(img_with_matches, mapping_points, frm1_id, frm2_id, h1)

            if args.output_dir:
                # Save to file
                output_path = os.path.join(
                    args.output_dir,
                    f"{frm1_id}_{frm2_id}_batch{batch_idx + 1}.png"
                )
                cv2.imwrite(output_path, img_with_matches)
                print(f"  Batch {batch_idx + 1}/{num_batches}: Saved to {output_path} ({len(batch_matches)} matches)")
                batch_idx += 1
            else:
                # Display with imshow
                octaves_str = ", ".join([f"({m['frm1_octave']},{m['frm2_octave']})" for m in batch_matches[:3]])
                if len(batch_matches) > 3:
                    octaves_str += "..."
                window_title = (
                    f"Matches [{pair_idx + 1}/{len(matches_list)}] "
                    f"frames {frm1_id}->{frm2_id} "
                    f"batch {batch_idx + 1}/{num_batches} "
                    f"({start_idx + 1}-{end_idx}/{num_matches}) "
                    f"octaves: {octaves_str}"
                )
                print(f"  {window_title}")
                cv2.imshow("Matches", img_with_matches)
                cv2.setWindowTitle("Matches", window_title)
                key = cv2.waitKey(0)

                # Handle keyboard input
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    print("Done!")
                    return
                elif key == ord(" ") or key == ord("n"):
                    # Next frame pair (reset batch to 0)
                    pair_idx += 1
                    batch_idx = 0
                    break
                elif key == ord("b") or key == ord("p"):
                    # Previous frame pair (reset batch to 0)
                    pair_idx -= 1
                    batch_idx = 0
                    break
                elif key == ord("]") or key == ord(">"):
                    # Skip 5 frame pairs forward (reset batch to 0)
                    pair_idx += 5
                    batch_idx = 0
                    break
                elif key == ord("[") or key == ord("<"):
                    # Skip 5 frame pairs backward (reset batch to 0)
                    pair_idx -= 5
                    batch_idx = 0
                    break
                elif key == ord("1"):
                    # Skip 10 frame pairs forward (reset batch to 0)
                    pair_idx += 10
                    batch_idx = 0
                    break
                elif key == ord("2"):
                    # Skip 10 frame pairs backward (reset batch to 0)
                    pair_idx -= 10
                    batch_idx = 0
                    break
                elif key == ord("3"):
                    # Skip 100 frame pairs forward (reset batch to 0)
                    pair_idx += 100
                    batch_idx = 0
                    break
                elif key == ord("4"):
                    # Skip 100 frame pairs backward (reset batch to 0)
                    pair_idx -= 100
                    batch_idx = 0
                    break
                elif key == ord("."):
                    # Next batch in current frame pair
                    batch_idx += 1
                elif key == ord(","):
                    # Previous batch in current frame pair
                    batch_idx -= 1
                    # Handle transition to previous frame pair if going negative
                    if batch_idx < 0:
                        pair_idx -= 1
                        if pair_idx >= 0:
                            prev_matches = matches_list[pair_idx][1]
                            num_prev_batches = math.ceil(len(prev_matches) / args.n)
                            batch_idx = num_prev_batches - 1
                        else:
                            pair_idx = 0
                            batch_idx = 0
                        break
                else:
                    # No navigation, just continue to next batch
                    batch_idx += 1
        else:
            # If we reach the end of batches without breaking, move to next pair
            pair_idx += 1
            batch_idx = 0

        # Ensure pair_idx stays in valid range
        if pair_idx < 0:
            pair_idx = 0
        elif pair_idx >= len(matches_list):
            break

    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()
