#!/usr/bin/env python3

import cv2
import csv
import random
from collections import defaultdict


def load_matches_from_csv(csv_path):
    """Load matching points from CSV file.

    Args:
        csv_path: Path to CSV file containing matches

    Returns:
        Dictionary mapping (frm1_id, frm2_id) -> list of dicts with keypoint and feature info
    """
    matches = defaultdict(list)

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frm1_id = int(row["frm1_id"])
            frm2_id = int(row["frm2_id"])
            frm1_u = float(row["frm1_u"])
            frm1_v = float(row["frm1_v"])
            frm1_octave = int(row["frm1_octave"])
            frm1_response = float(row["frm1_response"])
            frm2_u = float(row["frm2_u"])
            frm2_v = float(row["frm2_v"])
            frm2_octave = int(row["frm2_octave"])
            frm2_response = float(row["frm2_response"])
            is_outlier = str(row.get("is_outlier", "0")).strip() in {"1", "true", "True"}

            match_data = {
                "frm1_pt": (frm1_u, frm1_v),
                "frm1_octave": frm1_octave,
                "frm1_response": frm1_response,
                "frm2_pt": (frm2_u, frm2_v),
                "frm2_octave": frm2_octave,
                "frm2_response": frm2_response,
                "is_outlier": is_outlier,
            }
            matches[(frm1_id, frm2_id)].append(match_data)

    return matches


def get_frame_from_video(video_path, frame_id):
    """Get a specific frame from video by frame ID.

    Args:
        video_path: Path to video file
        frame_id: Frame index (0-based)

    Returns:
        Frame as numpy array, or None if frame not found
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()

    return frame if ret else None


def draw_matches_on_frames(
    frm1,
    frm2,
    matches_list,
    scale_by_response=False,
    response_scale=1.0,
    show_mode="inlier",
):
    """Draw matching points on concatenated frames.

    Args:
        frm1: First frame image
        frm2: Second frame image
        matches_list: List of match dicts with keypoint and feature info
        scale_by_response: Whether to scale circle radius by response value
        response_scale: Scaling factor for response-based radius adjustment
        show_mode: Which points to show: "inlier" or "outlier"

    Returns:
        Concatenated image with drawn matches
    """
    img_with_matches = cv2.vconcat([frm1, frm2])

    random.seed(0)

    for match_data in matches_list:
        is_outlier = match_data.get("is_outlier", False)
        if show_mode == "inlier" and is_outlier:
            continue
        if show_mode == "outlier" and not is_outlier:
            continue

        frm1_pt = match_data["frm1_pt"]
        frm2_pt = match_data["frm2_pt"]
        frm1_response = match_data["frm1_response"]
        frm2_response = match_data["frm2_response"]

        frm1_pt_int = (int(frm1_pt[0]), int(frm1_pt[1]))
        frm2_pt_int = (int(frm2_pt[0]), int(frm2_pt[1]) + frm1.shape[0])

        # Calculate radius based on response value
        if scale_by_response:
            radius1 = max(2, int(frm1_response * response_scale))
            radius2 = max(2, int(frm2_response * response_scale))
        else:
            radius1 = radius2 = 4

        cv2.circle(img_with_matches, frm1_pt_int, radius1, (255, 0, 0), 2)
        cv2.circle(img_with_matches, frm2_pt_int, radius2, (255, 0, 0), 2)

        b = random.randint(0, 255)
        g = random.randint(0, 255)
        r = random.randint(0, 255)
        line_color = (b, g, r)

        cv2.line(img_with_matches, frm1_pt_int, frm2_pt_int, line_color, 2)

    return img_with_matches


def load_local_map_points_from_csv(csv_path):
    """Load local-map tracking points from optimize_current_frame_with_local_map CSV."""
    points_by_frame = {}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"frm_id", "u", "v"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV must contain columns: {sorted(required)} (got: {reader.fieldnames})"
            )

        for row in reader:
            try:
                frm_id = int(row["frm_id"])
                u = float(row["u"])
                v = float(row["v"])
                is_outlier = str(row.get("is_outlier", "0")).strip() in {"1", "true", "True"}
            except (TypeError, ValueError, KeyError):
                continue

            points_by_frame.setdefault(frm_id, []).append((u, v, is_outlier))

    return points_by_frame


def load_mapping_points_from_csv(csv_path):
    """Load mapping keyframe points from mapping_with_new_keyframe CSV."""
    points_by_frame = {}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"frm_id", "u", "v"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV must contain columns: {sorted(required)} (got: {reader.fieldnames})"
            )

        for row in reader:
            try:
                frm_id = int(row["frm_id"])
                u = float(row["u"])
                v = float(row["v"])
                is_outlier = str(row.get("is_outlier", "0")).strip() in {"1", "true", "True"}
            except (TypeError, ValueError, KeyError):
                continue

            points_by_frame.setdefault(frm_id, []).append((u, v, is_outlier))

    return points_by_frame


def draw_local_map_points(img, points_by_frame, frm1_id, frm2_id, h1, show_mode="inlier"):
    """Draw local-map tracking points as green circles (radius 3)."""
    if points_by_frame is None:
        return
    
    for u, v, is_outlier in points_by_frame.get(frm1_id, []):
        if show_mode == "inlier" and is_outlier:
            continue
        if show_mode == "outlier" and not is_outlier:
            continue
        cv2.circle(img, (int(round(u)), int(round(v))), 3, (0, 255, 0), 1)
    
    for u, v, is_outlier in points_by_frame.get(frm2_id, []):
        if show_mode == "inlier" and is_outlier:
            continue
        if show_mode == "outlier" and not is_outlier:
            continue
        cv2.circle(img, (int(round(u)), int(round(v + h1))), 3, (0, 255, 0), 1)


def draw_mapping_points(img, points_by_frame, frm1_id, frm2_id, h1, show_mode="inlier"):
    """Draw mapping keyframe points as red circles (radius 4)."""
    if points_by_frame is None:
        return
    
    for u, v, is_outlier in points_by_frame.get(frm1_id, []):
        if show_mode == "inlier" and is_outlier:
            continue
        if show_mode == "outlier" and not is_outlier:
            continue
        cv2.circle(img, (int(round(u)), int(round(v))), 4, (0, 0, 255), 1)
    
    for u, v, is_outlier in points_by_frame.get(frm2_id, []):
        if show_mode == "inlier" and is_outlier:
            continue
        if show_mode == "outlier" and not is_outlier:
            continue
        cv2.circle(img, (int(round(u)), int(round(v + h1))), 4, (0, 0, 255), 1)
