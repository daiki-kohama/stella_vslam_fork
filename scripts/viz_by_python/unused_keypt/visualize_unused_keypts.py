#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import msgpack
import numpy as np


def load_map_database(path: Path) -> Dict[str, Any]:
    data = path.read_bytes()

    # Try MessagePack first (stella_vslam default)
    try:
        return msgpack.unpackb(data, raw=False)
    except Exception:
        pass

    # Fallback to JSON text
    try:
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to parse map database: {path}") from e


def sorted_keyframes(map_db: Dict[str, Any]) -> List[Tuple[int, Dict[str, Any]]]:
    keyframes = map_db.get("keyframes", {})
    pairs: List[Tuple[int, Dict[str, Any]]] = []

    if isinstance(keyframes, dict):
        for k, v in keyframes.items():
            try:
                pairs.append((int(k), v))
            except Exception:
                continue
    elif isinstance(keyframes, list):
        for i, v in enumerate(keyframes):
            if isinstance(v, dict):
                pairs.append((int(v.get("id", i)), v))

    pairs.sort(key=lambda x: x[0])
    return pairs


def get_point_from_keypoint_json(kp: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    pt = kp.get("pt")
    if not isinstance(pt, Sequence) or len(pt) < 2:
        return None

    try:
        x = int(round(float(pt[0])))
        y = int(round(float(pt[1])))
    except Exception:
        return None

    return x, y


def build_blank_canvas(keypts: List[Dict[str, Any]], min_w: int = 640, min_h: int = 480) -> np.ndarray:
    max_x = 0
    max_y = 0

    for kp in keypts:
        pt = get_point_from_keypoint_json(kp)
        if pt is None:
            continue
        max_x = max(max_x, pt[0])
        max_y = max(max_y, pt[1])

    w = max(min_w, max_x + 32)
    h = max(min_h, max_y + 32)
    return np.zeros((h, w, 3), dtype=np.uint8)


def read_video_frame(cap: cv2.VideoCapture, frame_id: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def draw_keypoints(
    base_img: np.ndarray,
    keypts: List[Dict[str, Any]],
    unused_indices: set,
) -> np.ndarray:
    vis = base_img.copy()

    for idx, kp in enumerate(keypts):
        pt = get_point_from_keypoint_json(kp)
        if pt is None:
            continue

        color = (0, 0, 255) if idx in unused_indices else (0, 255, 0)
        cv2.circle(vis, pt, 2, color, 1, cv2.LINE_AA)

    return vis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize keyframe unused_keypt_indices from stella_vslam map database"
    )
    parser.add_argument("--map-db", required=True, help="Path to map database (.msg/.msgpack or .json)")
    parser.add_argument("--video", default="", help="Optional source video path to draw on actual frames")
    parser.add_argument("--start", type=int, default=0, help="Start keyframe index in sorted keyframe list")
    args = parser.parse_args()

    map_path = Path(args.map_db)
    if not map_path.exists():
        raise FileNotFoundError(f"Map DB not found: {map_path}")

    map_db = load_map_database(map_path)
    keyframes = sorted_keyframes(map_db)
    if not keyframes:
        raise RuntimeError("No keyframes found in map database")

    cap: Optional[cv2.VideoCapture] = None
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {args.video}")

    idx = max(0, min(args.start, len(keyframes) - 1))
    win = "unused keypoint viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        keyfrm_id, kf = keyframes[idx]
        src_frm_id = int(kf.get("src_frm_id", -1))
        keypts = kf.get("undist_keypts", [])
        if not isinstance(keypts, list):
            keypts = []

        unused = set(int(v) for v in kf.get("unused_keypt_indices", []) if isinstance(v, (int, float)))

        if cap is not None and src_frm_id >= 0:
            base = read_video_frame(cap, src_frm_id)
            if base is None:
                base = build_blank_canvas(keypts)
        else:
            base = build_blank_canvas(keypts)

        vis = draw_keypoints(base, keypts, unused)

        overlay = (
            f"kf_id={keyfrm_id} ({idx + 1}/{len(keyframes)})  "
            f"src_frm_id={src_frm_id}  keypts={len(keypts)}  unused={len(unused)}"
        )
        cv2.putText(vis, overlay, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(vis, overlay, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 1, cv2.LINE_AA)

        help_text = "keys: n/Right=next, b/Left=prev, q or ESC=quit"
        cv2.putText(vis, help_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, help_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)

        cv2.imshow(win, vis)
        key = cv2.waitKey(0)

        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("n"), ord("N"), 83):  # Right arrow (OpenCV on many backends)
            idx = min(idx + 1, len(keyframes) - 1)
            continue
        if key in (ord("b"), ord("B"), 81):  # Left arrow
            idx = max(idx - 1, 0)
            continue

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
