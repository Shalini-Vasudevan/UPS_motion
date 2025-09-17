import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from lib.sort.sort import Sort

# Place generate_person_tracks and save_tracking_video functions here
def generate_person_tracks(video_path, output_dir, yolo_weights="yolov8n.pt", conf_thresh=0.5):
    """
    Runs YOLO + SORT over entire video and records bounding boxes per person (track id).
    Saves a JSON file with per-person start/end frame and per-frame bboxes.
    Returns: tracks dict {track_id: {"frames":[...], "bboxes":[[x1,y1,x2,y2], ...]}}
    """
    print("Running detection + SORT to collect tracks across the whole video...")
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Initialize detector and tracker
    human_model = YOLO(yolo_weights)
    people_sort = Sort(min_hits=0)

    tracks = {}  # track_id(int) -> {"frames": [...], "bboxes": [[x1,y1,x2,y2], ...], "scores": [...]}

    for ii in tqdm(range(video_length), desc="Detecting frames"):
        ret, frame = cap.read()
        if not ret:
            break

        yolo_results = human_model(frame, verbose=False, conf=conf_thresh)

        bboxs = []
        scores = []
        # ultralytics returns a Results object per image; processing similarly to your earlier snippet
        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                try:
                    # box.xyxy is usually tensor of shape (1,4)
                    xy = box.xyxy[0].tolist()
                    cls = int(box.cls[0].item()) if hasattr(box.cls, '__len__') else int(box.cls)
                    conf = float(box.conf[0].item()) if hasattr(box.conf, '__len__') else float(box.conf)
                except Exception:
                    # fallback: try direct attributes
                    xy = box.xyxy.tolist() if hasattr(box.xyxy, 'tolist') else box.xyxy
                    cls = int(box.cls)
                    conf = float(box.conf)
                if cls == 0:  # person class
                    x1, y1, x2, y2 = map(float, xy)
                    bboxs.append([x1, y1, x2, y2])
                    scores.append(conf)

        if bboxs:
            bboxs_arr = np.array(bboxs)
            scores_arr = np.array(scores)
            dets = np.hstack((bboxs_arr, scores_arr[:, None]))  # Nx5
            people_track = people_sort.update(dets)
        else:
            # pass empty detections (SORT should handle this)
            people_track = people_sort.update(np.empty((0, 5)))

        # people_track is an array of tracks with columns [x1,y1,x2,y2,track_id]
        if people_track is not None and len(people_track) > 0:
            for tr in people_track:
                x1, y1, x2, y2, tid = tr.tolist()
                tid = int(tid)
                if tid not in tracks:
                    tracks[tid] = {"frames": [], "bboxes": [], "scores": []}
                tracks[tid]["frames"].append(int(ii))
                tracks[tid]["bboxes"].append([float(x1), float(y1), float(x2), float(y2)])
                # attempt to find matching detection score (best-effort) - use mean score if not available
                tracks[tid]["scores"].append(float(np.mean(scores) if len(scores) > 0 else 0.0))
        # else: no tracks this frame

    cap.release()

    # Build summary list
    persons_summary = []
    for tid, data in tracks.items():
        start = int(min(data["frames"]))
        end = int(max(data["frames"]))
        persons_summary.append({
            "id": tid,
            "start_frame": start,
            "end_frame": end,
            "num_frames": len(data["frames"])
        })

    out_json = {
        "video": os.path.basename(video_path),
        "num_persons": len(persons_summary),
        "persons": persons_summary,
        "tracks": {str(k): v for k, v in tracks.items()}  # convert keys to strings for JSON
    }

    json_path = os.path.join(output_dir, "person_tracks.json")
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"Saved person tracks JSON -> {json_path}")
    return tracks, persons_summary, json_path
    
def save_tracking_video(video_path, output_dir, tracks):
    """
    Creates an intermediate video with bounding boxes and person IDs overlaid,
    using the detection+tracking results stored in 'tracks'.
    """
    print("Generating tracking video with bounding boxes and person IDs...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: cannot open video for tracking overlay.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_name = os.path.basename(video_path).split('.')[0]
    out_path = os.path.join(output_dir, video_name + "_tracking.mp4")

    # get frame size
    ret, frame = cap.read()
    if not ret:
        print("Error: could not read video.")
        return
    height, width = frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset to first frame
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # build lookup dictionary frame -> {id: bbox}
    frame_to_tracks = {}
    for pid, data in tracks.items():
        for f, bbox in zip(data["frames"], data["bboxes"]):
            if f not in frame_to_tracks:
                frame_to_tracks[f] = []
            frame_to_tracks[f].append((pid, bbox))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in frame_to_tracks:
            for pid, bbox in frame_to_tracks[frame_idx]:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {pid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Tracking video saved -> {out_path}")
