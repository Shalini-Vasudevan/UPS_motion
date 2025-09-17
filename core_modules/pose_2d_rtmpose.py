import sys
import os
import cv2
import numpy as np
import json
from tqdm import tqdm
from ultralytics import YOLO

# Ensure the project root is in the path for proper module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.preprocess import h36m_coco_format

def get_2d_pose_for_person_rtmpose(video_path, output_dir, all_tracks_data, person_id, start_frame=None, end_frame=None):
    """
    Takes a dictionary of pre-extracted tracking data and performs 2D pose estimation
    for a single, specified person using YOLOv8-pose.
    """
    print("🚀 Using pre-extracted tracking data to get 2D pose for the selected person...")

    model = YOLO('yolov8l-pose.pt')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return None, (0, 0)
    
    # The fix is to ensure the person_id is an integer when looking up in the dict.
    target_track_int = int(person_id)
    if target_track_int not in all_tracks_data:
        print(f"Error: Person with ID {person_id} was not found in the video data.")
        return None, (0, 0)
    
    person_data = all_tracks_data[target_track_int]
    
    # The correct way to access the list of frames and bboxes is from the `person_data` dictionary.
    frames_list = person_data["frames"]
    bboxes_list = person_data["bboxes"]

    # Use these lists to determine the frame range
    if start_frame is None or start_frame < min(frames_list):
        start_frame_used = min(frames_list)
    else:
        start_frame_used = start_frame
    
    if end_frame is None or end_frame > max(frames_list):
        end_frame_used = max(frames_list)
    else:
        end_frame_used = end_frame
        
    print(f"Filtering data for ID {person_id} from frame {start_frame_used} to {end_frame_used}")

    # Build a lookup dictionary for efficient access
    bbox_lookup = {f: b for f, b in zip(frames_list, bboxes_list)}
    
    keypoints_list = []
    scores_list = []
    
    for f_num in tqdm(range(start_frame_used, end_frame_used + 1), desc="2D Pose Estimation"):
        bbox = bbox_lookup.get(f_num)
        
        if bbox is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_num - 1)
            success, frame = cap.read()
            if not success or frame is None:
                print(f"Warning: Could not read frame {f_num}, appending zeros.")
                keypoints_list.append([np.zeros((17, 2), dtype=np.float32).tolist()])
                scores_list.append([np.zeros((17), dtype=np.float32).tolist()])
                continue
            
            x1, y1, x2, y2 = bbox
            
            cropped_frame = frame[int(y1):int(y2), int(x1):int(x2)]
            
            results = model(cropped_frame, conf=0.25, verbose=False)
            
            if results and results[0].keypoints:
                keypoints = results[0].keypoints.xy.cpu().numpy()[0]
                scores = results[0].keypoints.conf[0].cpu().numpy()
            else:
                keypoints = np.zeros((17, 2), dtype=np.float32)
                scores = np.zeros(17, dtype=np.float32)

            keypoints[:, 0] += x1
            keypoints[:, 1] += y1
            
            keypoints_list.append([keypoints.tolist()])
            scores_list.append([scores.tolist()])
        
        else:
            keypoints_list.append([np.zeros((17, 2), dtype=np.float32).tolist()])
            scores_list.append([np.zeros((17), dtype=np.float32).tolist()])

    cap.release()
    print("✅ 2D pose estimation complete.")

    keypoints_np = np.array(keypoints_list)
    scores_np = np.array(scores_list)

    keypoints_proc, scores_proc, valid_frames = h36m_coco_format(keypoints_np.transpose(1, 0, 2, 3), scores_np.transpose(1, 0, 2))
    keypoints_with_scores = np.concatenate((keypoints_proc, scores_proc[..., None]), axis=-1)

    output_dir_full = os.path.join(output_dir, 'input_2D/')
    os.makedirs(output_dir_full, exist_ok=True)
    output_npz = os.path.join(output_dir_full, 'keypoints.npz')
    np.savez_compressed(output_npz, reconstruction=keypoints_with_scores)
    print(f"✅ Saved 2D-keypoints .npz -> {output_npz}")
    

    return keypoints_with_scores, (start_frame_used, end_frame_used)
