import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, VitPoseForPoseEstimation
from lib.preprocess import h36m_coco_format, revise_kpts

# Place get_2d_pose_for_person function here
def get_2d_pose_for_person_vitpose(video_path, output_dir, person_tracks, person_id, start_frame=None, end_frame=None, det_dim=416):
    """
    Runs ViTPose only on frames in [start_frame, end_frame] for person_id using stored bounding boxes.
    Saves keypoints.npz into output_dir/input_2D/keypoints.npz (same structure as before).
    Returns keypoints array and scores array (before h36m_coco_format).
    """
    print(f"Running ViTPose for person id {person_id} ...")
    try:
        processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
        model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
    except Exception as e:
        print(f"Error loading ViTPose model: {e}")
        return None, None

    if torch.cuda.is_available():
        model = model.cuda()

    # ensure the person exists
    if person_id not in person_tracks:
        # person_tracks likely has int keys; if passed as string try that
        if str(person_id) in person_tracks:
            person_id = str(person_id)
        else:
            print(f"Person id {person_id} not found in tracks.")
            return None, None

    # person_data
    person_data = person_tracks[person_id] if isinstance(person_id, str) else person_tracks[person_id]
    frames_list = person_data["frames"]
    bboxes_list = person_data["bboxes"]

    if start_frame is None:
        start_frame = min(frames_list)
    if end_frame is None:
        end_frame = max(frames_list)

    # build per-frame lookup for bounding box (frame_idx -> bbox)
    bbox_by_frame = {int(f): bbox for f, bbox in zip(frames_list, bboxes_list)}

    # number of frames we will process
    frame_indices = list(range(start_frame, end_frame + 1))
    n_frames = len(frame_indices)
    num_person = 1

    kpts_result = []
    scores_result = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video for pose extraction.")
        return None, None

    last_bbox = None
    with torch.no_grad():
        for ii, frame_idx in enumerate(tqdm(frame_indices, desc="ViTPose frames")):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                # append zero keypoints
                kpts_result.append(np.zeros((num_person, 17, 2), dtype=np.float32))
                scores_result.append(np.zeros((num_person, 17), dtype=np.float32))
                continue

            if frame_idx in bbox_by_frame:
                x1, y1, x2, y2 = bbox_by_frame[frame_idx]
                last_bbox = [x1, y1, x2, y2]
            else:
                # use last_known bbox if available, else skip (zeros)
                if last_bbox is None:
                    kpts_result.append(np.zeros((num_person, 17, 2), dtype=np.float32))
                    scores_result.append(np.zeros((num_person, 17), dtype=np.float32))
                    continue
                else:
                    x1, y1, x2, y2 = last_bbox

            width = x2 - x1
            height = y2 - y1
            person_boxes_coco = [[float(x1), float(y1), float(width), float(height)]]

            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            try:
                inputs = processor(images=image_pil, boxes=[person_boxes_coco], return_tensors="pt")
            except Exception as e:
                print(f"Processor error on frame {frame_idx}: {e}")
                kpts_result.append(np.zeros((num_person, 17, 2), dtype=np.float32))
                scores_result.append(np.zeros((num_person, 17), dtype=np.float32))
                continue

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            outputs = model(**inputs)
            pose_results = processor.post_process_pose_estimation(outputs, boxes=[person_boxes_coco])

            final_kpts = np.zeros((num_person, 17, 2), dtype=np.float32)
            final_scores = np.zeros((num_person, 17), dtype=np.float32)

            if pose_results and pose_results[0]:
                # pose_results[0] is a list of persons (here only 1 box so expect 1)
                try:
                    person_result = pose_results[0][0]
                    person_kpts = person_result["keypoints"].cpu().numpy()
                    person_scores = person_result["scores"].cpu().numpy()
                    final_kpts[0, :, :] = person_kpts
                    final_scores[0, :] = person_scores
                except Exception:
                    # fallback: zeros
                    pass

            kpts_result.append(final_kpts)
            scores_result.append(final_scores)

    cap.release()

    # kpts_result is list of length n_frames each entry shape (1,17,2)
    keypoints = np.array(kpts_result)  # shape (n_frames, 1, 17, 2)
    scores = np.array(scores_result)   # shape (n_frames, 1, 17)
    # reformat to (num_person, n_frames, 17, 2)
    keypoints = keypoints.transpose(1, 0, 2, 3)
    scores = scores.transpose(1, 0, 2)
    # convert person ids to int keys if necessary
    # Save in same style as original code (will be processed by h36m_coco_format next)
    print("ViTPose extraction complete. Now running h36m_coco_format and saving keypoints.npz ...")
    keypoints_proc, scores_proc, valid_frames = h36m_coco_format(keypoints, scores)
    # concatenate scores as last channel
    keypoints_with_scores = np.concatenate((keypoints_proc, scores_proc[..., None]), axis=-1)  # (1, n_frames, 17, 3)

    output_dir_full = os.path.join(output_dir, 'input_2D/')
    os.makedirs(output_dir_full, exist_ok=True)
    output_npz = os.path.join(output_dir_full, 'keypoints.npz')
    np.savez_compressed(output_npz, reconstruction=keypoints_with_scores)
    print(f"Saved 2D-keypoints .npz -> {output_npz}")
    return keypoints_with_scores, (start_frame, end_frame)