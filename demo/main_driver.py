import sys
import shutil
import argparse
import os

# Import modules from the core_modules directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core_modules'))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the project root to the Python path
sys.path.insert(0, project_root)
from detection_tracking import generate_person_tracks, save_tracking_video
from pose_2d_vitpose import get_2d_pose_for_person_vitpose
from pose_2d_rtmpose import get_2d_pose_for_person_rtmpose
from pose_3d import get_pose3D
from video_utils import img2video

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='sample_video.mp4', help='input video filename inside ./demo/video/')
    parser.add_argument('--gpu', type=str, default='0', help='CUDA visible device(s)')
    parser.add_argument('--yolow', type=str, default='yolov8n.pt', help='YOLO weights to use (default yolov8n.pt)')
    parser.add_argument('--conf', type=float, default=0.5, help='detection confidence threshold')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    video_path = './demo/video/' + args.video
    video_name = os.path.basename(video_path).split('.')[0]
    output_dir = './demo/output/' + video_name + '/'

    if os.path.exists(output_dir):
        print(f"Clearing contents of directory: {output_dir}")
        # Remove all files and subdirectories
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory and its contents
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    # step 1: generate tracks for entire video and save JSON
    try:
        tracks_dict, persons_summary, json_path = generate_person_tracks(video_path, output_dir, yolo_weights=args.yolow, conf_thresh=args.conf)
    except Exception as e:
        print(f"Detection/tracking failed: {e}")
        sys.exit(1)

    if len(persons_summary) == 0:
        print("No persons detected in video. Exiting.")
        sys.exit(0)

    # Show summary and prompt user to pick a person id
    print("\nDetected persons summary:")
    for p in persons_summary:
        print(f"  ID {p['id']}: frames {p['start_frame']} -> {p['end_frame']} ({p['num_frames']} detections)")

    # Save intermediate tracking video with bounding boxes + IDs
    save_tracking_video(video_path, output_dir, tracks_dict)


    # Simple input loop (allows user to choose an id)
    chosen_id = None
    while True:
        user_in = input("\nEnter the person ID you want to run 2D/3D pose for (or 'q' to quit): ").strip()
        if user_in.lower() == 'q':
            print("Quitting.")
            sys.exit(0)
        try:
            parsed = int(user_in)
        except Exception:
            print("Please enter a valid integer person id.")
            continue
        if parsed in tracks_dict or str(parsed) in tracks_dict:
            chosen_id = parsed
            break
        else:
            print(f"Person id {parsed} not found. Try again.")

    # Determine start/end frames from tracks
    chosen_key = chosen_id if chosen_id in tracks_dict else str(chosen_id)
    chosen_data = tracks_dict[chosen_key] if isinstance(chosen_key, str) and chosen_key in tracks_dict else tracks_dict[chosen_key]
    default_start = min(chosen_data["frames"])
    default_end = max(chosen_data["frames"])
    print(f"Person {chosen_id} detected from frame {default_start} to {default_end}.")

    s, e = default_start, default_end

    # Run ViTPose for selected person & frames (s..e)
    input_model = input("Enter 1 to use ViTPose, 2 to use RTMPose (YOLOv8-pose), or 'q' to quit: ").strip()
    if input_model == '1':
        keypoints_np, (s_used, e_used) = get_2d_pose_for_person_vitpose(video_path, output_dir, tracks_dict, chosen_key, start_frame=s, end_frame=e)
    elif input_model == '2':
        keypoints_np, (s_used, e_used) = get_2d_pose_for_person_rtmpose(video_path, output_dir, tracks_dict, chosen_key, start_frame=s, end_frame=e)
    elif input_model.lower() == 'q':
        print("Quitting.")
        sys.exit(0)
    else:
        print("Invalid input. Please enter 1, 2, or 'q'.")
        sys.exit(1)
    if keypoints_np is None:
        print("2D pose extraction failed. Exiting.")
        sys.exit(1)

    # Run 3D reconstruction for selected frames
    get_pose3D(video_path, output_dir, s_used, e_used)

    # Compose video from poses
    img2video(video_path, output_dir)
    print("All done. Outputs are in:", output_dir)
