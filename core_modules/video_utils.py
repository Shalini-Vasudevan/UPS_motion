import os
import cv2
import glob

# Place img2video function here
def img2video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_name = os.path.basename(video_path).split('.')[0]
    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    if not names:
        print("Error: No pose images found to create video.")
        return
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])
    out_path = os.path.join(output_dir, video_name + '_pose_demo.mp4')
    videoWrite = cv2.VideoWriter(out_path, fourcc, fps, size)
    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)
    videoWrite.release()
    print(f"Saved demo video -> {out_path}")