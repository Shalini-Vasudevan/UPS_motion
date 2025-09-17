import os
import sys
import copy
import glob
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from model.MotionAGFormer import MotionAGFormer
from lib.utils import normalize_screen_coordinates, camera_to_world
from lib.preprocess import revise_kpts
import argparse
import cv2

def show2Dpose(kps, img):
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3
    for j, c in enumerate(connections):
        start = list(map(int, kps[c[0]]))
        end = list(map(int, kps[c[1]]))
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)
    return img

def show3Dpose(vals, ax):
    ax.view_init(elev=15., azim=70)
    lcolor = (0, 0, 1)
    rcolor = (1, 0, 0)
    I = np.array([0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])
    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)
    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color=lcolor if LR[i] else rcolor)
    RADIUS = 0.72
    RADIUS_Z = 0.7
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS_Z + zroot, RADIUS_Z + zroot])
    ax.set_aspect('auto')
    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)
    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)

def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)

def resample(n_frames):
    even = np.linspace(0, n_frames, num=243, endpoint=False)
    result = np.floor(even)
    result = np.clip(result, a_min=0, a_max=n_frames - 1).astype(np.uint32)
    return result

def turn_into_clips(keypoints):
    clips = []
    n_frames = keypoints.shape[1]
    if n_frames <= 243:
        new_indices = resample(n_frames)
        clips.append(keypoints[:, new_indices, ...])
        downsample = np.unique(new_indices, return_index=True)[1]
    else:
        for start_idx in range(0, n_frames, 243):
            keypoints_clip = keypoints[:, start_idx:start_idx + 243, ...]
            clip_length = keypoints_clip.shape[1]
            if clip_length != 243:
                new_indices = resample(clip_length)
                clips.append(keypoints_clip[:, new_indices, ...])
                downsample = np.unique(new_indices, return_index=True)[1]
            else:
                clips.append(keypoints_clip)
    return clips, downsample

def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]
    return flipped_data


@torch.no_grad()
def get_pose3D(video_path, output_dir, start_frame, end_frame):
    print("\nStarting 3D reconstruction for selected frames ...")
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.n_layers, args.dim_in, args.dim_feat, args.dim_rep, args.dim_out = 16, 3, 128, 512, 3
    args.mlp_ratio, args.act_layer = 4, nn.GELU
    args.attn_drop, args.drop, args.drop_path = 0.0, 0.0, 0.0
    args.use_layer_scale, args.layer_scale_init_value, args.use_adaptive_fusion = True, 0.00001, True
    args.num_heads, args.qkv_bias, args.qkv_scale = 8, False, None
    args.hierarchical = False
    args.use_temporal_similarity, args.neighbour_num, args.temporal_connection_len = True, 2, 1
    args.use_tcn, args.graph_only = False, False
    args.n_frames = 243
    args = vars(args)

    model = nn.DataParallel(MotionAGFormer(**args)).to('cpu')
    model_path = sorted(glob.glob(os.path.join('checkpoint', 'motionagformer-b-h36m.pth.tr')))[0]
    pre_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(pre_dict['model'], strict=True)
    model.eval()

    input_npz = os.path.join(output_dir, 'input_2D', 'keypoints.npz')
    if not os.path.exists(input_npz):
        print(f"Error: expected {input_npz} not found.")
        return

    keypoints = np.load(input_npz, allow_pickle=True)['reconstruction']  # shape (1, n_frames, 17, 3)
    clips, downsample = turn_into_clips(keypoints)
    cap = cv2.VideoCapture(video_path)

    # Generate 2D pose images (overlay)
    print('\nGenerating 2D pose images (overlay) for the selected frames...')
    output_dir_2D = output_dir + 'pose2D/'
    os.makedirs(output_dir_2D, exist_ok=True)
    frame_indices = list(range(start_frame, end_frame + 1))
    for offset, frame_idx in enumerate(tqdm(frame_indices, desc="2D overlays")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, img = cap.read()
        if not ret or img is None:
            continue
        input_2D = keypoints[0][offset][..., :2]  # x,y coords
        image = show2Dpose(input_2D, copy.deepcopy(img))
        cv2.imwrite(os.path.join(output_dir_2D, f"{frame_idx:04d}_2D.png"), image)

    # Generate 3D pose images
    print('\nGenerating 3D pose images (reconstruction)...')
    output_dir_3D = output_dir + 'pose3D/'
    os.makedirs(output_dir_3D, exist_ok=True)
    for idx, clip in enumerate(tqdm(clips, desc="3D reconstruction (clips)")):
        input_2D = normalize_screen_coordinates(clip, w= img.shape[1], h= img.shape[0])
        input_2D_aug = flip_data(input_2D)

        input_2D_t = torch.from_numpy(input_2D.astype('float32')).to('cpu')
        input_2D_aug_t = torch.from_numpy(input_2D_aug.astype('float32')).to('cpu')

        output_3D_non_flip = model(input_2D_t)
        output_3D_flip = flip_data(model(input_2D_aug_t))
        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        if idx == len(clips) - 1:
            output_3D = output_3D[:, downsample]
        output_3D[:, :, 0, :] = 0
        post_out_all = output_3D[0].cpu().detach().numpy()
        for j, post_out in enumerate(post_out_all):
            rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
            rot = np.array(rot, dtype='float32')
            post_out = camera_to_world(post_out, R=rot, t=0)
            post_out[:, 2] -= np.min(post_out[:, 2])
            max_value = np.max(post_out)
            if max_value != 0:
                post_out /= max_value
            fig = plt.figure(figsize=(9.6, 5.4))
            gs = gridspec.GridSpec(1, 1)
            gs.update(wspace=-0.00, hspace=0.05)
            ax = plt.subplot(gs[0], projection='3d')
            show3Dpose(post_out, ax)
            # absolute frame number calculation (matches 2D naming)
            abs_frame_idx = start_frame + (idx * 243 + j)
            save_path = os.path.join(output_dir_3D, f"{abs_frame_idx:04d}_3D.png")
            plt.savefig(save_path, dpi=200, format='png', bbox_inches='tight')
            plt.close(fig)

    print('Generating 3D pose successful!')
    # Compose demo side-by-side images
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))
    print('\nGenerating demo images combining 2D & 3D...')
    output_dir_pose = output_dir + 'pose/'
    os.makedirs(output_dir_pose, exist_ok=True)

    # pair them by sorted order - both use the same absolute frame numbers as filename prefix
    for i in tqdm(range(min(len(image_2d_dir), len(image_3d_dir))), desc="Composing demo images"):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])
        # Crop/pad similar to original code
        edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
        image_2d_cropped = image_2d[:, edge:image_2d.shape[1] - edge]
        edge2 = 130
        image_3d_cropped = image_3d[edge2:image_3d.shape[0] - edge2, edge2:image_3d.shape[1] - edge2]
        font_size = 12
        fig = plt.figure(figsize=(15.0, 5.4))
        ax = plt.subplot(121)
        showimage(ax, image_2d_cropped)
        ax.set_title("Input", fontsize=font_size)
        ax = plt.subplot(122)
        showimage(ax, image_3d_cropped)
        ax.set_title("Reconstruction", fontsize=font_size)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        # name using the same frame prefix as the 2D/3D files
        prefix = os.path.basename(image_2d_dir[i]).split('_')[0]
        plt.savefig(os.path.join(output_dir_pose, f"{prefix}_pose.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)

    print("Demo images generated.")