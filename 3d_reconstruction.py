#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
import torch
import argparse
import glob
import numpy as np
from mast3r.model import AsymmetricMASt3R
from dust3r.demo import set_print_with_timestamp
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
import matplotlib.pyplot as pl

pl.ion()
torch.backends.cuda.matmul.allow_tf32 = True

def get_args_parser():
    parser = argparse.ArgumentParser('MASt3R Dataset Point Cloud Generator', add_help=False)
    parser.add_argument('--model_name', default='MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric', type=str)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--dataset_path', required=True, type=str)
    parser.add_argument('--image_extensions', default=['jpg', 'jpeg', 'png', 'bmp', 'tiff'], nargs='+')
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--scene_name', default='scene', type=str)
    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--max_images', default=None, type=int)
    parser.add_argument('--confidence_threshold', default=1.5, type=float)
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--max_pairs', default=100, type=int)
    parser.add_argument('--save_ply', action='store_true')
    parser.add_argument('--tmp_dir', default=None, type=str)
    return parser

def load_dataset_images(dataset_path, extensions, max_images=None, stride=1):
    image_paths = []
    pattern = os.path.join(dataset_path, "frame-*.color.png")
    image_paths = sorted(glob.glob(pattern))

    if stride > 1:
        image_paths = image_paths[::stride]
        print(f"Using stride {stride}, selected {len(image_paths)} images")
    if max_images is not None:
        image_paths = image_paths[:max_images]
    print(f"Found {len(image_paths)} color images in dataset")
    return image_paths

def extract_scalar_idx(idx):
    idx_np = to_numpy(idx)
    # If array, flatten first
    if isinstance(idx_np, np.ndarray):
        idx_np = idx_np.flatten()
        return int(idx_np[0])
    # If list or tuple, get the first element
    if isinstance(idx_np, (list, tuple)):
        return int(idx_np[0])
    return int(idx_np)

def process_dataset_to_pointcloud(model, image_paths, device, image_size, confidence_threshold):
    print("Loading and preprocessing images...")
    images = load_images(image_paths, size=image_size, verbose=True)
    print(f"Running inference on {len(images)} images...")

    pairs = []
    for i in range(len(images) - 1):
        pairs.append((images[i], images[i + 1]))
    for i in range(0, len(images) - 2, 2):
        pairs.append((images[i], images[i + 2]))
    for i in range(0, len(images) - 5, 5):
        pairs.append((images[i], images[i + 5]))
    if len(images) > 2:
        pairs.append((images[0], images[-1]))

    print(f"Processing {len(pairs)} pairs (strategic subset)")
    if len(pairs) == 0:
        raise ValueError("Need at least 2 images for reconstruction")

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    output = inference(pairs, model, device, batch_size=1, verbose=True)
    print("Output keys:", output.keys())

    all_pts3d = []
    all_colors = []
    all_confidences = []

    batch_size = output['view1']['img'].shape[0]
    for i in range(batch_size):
        view1 = {k: v[i] if isinstance(v, (np.ndarray, torch.Tensor)) and v.shape[0] == batch_size else v for k, v in output['view1'].items()}
        pred1 = {k: v[i] if isinstance(v, (np.ndarray, torch.Tensor)) and v.shape[0] == batch_size else v for k, v in output['pred1'].items()}
        view2 = {k: v[i] if isinstance(v, (np.ndarray, torch.Tensor)) and v.shape[0] == batch_size else v for k, v in output['view2'].items()}
        pred2 = {k: v[i] if isinstance(v, (np.ndarray, torch.Tensor)) and v.shape[0] == batch_size else v for k, v in output['pred2'].items()}

        idx1 = extract_scalar_idx(view1['idx']) if 'idx' in view1 else None
        idx2 = extract_scalar_idx(view2['idx']) if 'idx' in view2 else None
        impath1 = image_paths[idx1] if idx1 is not None else None
        impath2 = image_paths[idx2] if idx2 is not None else None
        print("pred2.kesy:",pred2.keys())
        pts3d1 = pred1['pts3d']
        conf1 = pred1['conf']
        pts3d2 = pred2['pts3d_in_other_view']
        conf2 = pred2['conf']

        pts3d1_np = to_numpy(pts3d1)
        conf1_np = to_numpy(conf1)
        pts3d2_np = to_numpy(pts3d2)
        conf2_np = to_numpy(conf2)

        img1 = pl.imread(impath1) if impath1 else None
        img2 = pl.imread(impath2) if impath2 else None
        if img1 is not None and img1.max() > 1:
            img1 = img1 / 255.0
        if img2 is not None and img2.max() > 1:
            img2 = img2 / 255.0

        # View 1
    pts3d1_flat = pts3d1_np.reshape(-1, 3)
    conf1_flat = conf1_np.reshape(-1)
    valid_mask1 = conf1_flat > confidence_threshold
    if np.sum(valid_mask1) > 0:
        dummy_color = np.ones((np.sum(valid_mask1), 3), dtype=np.uint8) * 128  # gray
        all_pts3d.append(pts3d1_flat[valid_mask1])
        all_colors.append(dummy_color)
        all_confidences.append(conf1_flat[valid_mask1])

    # View 2
    pts3d2_flat = pts3d2_np.reshape(-1, 3)
    conf2_flat = conf2_np.reshape(-1)
    valid_mask2 = conf2_flat > confidence_threshold
    if np.sum(valid_mask2) > 0:
        dummy_color = np.ones((np.sum(valid_mask2), 3), dtype=np.uint8) * 128  # gray
        all_pts3d.append(pts3d2_flat[valid_mask2])
        all_colors.append(dummy_color)
        all_confidences.append(conf2_flat[valid_mask2])

        if len(all_pts3d) == 0:
            raise ValueError(f"No points passed confidence threshold of {confidence_threshold}")

    combined_pts3d = np.vstack(all_pts3d)
    combined_colors = np.vstack(all_colors)
    combined_confidences = np.concatenate(all_confidences)

    print(f"Generated {len(combined_pts3d)} 3D points")
    return combined_pts3d, combined_colors, combined_confidences

def save_point_cloud_ply(points, colors, confidences, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)
    with open(output_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property float confidence\nend_header\n")
        for i in range(len(points)):
            f.write(f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f} ")
            f.write(f"{colors[i,0]} {colors[i,1]} {colors[i,2]} ")
            f.write(f"{confidences[i]:.6f}\n")
    print(f"Saved point cloud to {output_path}")

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    set_print_with_timestamp()
    if not os.path.exists(args.dataset_path):
        raise ValueError(f"Dataset path does not exist: {args.dataset_path}")
    print("Loading MASt3R model...")
    weights_path = args.weights if args.weights is not None else "naver/" + args.model_name
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)
    print(f"Model loaded on {args.device}")
    image_paths = load_dataset_images(args.dataset_path, args.image_extensions, args.max_images, args.stride)
    if len(image_paths) < 2:
        raise ValueError("Need at least 2 images for 3D reconstruction")
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        points, colors, confidences = process_dataset_to_pointcloud(
            model, image_paths, args.device, args.image_size, args.confidence_threshold
        )
        if args.save_ply:
            ply_path = os.path.join(args.output_dir, f"{args.scene_name}.ply")
            save_point_cloud_ply(points, colors, confidences, ply_path)
        print(f"\nProcessing complete!")
        print(f"Generated {len(points)} 3D points")
        print(f"Output saved to: {args.output_dir}")
    except Exception as e:
        print(f"Error during processing: {e}")
        raise

if __name__ == '__main__':
    main()