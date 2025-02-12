import cv2
import os
import numpy as np


def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{output_folder}/frame_{frame_count:04d}.jpg", frame)
        frame_count += 1
    cap.release()
    print(f"Extracted {frame_count} frames.")


def compute_sift_features(frame_path):
    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(frame, None)
    return keypoints, descriptors


def visualize_optical_flow(flow, frame):
    h, w = frame.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_vis


def create_optical_flow_video(frame_folder, output_path):
    frames = sorted(os.listdir(frame_folder))
    out = None
    for i in range(1, len(frames)):
        prev_frame = cv2.imread(f"{frame_folder}/{frames[i-1]}")
        next_frame = cv2.imread(f"{frame_folder}/{frames[i]}")
        flow = compute_optical_flow(prev_frame, next_frame)
        flow_vis = visualize_optical_flow(flow, prev_frame)
        combined = cv2.addWeighted(next_frame, 0.5, flow_vis, 0.5, 0)
        if out is None:
            h, w, _ = combined.shape
            out = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
        out.write(combined)
    out.release()


def compute_optical_flow(prev_frame, next_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow


if __name__ == "__main__":
    # Path to your video
    video_path = "C:\\Users\\mites\\OneDrive\\Desktop\\CV_Assignment 2 (Pics)\\PXL_20241215_162247099.mp4"
    frame_folder = "C:\\Users\\mites\\OneDrive\\Desktop\\CV_Assignment 2 (Pics)\\frames"
    output_video_path = "C:\\Users\\mites\\OneDrive\\Desktop\\CV_Assignment 2 (Pics)\\optical_flow_dance.mp4"

    # Step 1: Extract Frames
    extract_frames(video_path, frame_folder)

    # Step 2-5: Compute and Generate Optical Flow Video
    create_optical_flow_video(frame_folder, output_video_path)
    print("Optical flow dance video created!")
