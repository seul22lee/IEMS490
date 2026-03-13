import os
import glob
import imageio.v2 as imageio
import numpy as np
from PIL import Image

FRAME_DIR = "/home/ftk3187/github/IEMS490/model-based-diffusion_pytorch_tide/results/png/frames_Nsample1024_lambda0.1_Ndiff50_pres"
OUTPUT_DIR = "/home/ftk3187/github/IEMS490/model-based-diffusion_pytorch_tide/results/png"

GIF_NAME = "mpc_prediction.gif"
MP4_NAME = "mpc_prediction.mp4"

FPS = 6

png_files = sorted(glob.glob(os.path.join(FRAME_DIR, "mpc_step_*.png")))

print("Found frames:", len(png_files))

if len(png_files) == 0:
    raise RuntimeError("No PNG files found")

# 기준 해상도 설정
first = Image.open(png_files[0])
width, height = first.size

# ffmpeg 안정성을 위해 16 배수로 맞춤
width = (width // 16) * 16
height = (height // 16) * 16

print("Video resolution:", width, height)

frames = []

for f in png_files:

    img = Image.open(f)

    # 모든 프레임 동일 크기
    img = img.resize((width, height))

    frames.append(np.array(img))

print("Frames resized")

# GIF 생성
gif_path = os.path.join(OUTPUT_DIR, GIF_NAME)

imageio.mimsave(
    gif_path,
    frames,
    fps=FPS
)

print("GIF saved:", gif_path)

# MP4 생성
mp4_path = os.path.join(OUTPUT_DIR, MP4_NAME)

writer = imageio.get_writer(
    mp4_path,
    fps=FPS,
    codec="libx264"
)

for frame in frames:
    writer.append_data(frame)

writer.close()

print("MP4 saved:", mp4_path)