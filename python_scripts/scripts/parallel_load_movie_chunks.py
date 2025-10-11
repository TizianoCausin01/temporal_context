import os, yaml, sys
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2

p = "/Users/tizianocausin/Desktop/bolt_moviechunk.mp4"
cap = cv2.VideoCapture(p)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(fps, n_frames)
# Seek to desired start frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 200)

ret, frame = cap.read()
if not ret:
    raise IOError("Failed to read frame from video.")
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
