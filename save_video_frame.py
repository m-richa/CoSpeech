import numpy as np
import os
import sys
import glob
from PIL import Image
import tqdm
import matplotlib.pyplot as plt
import cv2
import moviepy
import moviepy.editor
import torch
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from joblib import Parallel, delayed
import joblib

def save_frame(i, value):
    img = Image.fromarray(value).convert('RGB')
    img = img.resize((512,512), Image.BICUBIC)
    img.save(os.path.join(save_path, '%05d.png'%i))

# save_path = './data/full_body_2_long/origin'
save_path = './data/full_body_3/origin'
os.makedirs(save_path, exist_ok=True)
# clip = moviepy.editor.VideoFileClip("./data/full_body_2_long.mp4")
clip = moviepy.editor.VideoFileClip("./data/full_body_3.mp4")
print(int(clip.fps * clip.duration))
frames = clip.iter_frames()
counter = 0
joblib.Parallel(n_jobs=joblib.cpu_count())(joblib.delayed(save_frame)(i, value) for i, value in enumerate(frames))
