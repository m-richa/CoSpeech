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

import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

def save_keypoints(file):
    with mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
        name = file.rstrip('\n').split('/')[-1]
        image = Image.open(file).convert('RGB')
        image = np.array(image)
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
        # Print nose landmark.
        image_hight, image_width, _ = image.shape
        if not results.pose_landmarks:
            return

        # Draw pose landmarks.
        # print(f'Pose landmarks of {name}:')
        annotated_image = np.zeros_like(image).copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # resize_and_show(annotated_image)
        # plt.imshow(annotated_image)
        # plt.show()
        annotated_image = Image.fromarray(annotated_image).convert('L')
        annotated_image.save(os.path.join(save_path, name))

path = './data/full_body_3/origin'
files = glob.glob(path+'/*.png')
files.sort()
save_path = './data/full_body_3/body_mesh'
os.makedirs(save_path, exist_ok=True)

joblib.Parallel(n_jobs=joblib.cpu_count())(joblib.delayed(save_keypoints)(file) for file in files)