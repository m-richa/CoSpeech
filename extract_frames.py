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

import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

# save_path = './data/full_body_2/origin'
# os.makedirs(save_path, exist_ok=True)
# clip = moviepy.editor.VideoFileClip("./data/full_body_2.mp4")
# print(int(clip.fps * clip.duration))
# frames = clip.iter_frames()
# counter = 0
# # using loop to transverse the frames
# for value in tqdm.tqdm(frames):
#     # incrementing the counter
#     img = Image.fromarray(value).convert('RGB')
#     img.save(os.path.join(save_path, '%05d.png'%counter))
#     counter += 1
#     # break
     
# # printing the value of the counter
# print("Counter Value ", end = " : ")
# print(counter)


path = './data/full_body_2/origin'
files = glob.glob(path+'/*.png')
files.sort()
save_path = './data/full_body_2/body_mesh'
os.makedirs(save_path, exist_ok=True)

with mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
  
  for i in tqdm.tqdm(range(len(files))):
    name = files[i].rstrip('\n').split('/')[-1]
    image = Image.open(files[i]).convert('RGB')
    image = np.array(image)
    # Convert the BGR image to RGB and process it with MediaPipe Pose.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Print nose landmark.
    image_hight, image_width, _ = image.shape
    if not results.pose_landmarks:
        print(name)
        continue
    # print(
    #   f'Nose coordinates: ('
    #   f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
    #   f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_hight})'
    # )

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