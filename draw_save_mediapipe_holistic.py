import numpy as np
import os
import sys
import glob
from PIL import Image
import tqdm
import matplotlib.pyplot as plt
import cv2
import torch
import argparse
from tqdm import tqdm 
import joblib
from joblib import Parallel, delayed
from skimage.io import imread, imsave
from skimage.color import rgb2gray
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# PAII_VB_ROOT=os.environ["PAII_VB_ROOT"]
# sys.path.append(PAII_VB_ROOT)

from mediapipe_connections import BODY_POSE_NO_HAND_CONNECTIONS, \
    FACEMESH_CONDITIONAL_FEATURE_MAP_CONNECTIONS

# Create an Pose object.
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic

mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles


def save_landmarks(landmark_list: landmark_pb2.NormalizedLandmarkList):
    record = np.empty([0, 3], dtype=np.float32)
    for idx, landmark in enumerate(landmark_list.landmark):
        tmp = [[landmark.x, landmark.y, landmark.z]]
        record = np.append(record, tmp, axis=0)
    return record
    

def main(args):
    input_path = args.input_folder
    save_path = args.output_folder
    if not os.path.exists(save_path): # If output folder not exist, create one
        os.makedirs(save_path)
    filenames = [f for f in os.listdir(input_path) if f.endswith('.png')]
    filenames = sorted(filenames, key=lambda x: int(os.path.splitext(x)[0]))
    face_landmarks = []
    body_landmarks = []
    left_hand_landmarks = []
    right_hand_landmarks = []
    

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        for img_fn in tqdm(filenames):
            # make all black canvas
            image = cv2.imread(os.path.join(input_path,img_fn))

            # image.flags.writeable = False
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # results = holistic.process(image)
            image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 获取图像尺寸
            height, width, _ = image.shape

            # 计算填充的高度
            target_height = int(height * 1.5)

            # 创建新的图像
            extended_image = np.zeros((target_height, width, 3), dtype=np.uint8)
            # print('Origin extended image:', extended_image.shape)

            # 将半身图像复制到新图像的顶部
            extended_image[:height, :] = image
            # print('Add image extended image:', extended_image.shape)
            
            canvas = np.zeros((extended_image.shape[0], extended_image.shape[1], 3), dtype=np.uint8)
            # print('Canvas shape, should be same as extend image:', extended_image.shape)
            # print(canvas.shape)
            # break
            canvas.flags.writeable = True

            # 转换为 RGB
            frame_rgb = cv2.cvtColor(extended_image, cv2.COLOR_BGR2RGB)
            results = results = holistic.process(frame_rgb)

            # DRAW FACE
            mp_drawing.draw_landmarks(
                canvas,
                results.face_landmarks,
                FACEMESH_CONDITIONAL_FEATURE_MAP_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
            )
            # DRAW POSE
            mp_drawing.draw_landmarks(
                canvas,
                results.pose_landmarks,
                BODY_POSE_NO_HAND_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
            )
            # DRAW LEFTHAND
            mp_drawing.draw_landmarks(
                canvas,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
        )
            # DRAW RIGHTHAND
            mp_drawing.draw_landmarks(
                canvas,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
        )
            # print(canvas.shape)
            canvas = canvas[:512, :]
            
            cv2.imwrite(os.path.join(save_path, img_fn), canvas)
            # print('After math canvas shape', canvas.shape)
            
    #         # save landmarks
    #         cur_face_landmarks = save_landmarks(landmark_list=results.face_landmarks)
    #         cur_body_landmarks = save_landmarks(landmark_list=results.pose_landmarks)

    #         # save missing had landmarks to -1
    #         if results.left_hand_landmarks is None:
    #             cur_left_hand_landmarks = np.ones([21,3]) * -1
    #         else:
    #             cur_left_hand_landmarks = save_landmarks(landmark_list=results.left_hand_landmarks)
    #         if results.right_hand_landmarks is None:
    #             cur_right_hand_landmarks = np.ones([21,3]) * -1
    #         else:
    #             cur_right_hand_landmarks = save_landmarks(landmark_list=results.right_hand_landmarks)
    #         face_landmarks.append(cur_face_landmarks)
    #         body_landmarks.append(cur_body_landmarks)
    #         left_hand_landmarks.append(cur_left_hand_landmarks)
    #         right_hand_landmarks.append(cur_right_hand_landmarks)

    # np.save(os.path.join(os.path.dirname(args.output_folder), "face_landmarks.npy"), np.array([face_landmarks]).reshape([-1, 468, 3]))
    # np.save(os.path.join(os.path.dirname(args.output_folder), "body_landmarks.npy"), np.array([body_landmarks]).reshape([-1, 33, 3]))
    # np.save(os.path.join(os.path.dirname(args.output_folder), "left_landmarks.npy"), np.array([left_hand_landmarks]).reshape([-1, 21, 3]))
    # np.save(os.path.join(os.path.dirname(args.output_folder), "right_landmarks.npy"), np.array([right_hand_landmarks]).reshape([-1, 21, 3]))
    # print("process finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To extract body and hands feature maps ')
    # parser.add_argument('--input_folder', default='/NAS0/speech/data/virtual_being/C0209_cropped/origin_square/', help='To load images from with folder')
    parser.add_argument('--input_folder', default='/NAS0/speech/data/virtual_being/talkshow/three_characters/matted/video_frames512/chemistry/matted', help='To load images from with folder')
    parser.add_argument('--output_folder', default='/NAS0/speech/data/virtual_being/talkshow/chemistry_feature', help='To store in which folder, will create in same path as input folder if not exist.')
    args = parser.parse_args()
    main(args)
