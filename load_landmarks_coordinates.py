# Load npy file that saves landmarks coordinates by frame.
# Process afterwards.
import numpy as np
import mediapipe as mp
import cv2
import argparse

parser = argparse.ArgumentParser(description='Load npy file that stores landmarks lists ')
parser.add_argument('--image_height', default=512, help='Image shape')
parser.add_argument('--image_width', default=512, help='Image shape')
parser.add_argument('--file', default='luoxiang_16min_shoulders.npy', help='To load which file')
args = parser.parse_args()

def normalized_landmark2pixel(
    landmark: list,
    height: int,
    width: int
):
    pixel_height = int(landmark[0]*height)
    pixel_width = int(landmark[1]*width)
    return pixel_height,pixel_width
    
file = args.file
# file = 'test_face_shoulders.npy'

image_height = args.image_height
image_width = args.image_width
coordinates_list = np.load(file, allow_pickle=True)
print('The shape of npy is'+ str(coordinates_list.shape))
# for coordinate in coordinates_list:
#     print(coordinate)

frame_count = 0

for frame in coordinates_list:
    canvas = np.zeros((image_height, image_width, 3), dtype = "uint8")  # Make all black canvas.
    for landmark in frame:
        print(landmark)
        pixel_height, pixel_width = normalized_landmark2pixel(landmark, image_height, image_width)
        cv2.circle(canvas, (pixel_height, pixel_width), 1,(255,255,255), 1)
    cv2.imshow("Conditional Feature Map",canvas)
    frame_count+=1
    print('Successfully recover frame ' + str(frame_count))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
