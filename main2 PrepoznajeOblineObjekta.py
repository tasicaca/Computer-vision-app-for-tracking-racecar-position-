import os
import random

import cv2
import numpy as np

from util import get_detections

# define paths
cfg_path = './models/mask_rcnn_inception/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
weights_path = './models/mask_rcnn_inception/frozen_inference_graph.pb'
class_names_path = './models/mask_rcnn_inception/class.names'

# Input and output video paths
input_video_path = './lemansnimak.mp4'
output_video_path = './output_videoleman2.mp4'

# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# load model
net = cv2.dnn.readNetFromTensorflow(weights_path, cfg_path)

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    H, W, C = frame.shape

    # (6) convert image
    blob = cv2.dnn.blobFromImage(frame)

    # (7) get masks
    boxes, masks = get_detections(net, blob)

    # (8) draw masks
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(90)]

    empty_img = np.zeros((H, W, C))

    detection_th = 0.6
    for j in range(len(masks)):
        bbox = boxes[0, 0, j]

        class_id = bbox[1]
        score = bbox[2]

        if score > detection_th:
            mask = masks[j]

            x1, y1, x2, y2 = int(bbox[3] * W), int(bbox[4] * H), int(bbox[5] * W), int(bbox[6] * H)

            mask = mask[int(class_id)]

            mask = cv2.resize(mask, (x2 - x1, y2 - y1))

            _, mask = cv2.threshold(mask, 0.3, 1, cv2.THRESH_BINARY)

            for c in range(3):
                empty_img[y1:y2, x1:x2, c] = mask * colors[int(class_id)][c]

    # visualization
    overlay = ((0.5 * empty_img) + (0.5 * frame)).astype("uint8")

    # Write the frame to the output video
    out.write(overlay) ####overlay

    # Display the output video in a window
    cv2.imshow('Video', overlay)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()


