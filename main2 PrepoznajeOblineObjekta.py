import os
import random

import cv2
import numpy as np

from util import get_detections
###modifikovani kod tako da radi za video, daje output_video - motogp.avi. Poreklo koda je https://www.youtube.com/watch?v=FKmXzX0lsTM
# (1) define paths
cfg_path = './models/mask_rcnn_inception/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
weights_path = './models/mask_rcnn_inception/frozen_inference_graph.pb'
class_names_path = './models/mask_rcnn_inception/class.names'

# Input and output video paths
input_video_path = './lemansnimak.mp4'
output_video_path = './output_videoleman2.mp4'

# (2) Open the video file
cap = cv2.VideoCapture(input_video_path)

# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# (3) load model
net = cv2.dnn.readNetFromTensorflow(weights_path, cfg_path)

# (4) Create video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# (5) Process video frames
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

    # (9) visualization
    overlay = ((0.5 * empty_img) + (0.5 * frame)).astype("uint8")

    # (10) Write the frame to the output video
    out.write(overlay) ####overlay

    # Display the output video in a window
    cv2.imshow('Video', overlay)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# (11) Release the video capture and writer objects
cap.release()
out.release()

# (12) Close all OpenCV windows
cv2.destroyAllWindows()

"""import cv2
import numpy as np

# Load your contour mask and line mask
contour_mask = cv2.imread('contour_mask.png', cv2.IMREAD_GRAYSCALE)
line_mask = cv2.imread('line_mask.png', cv2.IMREAD_GRAYSCALE)

# Ensure the masks are binary (threshold if necessary)
_, contour_mask = cv2.threshold(contour_mask, 127, 255, cv2.THRESH_BINARY)
_, line_mask = cv2.threshold(line_mask, 127, 255, cv2.THRESH_BINARY)

# Find contours in the contour mask
contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through each contour
for contour in contours:
    # Create a mask for the current contour
    contour_mask_single = np.zeros_like(contour_mask)
    cv2.drawContours(contour_mask_single, [contour], 0, 255, thickness=cv2.FILLED)

    # Find the intersection of the contour mask and the line mask
    intersection_mask = cv2.bitwise_and(contour_mask_single, line_mask)

    # Display the result or perform further processing as needed
    cv2.imshow('Intersection Mask', intersection_mask)
    cv2.waitKey(0)

cv2.destroyAllWindows()
"""