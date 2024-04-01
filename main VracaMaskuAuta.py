#vraca maske vozila, a kod za konture je nastao odavde
import cv2
import numpy as np
from util import get_detections
import random
# Define paths
cfg_path = './models/mask_rcnn_inception/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
weights_path = './models/mask_rcnn_inception/frozen_inference_graph.pb'
class_names_path = './models/mask_rcnn_inception/class.names'

# Input and output video paths
input_video_path = './Leman2.mp4'
output_video_path = './output_videolemanNaOriginalnom.mp4'
output_video_maska_path = './output_videolemanMaska.mp4'

# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load the model
net = cv2.dnn.readNetFromTensorflow(weights_path, cfg_path)

# Create video writers for MP4
fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' or 'h264' are common choices for MP4
out = cv2.VideoWriter(output_video_path, fourcc_mp4, fps, (width, height))
outMaska = cv2.VideoWriter(output_video_maska_path, fourcc_mp4, fps, (width, height))

car_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
truck_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    H, W, C = frame.shape

    # Convert image
    blob = cv2.dnn.blobFromImage(frame)

    # Get masks
    boxes, masks = get_detections(net, blob)

    # Draw masks
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(90)]

    empty_img = np.zeros((H, W, C))

    detection_th = 0.2

    car_class_id = 2  # Assuming 'car' class ID is 2, change it if needed
    truck_class_id = 7

    for j in range(len(masks)):
        bbox = boxes[0, 0, j]

        class_id = bbox[1]
        score = bbox[2]

        if (class_id == car_class_id or class_id == truck_class_id) and score > detection_th:
            mask = masks[j]

            x1, y1, x2, y2 = int(bbox[3] * W), int(bbox[4] * H), int(bbox[5] * W), int(bbox[6] * H)

            mask = mask[int(car_class_id)]

            mask = cv2.resize(mask, (x2 - x1, y2 - y1))

            _, mask = cv2.threshold(mask, 0.3, 1, cv2.THRESH_BINARY)

            color = car_color if class_id == car_class_id else truck_color

            empty_img[y1:y2, x1:x2, :] = mask[:, :, np.newaxis] * color

            label = "Car" if class_id == car_class_id else "Truck"
            label_color = (255, 255, 255)  # White color for labels
            cv2.putText(empty_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2, cv2.LINE_AA)

    # Visualization
    overlay = ((0.5 * empty_img) + (0.5 * frame)).astype("uint8")

    # Write the frame to the output video
    out.write(overlay)  # Overlay video
    outMaska.write((empty_img * 255).astype(np.uint8))  # Mask video

    # Display the output video in a window
    cv2.imshow('Video', overlay)
    cv2.imshow('Mask', (empty_img * 255).astype(np.uint8))

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()
outMaska.release()

# Close all OpenCV windows
cv2.destroyAllWindows()