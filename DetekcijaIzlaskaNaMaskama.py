import cv2
import numpy as np

# Open the input videosjetbrains://pycharm/navigate/reference?project=Outside_segmentation.h5&path=semantic-segmentation-tensorflow-opencv-master/output_videolemanMaska.mp4
#video1 = cv2.VideoCapture('./semantic-segmentation-tensorflow-opencv-master/output_videolemanMaska.mp4')
#video1 = cv2.VideoCapture('C:/Users/Aca/PycharmProject2/pythonProject 17.1.2024 - IzmenePosleSugestija/semantic-segmentation-tensorflow-opencv-master\output_videolemanMaska.mp4')
video2 = cv2.VideoCapture('./output_videolmstaza.mp4')
video1 = cv2.VideoCapture('./output_videolemanMaska.mp4')

# Get video properties (assuming both videos have the same resolution and frame rate)
width = int(video1.get(3))
height = int(video1.get(4))
fps = int(video1.get(5))

# Define the codec and create VideoWriter object
output_video = cv2.VideoWriter('output_combined_highlight.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while True:
    # Read frames from the input videos
    ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()

    # Break the loop if the frames are not read successfully
    if not ret1 or not ret2:
        break

    # Resize frames to have the same dimensions
    frame1 = cv2.resize(frame1, (width, height))
    frame2 = cv2.resize(frame2, (width, height))

    # Create a mask of the intersection
    intersection_mask = cv2.bitwise_or(frame1, frame2)

    # Write the combined frame to the output video
    output_video.write(intersection_mask)

    # Display the combined frame
    cv2.imshow('Combined Video', intersection_mask)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release video captures and writer
video1.release()
video2.release()
output_video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()


