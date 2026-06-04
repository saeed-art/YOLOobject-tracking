from ultralytics import YOLO # Import the YOLO class from the ultralytics library, which is used for object detection.
import cv2 # Import the OpenCV library, which is used for image and video processing.

model = YOLO('Weights/yolov8n.pt') # Load the YOLOv8n model from the specified path. This model is used for object detection in images and videos.
results = model('Images/cars.jpg') # Perform object detection on the image located at 'Images/cars.jpg' using the loaded YOLO model. The results of the detection are stored in the 'results' variable, which contains information about the detected objects in the image.
cv2.imshow('Image', results[0].plot()) # Display the processed image with detected objects highlighted in a window titled "Image". The 'results[0].plot()' function is used to visualize the detection results on the image, showing bounding boxes and labels for the detected objects.
cv2.waitKey(0) # Wait indefinitely for a key press. This allows the displayed image to remain on the screen until the user presses any key, at which point the program will continue execution (or exit if there are no further instructions).