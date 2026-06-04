from ultralytics import YOLO
import cv2
import math

cap = cv2.VideoCapture(0)  # capture video from the default camera (0) using OpenCV

cap.set(3, 640)  # set the width of the video frame to 640 pixels
cap.set(4, 480)  # set the height of the video frame to 480 pixels

model = YOLO('Weights/yolov8n.pt')  # Load the YOLOv8n model from the specified path. This model is used for object detection in the video stream.

classNames = model.names  # Get the class names associated with the YOLO model. This allows us to identify the type of objects detected in the video stream based on their class IDs.   





while True: # Start an infinite loop to continuously read frames from the video stream and perform object detection on each frame.
    success, img = cap.read()  # Read a frame from the video stream. 'success' is a boolean indicating whether the frame was read successfully, and 'img' is the actual frame (image) captured from the video stream.
    results = model(img, stream=True)
    for r in results:  # Iterate through the results of the object detection. Each 'r' in 'results' contains information about the detected objects in the current frame.
        boxes = r.boxes # Extract the bounding boxes of the detected objects from the current result 'r'. The 'boxes' variable contains the coordinates and confidence scores of the detected objects.
        for box in boxes: # Iterate through each bounding box in 'boxes'. Each 'box' contains the coordinates of the bounding box and the confidence score for the detected object.
            x1, y1, x2, y2 = box.xyxy[0] # Extract the coordinates of the bounding box from 'box.xyxy[0]'. The coordinates are in the format (x1, y1, x2, y2), where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner of the bounding box.
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Convert the coordinates of the bounding box from floating-point numbers to integers. This is necessary because pixel coordinates must be integers when drawing on the image.
            cv2.rectangle(img, (x1, y1), (x2, y2), (50, 100, 10), 2) # Draw a rectangle (bounding box) on the image 'img' using the coordinates (x1, y1) and (x2, y2). The color of the rectangle is specified as (50, 100, 10) in BGR format, and the thickness of the rectangle border is set to 2 pixels.

            conf = math.ceil((box.conf[0] * 100)) / 100 # Calculate the confidence score for the detected object. The confidence score is obtained from 'box.conf[0]', which is multiplied by 100 to convert it to a percentage, and then rounded to two decimal places using 'math.ceil()' and division by 100.
            

            cls = int(box.cls[0]) # Extract the class ID of the detected object from 'box.cls[0]' and convert it to an integer. The class ID corresponds to the type of object detected (e.g., person, car, etc.).
            className = classNames[cls] # Get the class name associated with the class ID from the 'classNames' list.

            cv2.putText(img, f'{className} {conf}', (max(0, x1), max(35, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2) # Put the confidence score as text on the image 'img' at the position (x1, y1) using the specified font, font scale, color (255, 0, 255) in BGR format, and thickness of 2 pixels.




    cv2.imshow("Image", img) # Display the processed image 'img' in a window titled "Image". This window will show the video stream with the detected objects highlighted by bounding boxes and their confidence scores displayed as text.
    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'): # Wait for a key press for 1 millisecond and check if the 'q' key was pressed. If the 'q' key is pressed, the loop will break, allowing the program to exit gracefully.
        break

cap.release() # Release the video capture object to free up system resources. This is important to ensure that the camera is properly released and can be used by other applications after this program exits.
cv2.destroyAllWindows() # Close all OpenCV windows that were opened during the execution of the program. This is necessary to clean up the user interface and ensure that no windows remain open after the program has finished running.