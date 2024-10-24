# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 18:06:01 2024

@author: homap
"""

#pip install opencv-python
#pip install numpy

import cv2
import numpy as np

# Load the video
video_path = "sample.mp4"  # Change this to your video path
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Variables to store frames
ret, frame1 = cap.read()
fps = cap.get(cv2.CAP_PROP_FPS)
interval = int(fps)  # For 1 FPS, we take 1 second interval
frame_count = 0

# Parameters for event detection
movement_distance_threshold = 10  # Lower distance threshold for significant motion
previous_boxes = []  # Store previous bounding boxes

while True:
    # Get the next frame
    ret, frame2 = cap.read()
    if not ret:
        break  # End of video

    # Process frames only at defined intervals
    frame_count += 1
    if frame_count % interval == 0:
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate the absolute difference between the frames
        diff = cv2.absdiff(gray1, gray2)

        # Apply a threshold to the difference image
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Find contours of the moving objects
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # List to store bounding boxes for the current second
        boxes = []
        significant_event = False  # Flag to check if a significant event occurs

        for contour in contours:
            motion_area = cv2.contourArea(contour)
            if motion_area > 500:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append((x, y, w, h))

        # Check for significant movement compared to previous frame
        for (x, y, w, h) in boxes:
            # Check if there are previous boxes to compare with
            if previous_boxes:
                # Calculate the center of the current bounding box
                current_center = (x + w // 2, y + h // 2)

                # Find the closest previous box based on position
                closest_distance = float('inf')
                for (px, py, pw, ph) in previous_boxes:
                    previous_center = (px + pw // 2, py + ph // 2)

                    # Calculate Euclidean distance between centers
                    distance = np.sqrt((current_center[0] - previous_center[0]) ** 2 + 
                                       (current_center[1] - previous_center[1]) ** 2)

                    # Update the closest distance
                    if distance < closest_distance:
                        closest_distance = distance

                # Check if the movement exceeds the defined threshold
                if closest_distance > movement_distance_threshold:
                    significant_event = True  # Set flag if we have significant movement

        # Draw bounding boxes on the frame
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame with bounding boxes
        if significant_event:
            # Add "EVENT DETECTED!" annotation
            cv2.putText(frame2, "EVENT DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Calculate timestamp
            timestamp = frame_count // fps
            seconds = int(timestamp % 60)  # Seconds within the minute
            minutes = int(timestamp // 60)  # Total minutes

            # Add timestamp annotation below the event text
            cv2.putText(frame2, f"Timestamp: {minutes}:{seconds:02d}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Motion Detection', frame2)
        cv2.imshow('Threshold', thresh)

        # Update the previous boxes with the current boxes for the next iteration
        previous_boxes = boxes

        # Move the first frame to the current frame for the next iteration
        frame1 = frame2.copy()

    # Exit if 'q' is pressed
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
