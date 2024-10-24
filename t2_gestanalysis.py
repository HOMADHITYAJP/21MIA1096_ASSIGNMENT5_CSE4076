# -*- coding: utf-8 -*-
"""
3Created on Wed Oct 23 05:30:13 2024

@author: homap
"""
import cv2
import numpy as np
import os

# Path to the input image directory
input_directory = 'sampleimages/'  # Use the absolute path if necessary
output_directory = 'output_image/'  # Output directory for results

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Load Haar Cascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('cascades/haarcascade_mcs_mouth.xml')

# Function to classify emotions based on facial features
def classify_emotion(face_region, original_image, x, y):
    eyes = eye_cascade.detectMultiScale(face_region)
    mouths = mouth_cascade.detectMultiScale(face_region)

    emotion = "Neutral"  # Default emotion

    if len(mouths) > 0:
        (mx, my, mw, mh) = mouths[0]  # Assume the first detected mouth
        smile_ratio = mw / mh

        if smile_ratio > 1.3:
            emotion = "Happy"  # "Happy" detected
            # Annotate "mouth has upper curvature" on the original image with black color
            cv2.putText(original_image, "Mouth has upper curvature", (x + mx, y + my + mh + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Thicker black color (0, 0, 0), size 0.5, thickness 2
        elif smile_ratio < 0.8:
            emotion = "Sad"

    return emotion

# Function to detect hands and classify gestures
def detect_hands_and_gestures(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Improved skin color range for detecting hand regions
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)  # Adjusted skin color range
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours of the skin regions
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)

        # Area thresholds to filter hand size
        if 500 < area < 12000:  # Adjusted area threshold for hand detection
            x, y, w, h = cv2.boundingRect(contour)

            # Additional check to ignore empty spaces or small contours
            if cv2.contourArea(contour) > 1000:  # Minimum area to consider as valid gesture
                # Draw the bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # Create a convex hull to approximate the hand shape
                hull = cv2.convexHull(contour)
                cv2.drawContours(image, [hull], -1, (0, 255, 0), 2)

                # Calculate aspect ratio and height to width ratio for thumbs up gesture
                aspect_ratio = float(w) / h
                height_ratio = h / float(image.shape[0])

                # Check for a thumbs-up gesture based on thumb position
                if aspect_ratio < 1.5 and aspect_ratio > 0.5 and height_ratio > 0.25:
                    cv2.putText(image, "Thumbs Up", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return image

# Process each image in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        input_image_path = os.path.join(input_directory, filename)

        # Load the image
        image = cv2.imread(input_image_path)

        # Check if the image was loaded successfully
        if image is None:
            print(f"Error: Could not load image from {input_image_path}. Please check the file path.")
            continue

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))

        # Initialize counts for emotions
        happy_count = 0
        neutral_count = 0
        sad_count = 0

        # Detect hands and gestures
        image_with_hands = detect_hands_and_gestures(image)

        # Process and annotate detected faces with emotions
        for (x, y, w, h) in faces:
            face_region = gray[y:y + h, x:x + w]
            emotion = classify_emotion(face_region, image_with_hands, x, y)

            # Count emotions
            if emotion == "Happy":
                happy_count += 1
            elif emotion == "Sad":
                sad_count += 1
            else:
                neutral_count += 1

            cv2.rectangle(image_with_hands, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image_with_hands, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Determine overall sentiment
        if happy_count > sad_count:
            overall_sentiment = "Happy"
        elif sad_count > happy_count:
            overall_sentiment = "Sad"
        else:
            overall_sentiment = "Neutral"

        # Annotate overall sentiment on the image
        cv2.putText(image_with_hands, f"OVERALL SENTIMENT: {overall_sentiment}",
                    (image.shape[1] - 350, image.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        # Save the output image
        output_image_path = os.path.join(output_directory, f'detected_{filename}')
        cv2.imwrite(output_image_path, image_with_hands)

        print(f"Processed {filename}. Output saved as {output_image_path}.")

# Optionally display the last processed image
# cv2.imshow("Detected Faces, Hands, and Emotions", image_with_hands)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
