# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 05:07:18 2024

@author: homap
"""


import cv2
import os

# Define paths
dataset_path = 'gender'  # The main directory containing 'men' and 'women' folders
men_path = os.path.join(dataset_path, 'men')
women_path = os.path.join(dataset_path, 'women')
crop_men_path = os.path.join(dataset_path, 'crop_men')
crop_women_path = os.path.join(dataset_path, 'crop_women')

# Create directories for cropped images if they do not exist
os.makedirs(crop_men_path, exist_ok=True)
os.makedirs(crop_women_path, exist_ok=True)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to process images in a given directory
def process_images(folder_path, crop_folder):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Crop and save each detected face
            for (x, y, w, h) in faces:
                face = image[y:y+h, x:x+w]  # Crop the face
                normalized_face = cv2.resize(face, (200, 200))  # Normalize the face size
                cropped_image_path = os.path.join(crop_folder, filename)
                cv2.imwrite(cropped_image_path, normalized_face)  # Save the cropped image

# Process images for men and women
process_images(men_path, crop_men_path)
process_images(women_path, crop_women_path)

print("Face detection and cropping completed. Cropped images are saved in 'crop_men' and 'crop_women' folders.")
