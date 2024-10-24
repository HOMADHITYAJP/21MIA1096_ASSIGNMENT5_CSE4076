# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 06:07:18 2024

@author: homap
"""

import cv2
import dlib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define paths
dataset_path = 'gender'
crop_men_path = os.path.join(dataset_path, 'crop_men')  # Crop men folder
crop_women_path = os.path.join(dataset_path, 'crop_women')  # Crop women folder

# Load dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to display images
def display_images(crop_men_image_path, crop_women_image_path):
    # Load images
    crop_men_image = cv2.imread(crop_men_image_path)
    crop_women_image = cv2.imread(crop_women_image_path)

    # Convert BGR to RGB for displaying with matplotlib
    crop_men_image = cv2.cvtColor(crop_men_image, cv2.COLOR_BGR2RGB)
    crop_women_image = cv2.cvtColor(crop_women_image, cv2.COLOR_BGR2RGB)

    # Display images
    plt.figure(figsize=(8, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(crop_men_image)
    plt.title('Cropped Men Image')

    plt.subplot(1, 2, 2)
    plt.imshow(crop_women_image)
    plt.title('Cropped Women Image')

    plt.tight_layout()
    plt.show()

# Function to extract geometric features
def extract_geometric_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)

    if len(dets) > 0:
        shape = predictor(gray, dets[0])
        jaw_width = shape.part(16).x - shape.part(0).x  # Jaw width
        eye_distance = shape.part(42).x - shape.part(39).x  # Distance between eyes
        nose_width = shape.part(35).x - shape.part(31).x  # Nose width
        mouth_width = shape.part(54).x - shape.part(48).x  # Mouth width
        return {
            "jaw_width": jaw_width,
            "eye_distance": eye_distance,
            "nose_width": nose_width,
            "mouth_width": mouth_width,
        }
    return None

# Function to extract texture features using edge detection
def extract_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
    edges_canny = cv2.Canny(gray, 100, 200)

    return {
        "edges_sobel": np.sum(edges_sobel),
        "edges_canny": np.sum(edges_canny),
    }

# Function to process and extract features from images
def process_images(folder_path):
    features_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            # Extract geometric features
            geom_features = extract_geometric_features(image)
            # Extract texture features
            text_features = extract_texture_features(image)

            if geom_features and text_features:
                features = {
                    "filename": filename,
                    **geom_features,
                    **text_features,
                }
                features_list.append(features)
    return features_list

# Display images
display_images(
    os.path.join(crop_men_path, '0.jpg'),
    os.path.join(crop_women_path, '0.jpg')
)

# Extract features for men and women from cropped images
men_features = process_images(crop_men_path)
women_features = process_images(crop_women_path)

# Save features to CSV
men_df = pd.DataFrame(men_features)
women_df = pd.DataFrame(women_features)
men_df.to_csv('men.csv', index=False)
women_df.to_csv('women.csv', index=False)

# Save CSVs to Excel
with pd.ExcelWriter('gender_features.xlsx') as writer:
    men_df.to_excel(writer, sheet_name='Men Features', index=False)
    women_df.to_excel(writer, sheet_name='Women Features', index=False)

print("Feature extraction completed and saved to 'men.csv', 'women.csv', and 'gender_features.xlsx'.")
