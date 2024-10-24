
---

# Project Title: Motion Detection, Sentiment Analysis, and Gender Identification
### DATASET LINK : https://www.kaggle.com/datasets/trainingdatapro/gender-detection-and-classification-image-dataset?select=train 
## Table of Contents
1. [Objective](#objective)
2. [Tasks](#tasks)
   - [Task 1: Motion Estimation and Event Detection in a Video](#task-1-motion-estimation-and-event-detection-in-a-video)
   - [Task 2: Estimating Sentiments of People in a Crowd – Gesture Analysis and Image Categorization](#task-2-estimating-sentiments-of-people-in-a-crowd--gesture-analysis-and-image-categorization)
   - [Task 3: Gender Identification from Facial Features](#task-3-gender-identification-from-facial-features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Results](#results)
6. [License](#license)

## Objective
The objective of this project is to perform motion detection in videos, estimate sentiments of individuals in a crowd using gesture analysis, and identify gender based on facial features, all without relying on machine learning models.

## Tasks

### Task 1: Motion Estimation and Event Detection in a Video
#### Objective:
Detect motion and specific events in a video using frame differencing or optical flow to estimate motion and identify events without machine learning.

#### Task Description:
1. **Load Video:**
   ```python
   import cv2

   video_path = 'path_to_video.mp4'
   cap = cv2.VideoCapture(video_path)
   ```

2. **Motion Estimation:**
   ```python
   ret, prev_frame = cap.read()
   while cap.isOpened():
       ret, curr_frame = cap.read()
       diff = cv2.absdiff(prev_frame, curr_frame)
       ```

3. **Event Detection:**
   ```python
   threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]
   ```

4. **Result:**
   Visualize motion by highlighting moving regions in each frame.

### Task 2: Estimating Sentiments of People in a Crowd – Gesture Analysis and Image Categorization
#### Objective:
Estimate the sentiments of individuals in a crowd using basic gesture analysis techniques.

#### Task Description:
1. **Load Image Set:**
   ```python
   import os
   images = [cv2.imread(os.path.join('images', f)) for f in os.listdir('images')]
   ```

2. **Preprocessing:**
   ```python
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   ```

3. **Gesture Analysis:**
   ```python
   # Facial feature extraction and emotion classification
   if smile_ratio > 1.3:
       emotion = "Happy"
   ```

4. **Image Categorization:**
   Output the overall sentiment of the crowd.

### Task 3: Gender Identification from Facial Features
#### Objective:
Identify the gender of individuals based on facial features using traditional image processing techniques.

#### Task Description:
1. **Load Dataset:**
   ```python
   dataset = [cv2.imread(os.path.join('dataset', f)) for f in os.listdir('dataset')]
   ```

2. **Preprocessing:**
   ```python
   face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
   faces = face_cascade.detectMultiScale(gray)
   ```

3. **Feature Extraction:**
   ```python
   # Calculate geometric features
   jaw_width = calculate_jaw_width(face)
   ```

4. **Rule-Based Gender Identification:**
   ```python
   if jaw_width > threshold:
       gender = "Male"
   else:
       gender = "Female"
   ```



## Results
- Visual outputs of motion detection in videos.
- Sentiment analysis results for individuals in images.
- Gender identification results from the facial dataset.

