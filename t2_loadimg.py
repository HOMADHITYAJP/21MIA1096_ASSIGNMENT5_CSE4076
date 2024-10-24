# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 19:05:19 2024

@author: homap
"""

import cv2
import os

# Path to the image directory
image_dir = 'sampleimg'
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Load and display images
loaded_images = []

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)
    loaded_images.append(image)

    # Display the loaded image
    cv2.imshow(f"Image: {image_file}", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()

# Save loaded images for further processing
for idx, img in enumerate(loaded_images):
    cv2.imwrite(f'crowd_image_{idx}.jpg', img)

print(f"Total {len(loaded_images)} images loaded and saved.")