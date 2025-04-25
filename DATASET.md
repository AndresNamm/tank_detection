Dataset Description
This dataset is designed for object detection, specifically to identify and localize tanks in images using bounding boxes.

📦 Dataset Overview
Total Images: 1,729
Training Set: 1,296 images (80%)
Test Set: 433 images (20%)
The first half of the test set is used for public evaluation, the second half is used for private (final leaderboard) evaluation.
Source: Extracted from 3 different videos
Format: .jpg images (~500 kB each)
Annotations: Stored in annotations.json, including:
An id
Segmentation polygon for each element in each image
💡 A sample notebook is provided with a snippet to convert segmentation points into bounding boxes.

Images have been shuffled and split into training and testing sets using an 80/20 ratio. Each image may contain one or more tanks. The dataset includes only a single object class: tanks, labeled as 0.

🗂️ Directory Structure
train/
  ├── frames/
  └── annotations.json
test/
  └── frames/
🎯 Task Objective
The objective is to detect tanks in the images by drawing bounding boxes around them. Your model's performance will be evaluated using Intersection over Union (IoU).

📤 Submission Format
Submissions should be in a semicolon-delimited CSV file (.csv) with the following columns:

image_id	x_min	y_min	x_max	y_max	class_id
If multiple tanks are detected in a single image, their bounding box values should appear consecutively in the same row, like this:

123; 45; 60; 200; 220; 230; 80; 360; 240; 0