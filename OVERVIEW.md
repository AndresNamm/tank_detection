Overview
“The eyes of the future are upon you. You are the tip of the spear—the code that strikes where silence reigns.”

🪖Welcome, warriors of machine vision.

This isn’t a toy problem, and it’s not just another leaderboard grind. This is a real-world sub-problem with real implications. Your task: build a machine learning model that can detect tanks in images captured by Ukrainian kamikaze drones—footage taken just moments before impact.

You’ll be working with raw, shaky, imperfect visuals—just like the kind a drone sees in its final approach. Your model needs to recognize tanks, draw bounding boxes, and do it under a variety of conditions: poor lighting, low altitude, harsh terrain.

Now, let’s be clear. This model won’t fly the drone for you. It’s not a full autonomy stack. But it could become one crucial piece in a larger system—something that helps guide the drone in the final seconds when human control is lost due to signal jamming or distance. Your work here might one day power that last line of perception.

“No one ever won a war by waiting for perfect data. You fight with what you have, and you make it count.”

So: study the Description below to understand the mission and terrain. Read the Rules to avoid disqualification. And remember—this is about more than just getting the highest score. It’s about solving a hard, messy problem with potential real-world consequences.

Start

12 days ago
Close
16 days to go
Description
1. Situation:
"Welcome to the jungle, warriors!" In this competition, participants are tasked with developing machine learning models to detect and identify tanks in drone footage. The dataset is comprised of images extracted from videos taken by Ukrainian kamikaze drones as they approach their target destination. These open-source videos provide a unique perspective, capturing the final moments before impact.

2. Mission:
Your mission, should you choose to accept it, is to create a machine learning model that can:

Detect tanks from the drone's point of view with high accuracy.
Draw bounding boxes around each tank in the images extracted from drone footage.
Handle diverse conditions such as varying lighting, weather, and terrain.
In the future, this model could potentially be used to help guide drones autonomously, but for this competition, the focus is on accurate tank detection.

3. Execution:
Participants will:

Develop models to detect and classify tanks in drone footage.
Handle diverse conditions such as varying lighting, weather, and terrain.
Enable autonomous guidance for drones in the final meters of their approach.
Submit solutions through the Kaggle competition page, including detailed descriptions of methodologies.
4. Administration and Logistics:
Who Can Participate: Open to students and teachers of the Deep Learning for Computer Vision course at the University of Tartu.
How to Participate: Register on the Kaggle platform, download the dataset, and start developing your model.
Resources: Reach out to the competition organizers for any questions or assistance.
5. Command and Signal:
Evaluation Criteria: Models will be evaluated based on the Intersection over Union (IoU) score between the predicted bounding boxes and ground truth bounding boxes.

Communication: The organizers, team Image Innovators, can be reached on Slack in the DP4CV workspace.

Organizers:

Gustav Nikopensius
Simon Idoko
Marie-Johanna Perli
Mark-Erik Aan
For detailed rules and guidelines, please refer to the competition rules section.

Evaluation
Submissions are evaluated on Interesction over Union (IoU) aka the Jaccard Index between the predicted bounding box and the ground truth bounding box.

Submission File
The submission file format is specified under Data/Submission Format

Sample Notebook
Link to Sample Notebook with:

Image and target visualization
Bounding Box extraction script
Random Submission Generator
Citation
Gustav Nikopensius UT, Marie-Johanna Perli, Mark-Erik Aan, and Simon Idoko. DroneVision: Tank Detection from Aerial Videos. https://kaggle.com/competitions/image-innovators, 2025. Kaggle.