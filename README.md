# YOLOv8-Improved-2025
This is an unofficial implementation of the paper titled "Improvement of Yolov8 Object Detection Based on Lightweight Neck Model for Complex Images"

<hr/>

#### Model Architecture

<img width="1301" height="705" alt="image" src="https://github.com/user-attachments/assets/07f499c7-4d68-468e-833b-b821cdb2b571" />

<br/>
<br/>

🔍 Key improvements over YOLOv8:
- Higher mAP on benchmark datasets
- Faster inference times
- Smarter architectural tweaks that enhance feature learning
- Designed for complex scenes such as target occlusion and low-contrast scenes.

<br/>
<br/>

🧱 Key Blocks:
- Ghost Convolution -> more features through cheaper operations
- Lightweight Depth Ghost Convolution -> makes the Bottleneck lighter
- Channel Shuffle -> improves the richness of the features
- Coordinate Attention -> pays attention to every pixel
