# YOLOv8-Improved-2025
This is an unofficial implementation of the paper titled "Improvement of Yolov8 Object Detection Based on Lightweight Neck Model for Complex Images"

<hr/>

#### Model Architecture
<br/>
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

<br/>
<br/>

#### Model Variants
- Nano (n)
- Small (s)
- Medium (m)
- Large (l)
- Extra Large (x)

<br/>
<br/>

#### Model Comparisons

##### Detection Model
<br/>
<img width="1862" height="329" alt="image" src="https://github.com/user-attachments/assets/d7365951-b9f7-4252-b00b-8c357b7a20c8" />

<br/>
<br/>

##### Segmentation Model
<br/>
<img width="1867" height="328" alt="image" src="https://github.com/user-attachments/assets/d6dd7707-26e1-4f6e-999d-08997a506bc9" />
