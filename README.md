# YOLOv8-Improved-2025
This is an unofficial implementation of the paper titled "Improvement of Yolov8 Object Detection Based on Lightweight Neck Model for Complex Images" (https://www.ias-iss.org/ojs/IAS/article/view/3514)

<hr/>

### Model Architecture
<br/>
<img width="883" height="487" alt="image" src="https://github.com/user-attachments/assets/f9997942-5e84-4724-a884-2742108da6ae" />

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

### Model Variants
- Nano (n)
- Small (s)
- Medium (m)
- Large (l)
- Extra Large (x)

<br/>
<br/>

### Model Usage
```
from model import YoloV8I, YoloV8I_CONFIGS

# Model
model_type = "m"
model_config = YoloV8I_CONFIGS[model_type]
model_config.num_classes = 80

# Detection
model = YoloV8I(model_config, task="det")

# Segmentation
model = YoloV8I(model_config, task="seg")

# Export
model = YoloV8I(model_config, task="seg", export=True)
```

<br/>
<br/>

### Model Comparisons
<br/>
Note: These profiling were performed on RTX 3080 with num_classes = 80 and batch_size = 16; with export=True
<br/>

#### Detection Model
<img width="1449" height="323" alt="image" src="https://github.com/user-attachments/assets/a7f393bf-6e7a-4318-a953-a0480e738d4e" />


<br/>
<br/>

#### Segmentation Model
<img width="1457" height="329" alt="image" src="https://github.com/user-attachments/assets/f42ae626-b246-457f-b15e-796042e260ee" />

