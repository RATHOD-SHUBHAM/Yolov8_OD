# Yolov8

Anchor Free Detection
YOLOv8 is an anchor-free model. This means it predicts directly the center of an object instead of the offset from a known anchor box.

![anchor_box](https://github.com/RATHOD-SHUBHAM/Yolov8_OD/assets/58945964/2c3034dc-fcbd-4965-821e-1f98944bebc2)



Visualization of an anchor box in YOLO
Anchor boxes were a notoriously tricky part of earlier YOLO models, since they may represent the distribution of the target benchmark's boxes but not the distribution of the custom dataset.

<img width="672" alt="image-15" src="https://github.com/RATHOD-SHUBHAM/Yolov8_OD/assets/58945964/197465d1-d3f2-44b6-9c62-fadb58616fd3">



The detection head of YOLOv5, visualized in netron.app
Anchor free detection reduces the number of box predictions, which speeds up Non-Maximum Suppression (NMS), a complicated post processing step that sifts through candidate detections after inference.


The detection head for YOLOv8, visualized in netron.app
