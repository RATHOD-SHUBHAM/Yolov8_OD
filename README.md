# Yolov8

Anchor Free Detection
YOLOv8 is an anchor-free model. This means it predicts directly the center of an object instead of the offset from a known anchor box.


Visualization of an anchor box in YOLO
Anchor boxes were a notoriously tricky part of earlier YOLO models, since they may represent the distribution of the target benchmark's boxes but not the distribution of the custom dataset.


The detection head of YOLOv5, visualized in netron.app
Anchor free detection reduces the number of box predictions, which speeds up Non-Maximum Suppression (NMS), a complicated post processing step that sifts through candidate detections after inference.


The detection head for YOLOv8, visualized in netron.app
