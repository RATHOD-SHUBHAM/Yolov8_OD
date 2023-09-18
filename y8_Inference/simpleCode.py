import ultralytics
ultralytics.checks()

from ultralytics import YOLO

# Pretrained weight
# model = YOLO("preTrained_weight/yolov8x.pt")

# Custom weight
model = YOLO('DetectionWeight/best.pt')

source = "ip_video/New Folder With Items/ip1.mp4"

result = model.predict(source = source, conf = 0.35, save=True)

model.export(format="onnx",opset=12)  # export the model to ONNX format

# model.export(format="ncnn")  # export the model to ncnn format

# model.export(format="tflite", imgsz = 416)  # export the model to tflite format