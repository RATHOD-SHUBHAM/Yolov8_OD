import math

from ultralytics import YOLO
import cv2

# 0 - for webcam
cap = cv2.VideoCapture('ip_video/smallVideo.mp4')

# get the frame size
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

model = YOLO("preTrained_weight/yolov8x.pt")

# number of classes
nc: 80

# class names
names= [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

while True:
    # perform detection frame by frame
    success, img = cap.read()

    # stream = True will use the generator and it is more effecient than nomal
    results = model(img, stream=True)

    # once we have the result we can check for individual bounding boxes and see how well it will perform.
    # We will loop through the result and then extract BBOX
    # We will loop through BBOX

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1 , y1, x2 , y2 = box.xyxy[0]

            x1, y1, x2 , y2 = int(x1) , int(y1) , int(x2) , int(y2)

            print(x1, y1, x2, y2)

            cv2.rectangle(img, (x1,y1), (x2, y2), (242, 140, 40), 3)

            conf = math.ceil((box.conf[0] * 100)) / 100

            cls = int(box.cls[0])

            class_name = names[cls]

            label = f'{class_name}, {conf}'

            t_size = cv2.getTextSize(label , 0 , fontScale= 1, thickness= 2)[0]

            c2 = x1 + t_size[0], y1 - t_size[1] - 3

            cv2.rectangle(img, (x1, y1), c2, (242, 140, 40), -1, cv2.LINE_AA)

            cv2.putText(img, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    out.write(img)

    cv2.imshow("Image", img)



    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

out.release()


