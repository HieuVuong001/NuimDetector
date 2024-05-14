from ultralytics import YOLO

model = YOLO('/home/014105936/scripts/yolov8m.pt')

model.train(data='/home/014105936/yolov5/nuimages.yaml', epochs=20, name='yolov8_pretrained_AdamW', batch=-1, amp=False, optimizer='AdamW')

model.val()
