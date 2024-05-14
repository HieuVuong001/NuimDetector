from ultralytics import YOLO

model = YOLO('/home/014105936/scripts/yolov8m.yaml')

model.train(data='/home/014105936/yolov5/nuimages.yaml', epochs=20, pretrained=False, name='yolov8_yaml', batch=-1, amp=False)

model.val()
