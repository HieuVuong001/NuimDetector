from ultralytics import YOLO

model = YOLO('/home/014105936/scripts/yolov8m.pt')

model.train(data='/home/014105936/yolov5/nuimages.yaml', time=24, name='yolov8_pretrained', batch=-1, amp=False)

model.val()
