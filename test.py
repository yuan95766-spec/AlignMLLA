# 测试
from ultralytics import YOLO 
model = YOLO('/home/yuan/my_project/TwoStream_Yolov8-main/TwoStream_Yolov8-main/runs/detect/train_Align/train30/weights/best.pt') 
metrics = model.val(data='/home/yuan/my_project/TwoStream_Yolov8-main/TwoStream_Yolov8-main/data/drone2.yaml',split='test',imgsz=640,batch=2,workers=1)
