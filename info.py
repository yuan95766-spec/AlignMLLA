# 查看模型信息

from ultralytics import YOLO
model = YOLO('/home/yuan/my_project/TwoStream_Yolov8-main/TwoStream_Yolov8-main/yaml/AlignMLLA.yaml')
model.info()