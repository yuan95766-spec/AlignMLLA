#训练
from ultralytics import YOLO
import ultralytics.nn.tasks
model = YOLO('/home/yuan/my_project/TwoStream_Yolov8-main/TwoStream_Yolov8-main/yaml/MLLA.yaml')
results = model.train(data='/home/yuan/my_project/TwoStream_Yolov8-main/TwoStream_Yolov8-main/data/drone2.yaml',resume=True,batch=2,epochs=200,workers=1)
