#训练
from multiprocessing import freeze_support
from ultralytics import YOLO

import ultralytics.nn.tasks
def main():
    model = YOLO(r'D:\Paper\TwoStream_Yolov8-main\TwoStream_Yolov8-main\yaml\PC2f_MPF_yolov8n.yaml')
    results = model.train(data=r'D:\Paper\TwoStream_Yolov8-main\TwoStream_Yolov8-main\data\drone2.yaml',batch=2,epochs=100,workers=1)
   # model = YOLO(r'D:\Paper\TwoStream_Yolov8-main\TwoStream_Yolov8-main\runs\detect\train2\weights\last.pt')
   # results = model.train(resume=True)


if __name__ == "__main__":
    freeze_support()  # 加上这一句,防止windows环境下的多进程报错
    main()
