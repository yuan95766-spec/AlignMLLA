# YOLOV8多模态目标检测
## 前言：环境配置要求
torch                        2.7.1\
torchvision                  0.22.1\
Python 3.12.9\.
```
## 1. 数据集DroneVehicle数据集(可见光+热红外)
[DroneVehicle数据集下载地址](https://github.com/VisDrone/DroneVehicle) 

DroneVehicle 数据集由无人机采集的 56,878 张图像组成，其中一半是 RGB 图像，其余是红外图像。我们为这 5 个类别制作了丰富的注释，其中包含定向边界框。其中，汽车在 RGB 图像中有 389,779 个注释，在红外图像中有 428,086 个注释，卡车在 RGB 图像中有 22,123 个注释，在红外图像中有 25,960 个注释，公共汽车在 RGB 图像中有 15,333 个注释，在红外图像中有 16,590 个注释，厢式车在 RGB 图像中有 11,935 个注释，在红外图像中有 12,708 个注释，货车在 RGB 图像中有 13,400 个注释， 以及红外图像中的 17,173 个注释。\

## 2. 数据集文件格式(labeles: YOLO格式)
```
datasets
├── image
│   ├── test
│   ├── train
│   └── val
├── images
│   ├── test
│   ├── train
│   └── val
└── labels
    ├── test
    ├── train
    └── val
```
images 保存的是可见光图片\
image 保存的是热红外图片\
labels 公用一个标签(一般来说使用红外图片标签)

## 3. 配置模型yaml文件和数据集yaml文件
分别在yaml文件夹和data文件下进行模型和数据集文件的配置

## 4. 训练
```
python train.py

windows直接运行可能报多进程错误，请运行python train_for_windows.py

```

## 5. 测试
```
python test.py  
```
## 6. 打印模型信息
```
python info.py  
```
## 7. obb
推理：detect/obbDetect.py \
热图绘制: detect/hbbHeapmap.py\
onnx推理: detect/obbOnnxdetect.py
## 8. hbb
推理：detect/hbbDetect.py \
热图绘制: detect/hbbHeapmap.py
