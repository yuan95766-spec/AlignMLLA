
## Preface: Environment Requirements
- **PyTorch**: 2.7.1  
- **Torchvision**: 0.22.1  
- **Python**: 3.12.9  

---

## 1. Dataset: DroneVehicle (Visible RGB + Thermal Infrared)

**Download**: [DroneVehicle Dataset](https://github.com/VisDrone/DroneVehicle)

The DroneVehicle dataset contains **56,878** UAV-captured images, including **RGB** images and **infrared (thermal)** images (roughly half each). It provides rich annotations for **5 categories** with **oriented bounding boxes (OBB)**.

Annotation statistics:
- **Car**: 389,779 (RGB) / 428,086 (Infrared)
- **Truck**: 22,123 (RGB) / 25,960 (Infrared)
- **Bus**: 15,333 (RGB) / 16,590 (Infrared)
- **Van**: 11,935 (RGB) / 12,708 (Infrared)
- **Freight car / Cargo vehicle**: 13,400 (RGB) / 17,173 (Infrared)

---

## 2. Dataset Directory Structure (YOLO Labels)

Labels are in **YOLO format**:

```text
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

- `images/` stores **visible RGB** images  
- `image/` stores **thermal infrared** images  
- `labels/` is **shared** for both modalities (typically using the infrared labels)

---

## 3. Configure YAML Files

- Configure the **model YAML** under the `yaml/` directory  
- Configure the **dataset YAML** under the `data/` directory  

---

## 4. Training

```bash
python train.py
```

On **Windows**, direct execution may raise multiprocessing-related errors. Use:

```bash
python train_for_windows.py
```

---

## 5. Testing

```bash
python test.py
```

---

## 6. Print Model Information

```bash
python info.py
```

---

## 7. HBB (Horizontal Bounding Box)

- Inference: `detect/hbbDetect.py`  
- Heatmap visualization: `detect/hbbHeapmap.py`
```
